module KernelSelection

using ..Utils
using ..WaveletOps
using Plots
using StatsBase
using Distributions

export wavelet_smooth_mean_function, create_wavelet_comparison_plot,
       posterior_summaries_from_samples, score_wavelet_candidate, select_wavelet

"""
    wavelet_smooth_mean_function(mean_function, wf, J, boundary)

Apply MCMC-based wavelet smoothing to the mean function using Besov spike-and-slab priors.
Runs a separate MCMC algorithm with iter=3000, burnin=1000, thin=1.
"""
function wavelet_smooth_mean_function(Y_mats::Vector{Matrix{Float64}}, revealed_idx::Vector{Int}, wf::String, J::Int, boundary::String;
                                      return_beta::Bool=false, n_iter_selection::Int=3000, burnin_selection::Int=1000, thin_selection::Int=1)
    # Compute cluster mean from all revealed samples (this will be the blue curve)
    P, M = size(Y_mats[1])
    cluster_mean = zeros(P, M)
    
    for idx in revealed_idx
        cluster_mean .+= Y_mats[idx]
    end
    cluster_mean ./= length(revealed_idx)
    
    # MCMC parameters for wavelet selection
    # n_iter_selection = 3000
    # burnin_selection = 1000
    # thin_selection = 1
    
    # Run MCMC on the cluster mean itself (not individual samples)
    Y_list = [cluster_mean]
    
    # Initialize wavelet parameters properly
    # First, get wavelet level names by doing a test transform
    test_wt = WaveletOps.wt_forward_1d(cluster_mean[:, 1]; wf=wf, J=J, boundary=boundary)
    lev_names = [String(k) for k in keys(test_wt.map.idx)]
    det_names = filter(name -> startswith(name, "d"), lev_names)
    
    # Initialize WaveletParams with proper structure
    pi_level = Dict(name => 0.5 for name in det_names)
    g_level = Dict(name => 2.0 for name in det_names)
    ncoeff = length(test_wt.coeff)
    gamma_ch = [Int.(Base.rand(Bernoulli(0.2), ncoeff)) for _ in 1:M]
    
    # Hyperparameters for WaveletParams (matching those used in update_cluster_wavelet_params_besov)
    kappa_pi = 1.0
    c2 = 1.0
    tau_pi = 40.0
    a_g = 2.0
    b_g = 2.0
    a_sig = 2.5
    b_sig = 0.02
    a_tau = 2.0
    b_tau = 2.0
    
    wpar = Utils.WaveletParams(lev_names, pi_level, g_level, gamma_ch,
                               kappa_pi, c2, tau_pi,
                               a_g, b_g,
                               a_sig, b_sig, a_tau, b_tau)
    
    # Initialize sigma2 and tau_sigma
    sigma2_m = [1.0 for _ in 1:M]
    tau_sigma = 1.0
    
    # Precompute wavelets for the cluster mean
    precomp = WaveletOps.precompute_wavelets(Y_list, wf, J, boundary)
    
    # Run MCMC for wavelet smoothing on the cluster mean
    println("    Running MCMC for wavelet $wf (iter=$n_iter_selection, burnin=$burnin_selection)...")
    
    # Store samples
    keep = max(0, Int(floor((n_iter_selection - burnin_selection) / thin_selection)))
    beta_samples = []
    
    for iter in 1:n_iter_selection
        # Update wavelet parameters using Besov spike-and-slab MCMC on the cluster mean
        upd = WaveletOps.update_cluster_wavelet_params_besov(
            [1], precomp, M, wpar, sigma2_m, tau_sigma;
            kappa_pi = 1.0, c2 = 1.0, tau_pi = 40.0,
            g_hyp = nothing,
            a_sig = 2.5, b_sig = 0.02,
            a_tau = 2.0, b_tau = 2.0
        )
        
        wpar = upd.wpar
        sigma2_m = upd.sigma2_m
        tau_sigma = upd.tau_sigma
        
        # Store samples after burn-in
        if iter > burnin_selection && ((iter - burnin_selection) % thin_selection == 0)
            push!(beta_samples, copy(upd.beta_ch))
        end
    end
    
    # Compute posterior mean of wavelet coefficients
    if isempty(beta_samples)
        println("    Warning: No samples collected for wavelet $wf")
        return cluster_mean  # Return cluster mean if MCMC failed
    end
    
    # Average across samples to get posterior mean coefficients
    n_samples = length(beta_samples)
    posterior_mean_beta = [zeros(length(beta_samples[1][m])) for m in 1:M]
    
    for m in 1:M
        for sample_idx in 1:n_samples
            posterior_mean_beta[m] .+= beta_samples[sample_idx][m]
        end
        posterior_mean_beta[m] ./= n_samples
    end
    
    # Reconstruct smoothed cluster mean from posterior mean coefficients
    smoothed = WaveletOps.compute_mu_from_beta(posterior_mean_beta, wf, J, boundary, P)
    
    println("    MCMC completed for wavelet $wf ($n_samples samples)")
    
    if return_beta
        return (smoothed_mean = smoothed, beta_summaries = (beta_mean = posterior_mean_beta, gamma_last = wpar.gamma_ch))
    else
        return smoothed
    end
end

"""
    create_wavelet_comparison_plot(original, smoothed, t, wf, plot_num)

Create a comparison plot showing original vs smoothed mean function.
"""
function create_wavelet_comparison_plot(original::Matrix{Float64}, smoothed::Matrix{Float64}, t::Vector{Float64}, wf::String, plot_num::Int)
    P, M = size(original)
    
    # For multi-dimensional data, plot the first dimension or average
    if M == 1
        y_orig = original[:, 1]
        y_smooth = smoothed[:, 1]
    else
        y_orig = mean(original, dims=2)[:, 1]
        y_smooth = mean(smoothed, dims=2)[:, 1]
    end
    
    p = plot(t, y_orig, 
             label="Original", 
             color=:blue, 
             linewidth=2,
             title="Wavelet: $wf",
             xlabel="Time (normalized)",
             ylabel="Mean Function Value",
             legend=:topright)
    
    plot!(p, t, y_smooth, 
          label="Smoothed", 
          color=:red, 
          linewidth=2)
    
    return p
end

"""
    posterior_summaries_from_samples(beta_samples::Vector{Vector{Vector{Float64}}}, last_gamma_ch)

Compute posterior summaries from MCMC samples.
"""
function posterior_summaries_from_samples(beta_samples::Vector{Vector{Vector{Float64}}}, last_gamma_ch)
    M = length(beta_samples[1])
    ncoeff = length(beta_samples[1][1])
    β̄_post = [zeros(ncoeff) for _ in 1:M]
    for s in beta_samples, m in 1:M
        β̄_post[m] .+= s[m]
    end
    for m in 1:M
        β̄_post[m] ./= length(beta_samples)
    end
    return (beta_mean = β̄_post, gamma_last = last_gamma_ch)
end

"""
    score_wavelet_candidate(Y_mats::Vector{Matrix{Float64}}, revealed_idx::Vector{Int}, t::AbstractVector,
                           wf::String, J, boundary::String; mcmc = (n_iter = 3000, burnin = 1000, thin = 1))

Score a wavelet candidate using MCMC-based metrics.
"""
function score_wavelet_candidate(Y_mats::Vector{Matrix{Float64}}, revealed_idx::Vector{Int}, t::AbstractVector,
                                 wf::String, J, boundary::String;
                                 mcmc = (n_iter = 3000, burnin = 1000, thin = 1))

    P, M = size(Y_mats[1])

    # Empirical mean over revealed normals
    mean_function = zeros(P, M); cnt = 0
    for i in revealed_idx
        if 1 <= i <= length(Y_mats)
            mean_function .+= Y_mats[i]; cnt += 1
        end
    end
    cnt == 0 && error("No indices in revealed_idx")
    mean_function ./= cnt

    # Ensure J is an integer
    Jv = isnothing(J) ? Utils.ensure_dyadic_J(P, nothing) : J
    
    # Short smoothing MCMC with return_beta
    smoothed_pack = wavelet_smooth_mean_function(Y_mats, revealed_idx, wf, Jv, boundary;
                                                 n_iter_selection = mcmc.n_iter,
                                                 burnin_selection = mcmc.burnin,
                                                 thin_selection   = mcmc.thin,
                                                 return_beta      = true)
    smoothed = smoothed_pack.smoothed_mean
    βsumm    = smoothed_pack.beta_summaries
    β̄_post  = βsumm.beta_mean
    γ_last   = βsumm.gamma_last

    # Time-domain MSE
    mse_time = mean((mean_function .- smoothed).^2)

    # Wavelet-domain MSE (vs average observed coeffs for same wf)
    precomp = WaveletOps.precompute_wavelets(Y_mats, wf, Jv, boundary)
    stk     = WaveletOps.stack_D_from_precomp(precomp, revealed_idx, M)
    D       = stk.D_arr  # ncoeff × N × M
    Dbar    = [vec(mean(view(D, :, :, m); dims = 2)) for m in 1:M]
    @assert all(length(β̄_post[m]) == length(Dbar[m]) for m in 1:M)
    mse_coeff = mean(vcat([(@. (β̄_post[m] - Dbar[m])^2) for m=1:M]...))

    # Sparsity on detail coeffs only
    maps = stk.maps
    inactive = 0; total_det = 0
    for m in 1:M
        for (k, rng) in maps[m].idx
            name = String(k); startswith(name, "d") || continue
            total_det += length(rng)
            inactive  += count(==(0), γ_last[m][rng])
        end
    end
    frac_inactive = total_det == 0 ? 0.0 : inactive / total_det

    return (mse_time = mse_time, mse_coeff = mse_coeff, sparsity = frac_inactive,
            smoothed = smoothed, beta_mean = β̄_post)
end

"""
    select_wavelet(Y, t, revealed_idx; wf_candidates = nothing, J = nothing, boundary = "periodic",
                   metric_weights = (w_time = 0.5, w_coeff = 0.3, w_sparsity = 0.2),
                   mcmc = (n_iter = 3000, burnin = 1000, thin = 1))

Select the best wavelet using MCMC-based scoring.
"""
function select_wavelet(Y, t, revealed_idx;
                        wf_candidates = nothing, J = nothing, boundary = "periodic",
                        metric_weights = (w_time = 0.5, w_coeff = 0.3, w_sparsity = 0.2),
                        mcmc = (n_iter = 3000, burnin = 1000, thin = 1))

    # Normalize inputs to P×M matrices
    Y_mats = Vector{Matrix{Float64}}(undef, length(Y))
    for i in 1:length(Y)
        Y_mats[i] = isa(Y[i], AbstractVector) ? reshape(Float64.(Y[i]), :, 1) : Matrix{Float64}(Y[i])
    end

    # Default candidate set from WaveletOps aliases
    if wf_candidates === nothing
        wf_candidates = ["haar", "db2", "db4", "db6", "db8",
                         "coif2", "coif4", "coif6",
                         "sym4", "sym6", "sym8",
                         "batt2", "batt4", "batt6"]
    end

    results = NamedTuple[]
    best = (wf = "", score = Inf)

    println("Testing $(length(wf_candidates)) wavelet candidates with MCMC scoring...")
    println("MCMC parameters: $(mcmc.n_iter) iterations, $(mcmc.burnin) burn-in, thin=$(mcmc.thin)")

    for wf in wf_candidates
        try
            println("  Testing wavelet: $wf")
            met = score_wavelet_candidate(Y_mats, revealed_idx, t, wf, J, boundary; mcmc = mcmc)
            score = metric_weights.w_time    * met.mse_time +
                    metric_weights.w_coeff   * met.mse_coeff +
                    metric_weights.w_sparsity * (1.0 - met.sparsity)

            push!(results, (wf = wf,
                            mse_time = met.mse_time,
                            mse_coeff = met.mse_coeff,
                            sparsity = met.sparsity,
                            score = score))

            if score < best.score
                best = (wf = wf, score = score)
            end
            
            println("    MSE_time: $(round(met.mse_time, digits=6)), MSE_coeff: $(round(met.mse_coeff, digits=6)), Sparsity: $(round(met.sparsity, digits=3)), Score: $(round(score, digits=6))")
        catch e
            @warn "Wavelet $wf failed during selection: $e"
        end
    end

    if isempty(results)
        error("All wavelet candidates failed during selection")
    end

    results_sorted = sort(results; by = r -> r.score)
    
    println("\nWavelet Selection Results:")
    println("wf        MSE_time    MSE_coeff   Sparsity   Score")
    println("-" ^ 70)
    for r in results_sorted
        println("$(lpad(r.wf, 8)) $(lpad(round(r.mse_time, digits=6), 10)) $(lpad(round(r.mse_coeff, digits=6), 10)) $(lpad(round(r.sparsity, digits=3), 9)) $(lpad(round(r.score, digits=6), 10))")
    end
    println("Selected wavelet: $(best.wf) (score: $(round(best.score, digits=6)))")
    
    return (selected_wf = best.wf, table = results_sorted)
end

end # module
