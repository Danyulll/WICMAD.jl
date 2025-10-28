module WICMAD

# Core dependencies
using LinearAlgebra
using Distributions
using StatsBase
using StatsFuns
using ProgressMeter

# Include all module components
include("utils.jl")
include("wavelets.jl")
include("icm_cache.jl")
include("mh_updates.jl")
include("postprocessing.jl")
include("plotting.jl")
include("kernel_selection.jl")

# Re-export submodules
using .Utils
using .WaveletOps
using .ICMCache
using .MHUpdates
using .PostProcessing
using .Plotting
using .KernelSelection

# Main exports
export wicmad, adj_rand_index, choose2, init_diagnostics,
       plot_cluster_means_diagnostic, plot_kernel_switches_diagnostic, plot_wicmad_diagnostics,
       interactive_kernel_selection

# Utility functions
choose2(n::Int) = n < 2 ? 0.0 : n * (n - 1) / 2

function adj_rand_index(z1::Vector{Int}, z2::Vector{Int})
    length(z1) == length(z2) || error("adj_rand_index: vectors must have same length")
    
    # Handle edge cases
    if isempty(z1) || isempty(z2)
        return 0.0
    end
    
    # Get unique labels and create mappings
    labs1 = unique(z1)
    labs2 = unique(z2)
    
    # Handle case where all labels are the same
    if length(labs1) == 1 && length(labs2) == 1
        return labs1[1] == labs2[1] ? 1.0 : 0.0
    end
    
    map1 = Dict(lab => i for (i, lab) in enumerate(labs1))
    map2 = Dict(lab => i for (i, lab) in enumerate(labs2))
    
    # Create contingency table
    tab = zeros(Int, length(labs1), length(labs2))
    for i in eachindex(z1)
        tab[map1[z1[i]], map2[z2[i]]] += 1
    end
    
    # Calculate ARI components
    sum_comb = sum(choose2(tab[i, j]) for i in axes(tab, 1), j in axes(tab, 2))
    a = [sum(tab[i, :]) for i in axes(tab, 1)]
    b = [sum(tab[:, j]) for j in axes(tab, 2)]
    sum_a = sum(choose2(ai) for ai in a)
    sum_b = sum(choose2(bj) for bj in b)
    tot = choose2(length(z1))
    
    # Calculate ARI
    denom = 0.5 * (sum_a + sum_b) - (sum_a * sum_b) / tot
    if denom == 0
        return 0.0
    end
    
    ari = (sum_comb - (sum_a * sum_b) / tot) / denom
    
    # Clamp ARI to valid range [-1, 1] and then to [0, 1] for our purposes
    return max(0.0, min(1.0, ari))
end

function init_diagnostics(diagnostics::Bool, keep::Int, n_iter::Int)
    diagnostics || return nothing
    Dict(
        :global => Dict(
            :K_occ => fill(Float64(NaN), keep),
            :alpha => fill(Float64(NaN), keep),
            :loglik => fill(Float64(NaN), keep),
            :K_occ_all => fill(Float64(NaN), n_iter)
        ),
        :ari => fill(Float64(NaN), max(keep - 1, 0)),
        :ari_all => fill(Float64(NaN), n_iter)
    )
end

function wicmad(
    Y::Vector,
    t;
    n_iter::Int = 6000,
    burn::Int = 3000,
    thin::Int = 5,
    alpha_prior::Tuple{Float64,Float64} = (10.0, 1.0),
    wf::String = "sym8",
    J = nothing,
    boundary::String = "periodic",
    mh_step_L::Float64 = 0.03,
    mh_step_eta::Float64 = 0.10,
    mh_step_tauB::Float64 = 0.15,
    revealed_idx::Vector{Int} = Int[],
    K_init::Int = 5,
    warmup_iters::Int = 100,
    unpin::Bool = false,
    kappa_pi::Float64 = 0.6,
    c2::Float64 = 1.0,
    tau_pi::Float64 = 40.0,
    a_sig::Float64 = 2.5,
    b_sig::Float64 = 0.02,
    a_tau::Float64 = 2.0,
    b_tau::Float64 = 2.0,
    a_eta::Float64 = 2.0,
    b_eta::Float64 = 0.1,
    diagnostics::Bool = true,
    track_ids = nothing,
    monitor_levels = nothing,
    wf_candidates = nothing,
)
    n_iter < 2 && error("n_iter must be >= 2")
    thin = max(thin, 1)
    if burn >= n_iter
        burn = max(0, n_iter - max(2, thin))
    end
    if (n_iter - burn) < thin
        thin = max(1, n_iter - burn)
    end

    N = length(Y)
    N == 0 && error("Y must contain at least one curve")
    # Convert Y elements to matrices, handling both vectors and matrices
    Y_mats = Vector{Matrix{Float64}}(undef, N)
    for i in 1:N
        if isa(Y[i], AbstractVector)
            Y_mats[i] = reshape(Float64.(Y[i]), :, 1)
        else
            Y_mats[i] = Matrix{Float64}(Y[i])
        end
    end
    P = size(Y_mats[1], 1)
    M = size(Y_mats[1], 2)
    Jv = Utils.ensure_dyadic_J(P, J)
    t_norm = Utils.normalize_t(t, P)
    t_scaled = Utils.scale_t01(t_norm)

    # Automatic wavelet selection if revealed_idx is provided and no specific wavelet is given
    if !isempty(revealed_idx) && wf == "sym8"  # Only run if using default wavelet
        println("\n" * "="^60)
        println("AUTOMATIC WAVELET SELECTION")
        println("="^60)
        sel = KernelSelection.select_wavelet(Y_mats, t, revealed_idx; 
                                             wf_candidates=wf_candidates, 
                                             J=J, 
                                             boundary=boundary,
                                             mcmc=(n_iter=3000, burnin=1000, thin=1))
        wf = sel.selected_wf
        println("Selected wavelet '$wf' will be used for the analysis.")
        println("="^60 * "\n")
    end

    kernels = make_kernels(add_bias_variants = false)
    alpha = Base.rand(Gamma(alpha_prior[1], 1 / alpha_prior[2]))
    v = [Base.rand(Beta(1, alpha)) for _ in 1:K_init]
    params = ClusterParams[]
    for _ in 1:length(v)
        cp = draw_new_cluster_params(M, P, t_scaled, kernels; wf = wf, J = Jv, boundary = boundary)
        cp.cache = ICMCacheState()  # Initialize cache after ICMCache is loaded
        cp = ensure_complete_cache!(cp, kernels, t_scaled, M)
        cp.cache = build_icm_cache(t_scaled, kernels[cp.kern_idx], cp.thetas[cp.kern_idx], cp.L, cp.eta, cp.tau_B, cp.cache)
        push!(params, cp)
    end

    z = [Base.rand(1:length(v)) for _ in 1:N]
    if !isempty(revealed_idx)
        for idx in revealed_idx
            if 1 <= idx <= N
                z[idx] = 1
            end
        end
    end

    keep = max(0, Int(floor((n_iter - burn) / thin)))
    Z_s = keep > 0 ? Array{Int}(undef, keep, N) : Array{Int}(undef, 0, N)
    alpha_s = keep > 0 ? Vector{Float64}(undef, keep) : Float64[]
    kern_s = keep > 0 ? Vector{Int}(undef, keep) : Int[]
    K_s = keep > 0 ? Vector{Int}(undef, keep) : Int[]
    loglik_s = keep > 0 ? Vector{Float64}(undef, keep) : Float64[]

    diag = init_diagnostics(diagnostics, keep, n_iter)

    precomp_all = precompute_wavelets(Y_mats, wf, Jv, boundary)


    function ll_curve_k(k::Int, Yi::Matrix{Float64}, mu_k::Matrix{Float64})
        cp = params[k]
        kc = kernels[cp.kern_idx]
        kp = cp.thetas[cp.kern_idx]
        cache = build_icm_cache(t_scaled, kc, kp, cp.L, cp.eta, cp.tau_B, cp.cache)
        params[k].cache = cache
        fast_icm_loglik_curve(Yi - mu_k, cache)
    end

    function ensure_mu_cached!(k::Int, iter::Int)
        cp = params[k]
        if cp.mu_cached === nothing || cp.mu_cached_iter != iter
            mu_k = compute_mu_from_beta(cp.beta_ch, wf, Jv, boundary, P)
            params[k].mu_cached = mu_k
            params[k].mu_cached_iter = iter
        end
    end

    sidx = 0
    
    # Progress bar setup
    progress_bar = Progress(n_iter, desc="WICMAD MCMC", 
                           showspeed=true, 
                           color=:blue)
    
    println("Starting WICMAD MCMC with $n_iter iterations...")

    for iter in 1:n_iter
        pi = Utils.stick_to_pi(v)
        u = [Base.rand(Uniform(0, pi[z[i]])) for i in 1:N]
        u_star = minimum(u)
        v = Utils.extend_sticks_until(v, alpha, u_star)
        pi = Utils.stick_to_pi(v)
        K = length(v)
        while length(params) < K
            cp = draw_new_cluster_params(M, P, t_scaled, kernels; wf = wf, J = Jv, boundary = boundary)
            cp.cache = ICMCacheState()  # Initialize cache after ICMCache is loaded
            cp = ensure_complete_cache!(cp, kernels, t_scaled, M)
            cp.cache = build_icm_cache(t_scaled, kernels[cp.kern_idx], cp.thetas[cp.kern_idx], cp.L, cp.eta, cp.tau_B, cp.cache)
            push!(params, cp)
        end
        for k in 1:length(params)
            params[k] = ensure_complete_cache!(params[k], kernels, t_scaled, M)
            params[k].cache = build_icm_cache(t_scaled, kernels[params[k].kern_idx], params[k].thetas[params[k].kern_idx], params[k].L, params[k].eta, params[k].tau_B, params[k].cache)
        end

        z_prev = diagnostics ? copy(z) : nothing

        for i in 1:N
            if !isempty(revealed_idx) && (i in revealed_idx) && (!unpin || (warmup_iters > 0 && iter <= warmup_iters))
                z[i] = 1
                continue
            end
            S = findall(x -> x > u[i], pi)
            isempty(S) && (S = [1])
            logw = Vector{Float64}(undef, length(S))
            for (idx_s, k) in enumerate(S)
                ensure_mu_cached!(k, iter)
                ll = ll_curve_k(k, Y_mats[i], params[k].mu_cached)
                logw[idx_s] = log(pi[k]) + ll
            end
            logw .-= maximum(logw)
            w = exp.(logw)
            w ./= sum(w)
            z[i] = S[sample(1:length(S), Weights(w))]
        end

        v = Utils.update_v_given_z(v, z, alpha)
        pi = Utils.stick_to_pi(v)
        K = length(v)

        for k in 1:K
            idx = findall(==(k), z)
            isempty(idx) && continue
            upd = update_cluster_wavelet_params_besov(idx, precomp_all, M, params[k].wpar,
                params[k].sigma2, params[k].tau_sigma; kappa_pi = kappa_pi, c2 = c2, tau_pi = tau_pi,
                g_hyp = params[k].g_hyp, a_sig = a_sig, b_sig = b_sig, a_tau = a_tau, b_tau = b_tau)
            params[k].wpar = upd.wpar
            params[k].beta_ch = upd.beta_ch
            params[k].sigma2 = upd.sigma2_m
            params[k].tau_sigma = upd.tau_sigma
            mu_k = compute_mu_from_beta(params[k].beta_ch, wf, Jv, boundary, P)
            params[k].mu_cached = mu_k
            params[k].mu_cached_iter = iter
            Yk = [Y_mats[ii] for ii in idx]
            params = cc_switch_kernel_eig(k, params, kernels, t_scaled, Yk)
            params = mh_update_kernel_eig(k, params, kernels, t_scaled, Yk, a_eta, b_eta)
            params = mh_update_L_eig(k, params, kernels, t_scaled, Yk, mh_step_L)
            params = mh_update_eta_eig(k, params, kernels, t_scaled, Yk, mh_step_eta, a_eta, b_eta)
            params = mh_update_tauB_eig(k, params, kernels, t_scaled, Yk, mh_step_tauB)
        end

        Kocc = length(unique(z))
        if diagnostics && diag !== nothing
            diag[:ari_all][iter] = z_prev === nothing ? NaN : adj_rand_index(z_prev, z)
            diag[:global][:K_occ_all][iter] = Kocc
        end

        eta_aux = Base.rand(Beta(alpha + 1, N))
        mix = (alpha_prior[1] + Kocc - 1) / (N * (alpha_prior[2] - log(eta_aux)) + alpha_prior[1] + Kocc - 1)
        if Base.rand() < mix
            alpha = Base.rand(Gamma(alpha_prior[1] + Kocc, 1 / (alpha_prior[2] - log(eta_aux))))
        else
            alpha = Base.rand(Gamma(alpha_prior[1] + Kocc - 1, 1 / (alpha_prior[2] - log(eta_aux))))
        end

        # Update progress bar
        next!(progress_bar, showvalues=[(:iterations, iter), (:clusters, Kocc)])

        if keep > 0 && iter > burn && ((iter - burn) % thin == 0)
            sidx += 1
            Z_s[sidx, :] = z
            alpha_s[sidx] = alpha
            counts = countmap(z)
            sorted_keys = sort(collect(keys(counts)); by = k -> -counts[k])
            k_big = sorted_keys[1]
            k_sec = length(sorted_keys) >= 2 ? sorted_keys[2] : k_big
            kern_s[sidx] = params[k_big].kern_idx
            K_s[sidx] = Kocc
            totll = 0.0
            for i in 1:N
                ki = z[i]
                ensure_mu_cached!(ki, iter)
                totll += ll_curve_k(ki, Y_mats[i], params[ki].mu_cached)
            end
            loglik_s[sidx] = totll
            if diagnostics && diag !== nothing
                diag[:global][:K_occ][sidx] = Kocc
                diag[:global][:alpha][sidx] = alpha
                diag[:global][:loglik][sidx] = totll
                if sidx >= 2
                    diag[:ari][sidx - 1] = adj_rand_index(vec(Z_s[sidx - 1, :]), vec(Z_s[sidx, :]))
                end
            end
        end
    end
    
    # Calculate final number of clusters
    final_Kocc = length(unique(z))
    println("\nMCMC completed! Final clusters: $final_Kocc, Samples collected: $sidx")

    (; Z = Z_s, alpha = alpha_s, kern = kern_s, params = params, v = v, pi = Utils.stick_to_pi(v),
        revealed_idx = revealed_idx, K_occ = K_s, loglik = loglik_s, diagnostics = diag)
end

end # module