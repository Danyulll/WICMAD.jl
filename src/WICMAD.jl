module WICMAD

# Core dependencies
using LinearAlgebra
using Distributions
using StatsBase
using StatsFuns
using ProgressMeter
using Random
using Distributed
using Base.Threads
using Dates

# Include all module components
include("utils.jl")
include("wavelets.jl")
include("icm_cache.jl")
include("tprocess.jl")
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
using .TProcess

# Main exports
export wicmad, adj_rand_index, choose2, init_diagnostics,
       plot_cluster_means_diagnostic, plot_kernel_switches_diagnostic, plot_wicmad_diagnostics,
       interactive_kernel_selection, wicmad_bootstrap_driver,
       dahl_from_res, map_from_res, dahl_partition, map_partition

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
    kappa_pi::Float64 = 1.0,
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
    # Bootstrap parameters (defaults ON)
    bootstrap_runs::Int = 1000,
    bootstrap_method::Symbol = :bag,
    bootstrap_fraction::Float64 = 1.0,
    bootstrap_parallel::Symbol = :threads,
    bootstrap_nworkers::Int = max(1, Sys.CPU_THREADS - 1),
    bootstrap_chunk::Int = 8,
    bootstrap_seed::Int = 2025,
    parallel::Bool = false,
    rng::AbstractRNG = Random.default_rng(),
    # Process model options
    process::Symbol = :gp,
    nu_df::Float64 = 5.0,
    learn_nu::Bool = false,
    mh_step_log_nu::Float64 = 0.1,
    # Optional initialization of cluster labels
    z_init::Union{Nothing,Vector{Int}} = nothing,
    # Verbose output control
    verbose::Bool = true,
)
    # Bootstrap dispatch: if bootstrap_runs > 0, use bootstrap driver
    if bootstrap_runs > 0
        # Ensure Y elements are matrices for the bootstrap driver
        N_bt = length(Y)
        Y_mats_bt = Vector{Matrix{Float64}}(undef, N_bt)
        for i in 1:N_bt
            if isa(Y[i], AbstractVector)
                Y_mats_bt[i] = reshape(Float64.(Y[i]), :, 1)
            else
                Y_mats_bt[i] = Matrix{Float64}(Y[i])
            end
        end
        return wicmad_bootstrap_driver(Y_mats_bt, t;
            bootstrap_runs=bootstrap_runs,
            bootstrap_method=bootstrap_method,
            bootstrap_fraction=bootstrap_fraction,
            bootstrap_parallel=bootstrap_parallel,
            bootstrap_nworkers=bootstrap_nworkers,
            bootstrap_chunk=bootstrap_chunk,
            bootstrap_seed=bootstrap_seed,
            verbose=verbose,
            n_iter=n_iter, burn=burn, thin=thin, alpha_prior=alpha_prior,
            wf=wf, J=J, boundary=boundary, mh_step_L=mh_step_L,
            mh_step_eta=mh_step_eta, mh_step_tauB=mh_step_tauB,
            revealed_idx=revealed_idx, K_init=K_init, warmup_iters=warmup_iters,
            unpin=unpin, kappa_pi=kappa_pi, c2=c2, tau_pi=tau_pi,
            a_sig=a_sig, b_sig=b_sig, a_tau=a_tau, b_tau=b_tau,
            a_eta=a_eta, b_eta=b_eta, diagnostics=diagnostics,
            track_ids=track_ids, monitor_levels=monitor_levels,
            wf_candidates=wf_candidates, rng=rng, z_init=z_init)
    end

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
    # Note: We only auto-select if wf is exactly "sym8" (the default), not if it was explicitly set to "sym8"
    if !isempty(revealed_idx) && wf == "sym8" && wf_candidates === nothing  # Only run if using default wavelet and no candidates specified
        verbose && println("\n" * "="^60)
        verbose && println("AUTOMATIC WAVELET SELECTION")
        verbose && println("="^60)
        try
            sel = KernelSelection.select_wavelet(Y_mats, t, revealed_idx; 
                                                 wf_candidates=wf_candidates, 
                                                 J=J, 
                                                 boundary=boundary,
                                                 mcmc=(n_iter=3000, burnin=1000, thin=1),
                                                 verbose=verbose)
            wf = sel.selected_wf
        catch
            verbose && println("Wavelet selection failed, using default 'sym8'")
            wf = "sym8"
        end
        verbose && println("Selected wavelet '$wf' will be used for the analysis.")
        verbose && println("="^60 * "\n")
    end

    kernels = make_kernels(add_bias_variants = false)
    alpha = rand(rng, Gamma(alpha_prior[1], 1 / alpha_prior[2]))
    # If z_init provided, ensure at least that many initial sticks
    K_from_init = z_init === nothing ? K_init : max(K_init, maximum(relabel_to_consecutive(z_init)))
    v = [rand(rng, Beta(1, alpha)) for _ in 1:K_from_init]
    params = ClusterParams[]
    for _ in 1:length(v)
        cp = draw_new_cluster_params(M, P, t_scaled, kernels; wf = wf, J = Jv, boundary = boundary)
        cp.cache = ICMCacheState()  # Initialize cache after ICMCache is loaded
        cp = ensure_complete_cache!(cp, kernels, t_scaled, M)
        cp.cache = build_icm_cache(t_scaled, kernels[cp.kern_idx], cp.thetas[cp.kern_idx], cp.L, cp.eta, cp.tau_B, cp.cache)
        push!(params, cp)
    end

    # Initialize labels
    if z_init !== nothing
        z = relabel_to_consecutive(Vector{Int}(z_init))
        length(z) == N || error("z_init length must match number of observations")
        # Expand sticks/params if needed
        needed_K = maximum(z)
        while length(v) < needed_K
            push!(v, rand(rng, Beta(1, alpha)))
            cp = draw_new_cluster_params(M, P, t_scaled, kernels; wf = wf, J = Jv, boundary = boundary)
            cp.cache = ICMCacheState()
            cp = ensure_complete_cache!(cp, kernels, t_scaled, M)
            cp.cache = build_icm_cache(t_scaled, kernels[cp.kern_idx], cp.thetas[cp.kern_idx], cp.L, cp.eta, cp.tau_B, cp.cache)
            push!(params, cp)
        end
    else
        z = [rand(rng, 1:length(v)) for _ in 1:N]
        if !isempty(revealed_idx)
            for idx in revealed_idx
                if 1 <= idx <= N
                    z[idx] = 1
                end
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
    proc_cfg = Utils.ProcessConfig(process, nu_df, learn_nu)
    # Always maintain per-curve scale vector; for GP this remains all ones
    lambda = ones(Float64, N)
    
    # Progress reporting setup (no progress bar; periodic thread-aware prints)
    print_interval = max(1, n_iter ÷ 20)  # Print every 5% of iterations
    mode_tag = parallel ? "parallel" : "single-thread"
    verbose && println("Starting WICMAD MCMC with $n_iter iterations ($mode_tag mode)...")

    for iter in 1:n_iter
        pi = Utils.stick_to_pi(v)
        u = [rand(rng, Uniform(0, pi[z[i]])) for i in 1:N]
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
                if process == :tprocess
                    # Build cache and compute TP loglik with per-curve lambda
                    cp = params[k]
                    kc = kernels[cp.kern_idx]
                    kp = cp.thetas[cp.kern_idx]
                    cache = build_icm_cache(t_scaled, kc, kp, cp.L, cp.eta, cp.tau_B, cp.cache)
                    params[k].cache = cache
                    ll = TProcess.loglik_residual_tp_matrix(Y_mats[i] - params[k].mu_cached, cache, lambda[i])
                else
                    ll = ll_curve_k(k, Y_mats[i], params[k].mu_cached)
                end
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
            upd = WaveletOps.update_cluster_wavelet_params_besov_fullbayes(
                idx, precomp_all, M, params[k].wpar, params[k].sigma2, params[k].tau_sigma
            )
            params[k].wpar      = upd.wpar
            params[k].beta_ch   = upd.beta_ch
            params[k].sigma2    = upd.sigma2_m
            params[k].tau_sigma = upd.tau_sigma

            mu_k = WaveletOps.compute_mu_from_beta(params[k].beta_ch, wf, Jv, boundary, P)
            params[k].mu_cached = mu_k
            params[k].mu_cached_iter = iter
            Yk = [Y_mats[ii] for ii in idx]
            params = cc_switch_kernel_eig(k, params, kernels, t_scaled, Yk, process, lambda[idx])
            params = mh_update_kernel_eig(k, params, kernels, t_scaled, Yk, a_eta, b_eta, process, lambda[idx])
            params = mh_update_L_eig(k, params, kernels, t_scaled, Yk, mh_step_L, process, lambda[idx])
            params = mh_update_eta_eig(k, params, kernels, t_scaled, Yk, mh_step_eta, a_eta, b_eta, process, lambda[idx])
            params = mh_update_tauB_eig(k, params, kernels, t_scaled, Yk, mh_step_tauB, process, lambda[idx])
        end

        # T-process latent scale updates (after params and mu are updated)
        if process == :tprocess
            residuals = Vector{Matrix{Float64}}(undef, N)
            @inbounds for n in 1:N
                kn = z[n]
                ensure_mu_cached!(kn, iter)
                residuals[n] = Y_mats[n] - params[kn].mu_cached
            end
            TProcess.sample_lambda!(lambda, proc_cfg.nu_df, z, residuals, params)
            if proc_cfg.learn_nu
                proc_cfg = Utils.ProcessConfig(process, TProcess.mh_update_nu!(proc_cfg.nu_df, lambda, mh_step_log_nu), true)
            end
        end

        Kocc = length(unique(z))
        if diagnostics && diag !== nothing
            diag[:ari_all][iter] = z_prev === nothing ? NaN : adj_rand_index(z_prev, z)
            diag[:global][:K_occ_all][iter] = Kocc
        end

        eta_aux = rand(rng, Beta(alpha + 1, N))
        mix = (alpha_prior[1] + Kocc - 1) / (N * (alpha_prior[2] - log(eta_aux)) + alpha_prior[1] + Kocc - 1)
        if rand(rng) < mix
            alpha = rand(rng, Gamma(alpha_prior[1] + Kocc, 1 / (alpha_prior[2] - log(eta_aux))))
        else
            alpha = rand(rng, Gamma(alpha_prior[1] + Kocc - 1, 1 / (alpha_prior[2] - log(eta_aux))))
        end

        # Update progress reporting (periodic, per thread)
        if iter % print_interval == 0 || iter == n_iter
            percentage = round(100 * iter / n_iter, digits=1)
            verbose && println("Thread $(Threads.threadid()): $iter/$n_iter ($percentage%) - Clusters: $Kocc")
        end

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
                if process == :tprocess
                    cp = params[ki]
                    kc = kernels[cp.kern_idx]
                    kp = cp.thetas[cp.kern_idx]
                    cache = build_icm_cache(t_scaled, kc, kp, cp.L, cp.eta, cp.tau_B, cp.cache)
                    params[ki].cache = cache
                    totll += TProcess.loglik_residual_tp_matrix(Y_mats[i] - params[ki].mu_cached, cache, lambda[i])
                else
                    totll += ll_curve_k(ki, Y_mats[i], params[ki].mu_cached)
                end
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
    if verbose
        if parallel
            println("Thread $(Threads.threadid()): MCMC completed! Final clusters: $final_Kocc, Samples collected: $sidx")
        else
            println("\nMCMC completed! Final clusters: $final_Kocc, Samples collected: $sidx")
        end
    end

    (; Z = Z_s, alpha = alpha_s, kern = kern_s, params = params, v = v, pi = Utils.stick_to_pi(v),
        revealed_idx = revealed_idx, K_occ = K_s, loglik = loglik_s, diagnostics = diag,
        process = proc_cfg)
end

# =============================================================================
# BOOTSTRAP FUNCTIONALITY
# =============================================================================

"""
    relabel_to_consecutive(z::Vector{Int}) -> Vector{Int}

Relabel cluster assignments to consecutive integers starting from 1.
"""
function relabel_to_consecutive(z::Vector{Int})
    unique_labels = unique(z)
    label_map = Dict(old_label => new_label for (new_label, old_label) in enumerate(unique_labels))
    return [label_map[label] for label in z]
end

"""
    coassign_from_labels(z::AbstractVector{Int}) -> Matrix{Float64}

Create coassignment matrix from cluster labels.
"""
function coassign_from_labels(z::AbstractVector{Int})
    N = length(z)
    S = zeros(Float64, N, N)
    for i in 1:N
        for j in i:N
            S[i, j] = (z[i] == z[j]) ? 1.0 : 0.0
            if j != i
                S[j, i] = S[i, j]
            end
        end
    end
    return S
end

"""
    assign_argmax_cluster(y::Matrix{Float64}, t::Union{Vector{Float64}, Vector{Int64}}, params_map::Vector{ClusterParams}; rng::AbstractRNG) -> Int

Assign observation to cluster with maximum likelihood using frozen MAP parameters.
"""
function assign_argmax_cluster(y::Matrix{Float64}, t::Union{Vector{Float64}, Vector{Int64}}, params_map::Vector{ClusterParams}; rng::AbstractRNG)
    K = length(params_map)
    logliks = Vector{Float64}(undef, K)
    
    for k in 1:K
        cp = params_map[k]
        # Use cached mu if available, otherwise compute
        if cp.mu_cached !== nothing
            mu_k = cp.mu_cached
        else
            # This would need the wavelet parameters - simplified for now
            mu_k = zeros(size(y))
        end
        
        # Compute log-likelihood using ICM cache
        cache = cp.cache
        logliks[k] = fast_icm_loglik_curve(y - mu_k, cache)
    end
    
    return argmax(logliks)
end

"""
    _wicmad_bootstrap_one_run(Y::Vector{Matrix{Float64}}, t::Vector{Float64}; kwargs...) -> NamedTuple

Run a single bootstrap iteration: fit on in-bag data, assign OOB data using MAP parameters.
Returns both the full assignment and the MAP partition from the in-bag fit.
"""
function _wicmad_bootstrap_one_run(
    Y::Vector{Matrix{Float64}}, 
    t::Union{Vector{Float64}, Vector{Int64}};
    n_inbag::Int, 
    method::Symbol = :bag, 
    rng::AbstractRNG = Random.default_rng(),
    parallel::Bool = false,
    kwargs...
)
    N = length(Y)
    
    # Draw in-bag indices
    if method == :bag
        I_b = rand(rng, 1:N, n_inbag)
    else  # :subsample
        I_b = randperm(rng, N)[1:n_inbag]
    end
    I_bu = unique(I_b)
    OOB = setdiff(1:N, I_bu)
    
    # Subset z_init to in-bag if provided
    z_init_kw = get(kwargs, :z_init, nothing)
    local kwargs2
    if z_init_kw === nothing
        kwargs2 = NamedTuple(kwargs)
    else
        z_init_inbag = Vector{Int}(undef, length(I_bu))
        for (j, ii) in enumerate(I_bu)
            z_init_inbag[j] = z_init_kw[ii]
        end
        kwargs2 = merge(NamedTuple(kwargs), (z_init = z_init_inbag,))
    end

    # Fit DP model on in-bag only
    fit = wicmad(Y[I_bu], t; rng=rng, bootstrap_runs=0, parallel=parallel, kwargs2...)
    
    # Get MAP partition and parameters
    # For now, use the last MCMC sample as MAP approximation
    zinbag_map = vec(fit.Z[end, :])
    params_map = fit.params
    
    # Initialize full assignment vector
    zfull = Vector{Int}(undef, N)
    zfull[I_bu] = zinbag_map
    
    # Assign OOB observations using argmax likelihood
    for o in OOB
        zfull[o] = assign_argmax_cluster(Y[o], t, params_map; rng=rng)
    end
    
    # Create MAP partition for full dataset (only in-bag observations have MAP assignments)
    zmap_full = Vector{Int}(undef, N)
    zmap_full[I_bu] = zinbag_map
    # For OOB observations, we don't have MAP assignments, so we'll use the argmax assignments
    zmap_full[OOB] = zfull[OOB]
    
    return (
        z_full = relabel_to_consecutive(zfull),
        z_map = relabel_to_consecutive(zmap_full),
        params_map = params_map,
        inbag_indices = I_bu,
        oob_indices = OOB
    )
end

"""
    _bootstrap_aggregate_to_consensus(Zhat_full::Matrix{Int}, Zhat_map::Matrix{Int})

Aggregate bootstrap results using Dahl least-squares consensus method.
Returns both consensus clustering and MAP clustering.
"""
function _bootstrap_aggregate_to_consensus(Zhat_full::Matrix{Int}, Zhat_map::Matrix{Int})
    N, B = size(Zhat_full)
    
    # Build coassignment matrix for full assignments
    A_full = zeros(Float64, N, N)
    for b in 1:B
        zb = view(Zhat_full, :, b)
        for i in 1:N
            @inbounds for j in i:N
                v = (zb[i] == zb[j]) ? 1.0 : 0.0
                A_full[i, j] += v
                if j != i
                    A_full[j, i] += v
                end
            end
        end
    end
    A_full ./= B
    
    # Build coassignment matrix for MAP assignments
    A_map = zeros(Float64, N, N)
    for b in 1:B
        zb = view(Zhat_map, :, b)
        for i in 1:N
            @inbounds for j in i:N
                v = (zb[i] == zb[j]) ? 1.0 : 0.0
                A_map[i, j] += v
                if j != i
                    A_map[j, i] += v
                end
            end
        end
    end
    A_map ./= B
    
    # Dahl least-squares: choose run minimizing ||S_b - A||_F^2 for full assignments
    best_b_full, best_loss_full = 1, Inf
    for b in 1:B
        zb = view(Zhat_full, :, b)
        S = coassign_from_labels(zb)
        loss = sum(@. (S - A_full)^2)
        if loss < best_loss_full
            best_loss_full = loss
            best_b_full = b
        end
    end
    
    # Dahl least-squares: choose run minimizing ||S_b - A||_F^2 for MAP assignments
    best_b_map, best_loss_map = 1, Inf
    for b in 1:B
        zb = view(Zhat_map, :, b)
        S = coassign_from_labels(zb)
        loss = sum(@. (S - A_map)^2)
        if loss < best_loss_map
            best_loss_map = loss
            best_b_map = b
        end
    end
    
    z_consensus_full = copy(view(Zhat_full, :, best_b_full))
    z_consensus_map = copy(view(Zhat_map, :, best_b_map))
    
    return (
        z_consensus = z_consensus_full,
        z_map_consensus = z_consensus_map,
        A_full = A_full,
        A_map = A_map,
        Zhat_full = Zhat_full,
        Zhat_map = Zhat_map,
        meta = (B=B, best_b_full=best_b_full, best_b_map=best_b_map)
    )
end

"""
    wicmad_bootstrap_driver(Y::Vector{Matrix{Float64}}, t::Vector{Float64}; kwargs...)

Main bootstrap driver that orchestrates parallel bootstrap runs and returns consensus clustering.
Runs wavelet selection first, then uses the selected wavelet for all bootstrap runs.
"""
function wicmad_bootstrap_driver(
    Y::Vector{Matrix{Float64}}, 
    t::Union{Vector{Float64}, Vector{Int64}};
    bootstrap_runs::Int = 1000,
    bootstrap_method::Symbol = :bag,
    bootstrap_fraction::Float64 = 1.0,
    bootstrap_parallel::Symbol = :threads,
    bootstrap_nworkers::Int = max(1, Sys.CPU_THREADS - 1),
    bootstrap_chunk::Int = 8,
    bootstrap_seed::Int = 2025,
    verbose::Bool = true,
    kwargs...
)
    N = length(Y)
    B = bootstrap_runs
    n_inbag = max(1, ceil(Int, bootstrap_fraction * N))
    
    # Extract wavelet selection parameters from kwargs
    wf = get(kwargs, :wf, "sym8")
    J = get(kwargs, :J, nothing)
    boundary = get(kwargs, :boundary, "periodic")
    revealed_idx = get(kwargs, :revealed_idx, Int[])
    wf_candidates = get(kwargs, :wf_candidates, nothing)
    
    # Run wavelet selection first if revealed_idx is provided and using default wavelet
    selected_wf = wf
    if !isempty(revealed_idx) && wf == "sym8"  # Only run if using default wavelet
        verbose && println("\n" * "="^60)
        verbose && println("BOOTSTRAP WAVELET SELECTION")
        verbose && println("="^60)
        
        # Convert Y to matrix format for wavelet selection
        Y_mats = Vector{Matrix{Float64}}(undef, N)
        for i in 1:N
            if isa(Y[i], AbstractVector)
                Y_mats[i] = reshape(Float64.(Y[i]), :, 1)
            else
                Y_mats[i] = Matrix{Float64}(Y[i])
            end
        end
        
        # Run wavelet selection
        try
            sel = KernelSelection.select_wavelet(Y_mats, t, revealed_idx; 
                                                 wf_candidates=wf_candidates, 
                                                 J=J, 
                                                 boundary=boundary,
                                                 mcmc=(n_iter=3000, burnin=1000, thin=1),
                                                 verbose=verbose)
        catch
            verbose && println("Wavelet selection failed, using default 'sym8'")
            sel = (selected_wf = "sym8", table = NamedTuple[])
        end
        selected_wf = sel.selected_wf
        verbose && println("Selected wavelet '$selected_wf' will be used for all bootstrap runs.")
        verbose && println("="^60 * "\n")
    end
    
    # Update kwargs with selected wavelet and verbose
    kwargs_updated = merge(NamedTuple(kwargs), (wf=selected_wf, verbose=verbose))
    
    # Storage: labels for all N obs × B runs (both full and MAP)
    Zhat_full = Matrix{Int}(undef, N, B)
    Zhat_map = Matrix{Int}(undef, N, B)
    
    verbose && println("Starting bootstrap with $B runs using $bootstrap_parallel parallelization...")
    verbose && println("Using wavelet: $selected_wf")
    
    # Diagnostic information
    if verbose
        if bootstrap_parallel == :threads
            println("Threading info: $(Threads.nthreads()) threads available")
        elseif bootstrap_parallel == :processes
            println("Process info: $(nprocs()) processes available")
        end
    end
    
    if bootstrap_parallel == :none
        for b in 1:B
            rng = MersenneTwister(bootstrap_seed + b)
            result = _wicmad_bootstrap_one_run(Y, t; n_inbag=n_inbag, method=bootstrap_method, rng=rng, kwargs_updated...)
            Zhat_full[:, b] = result.z_full
            Zhat_map[:, b] = result.z_map
        end
        
    elseif bootstrap_parallel == :threads
        Threads.@threads for b in 1:B
            verbose && println("Thread $(Threads.threadid()) starting bootstrap run $b")
            rng = MersenneTwister(bootstrap_seed + b)
            result = _wicmad_bootstrap_one_run(Y, t; n_inbag=n_inbag, method=bootstrap_method, rng=rng, parallel=true, kwargs_updated...)
            Zhat_full[:, b] = result.z_full
            Zhat_map[:, b] = result.z_map
            verbose && println("Thread $(Threads.threadid()) completed bootstrap run $b")
        end
        
    elseif bootstrap_parallel == :processes
        if nprocs() == 1
            addprocs(bootstrap_nworkers)
        end
        
        # Make code visible on workers
        Distributed.@everywhere begin
            # Random and WICMAD are already available
        end
        
        # Ship immutable views/args
        jobs = collect(1:B)
        kwargs_nt = (; kwargs_updated...)
        
        Distributed.@everywhere function _bootstrap_job(b::Int, Y, t, n_inbag, method, seed, kwargs_nt)
            rng = MersenneTwister(seed + b)
            return WICMAD._wicmad_bootstrap_one_run(Y, t; n_inbag=n_inbag, method=method, rng=rng, (;kwargs_nt)...)
        end
        
        results = pmap(b -> _bootstrap_job(b, Y, t, n_inbag, bootstrap_method, bootstrap_seed, kwargs_nt),
                       jobs; batch_size=bootstrap_chunk)
        
        for (k, b) in enumerate(jobs)
            Zhat_full[:, b] = results[k].z_full
            Zhat_map[:, b] = results[k].z_map
        end
        
    else
        error("bootstrap_parallel must be :none | :threads | :processes")
    end
    
    println("Bootstrap runs completed. Computing consensus clustering...")
    
    return _bootstrap_aggregate_to_consensus(Zhat_full, Zhat_map)
end

end # module