module MHUpdates

using ..Utils
using ..ICMCache
using ..TProcess
using LinearAlgebra
using Distributions
using StatsBase

export sum_ll_curves, mh_update_kernel_eig, mh_update_L_eig, mh_update_eta_eig,
       mh_update_tauB_eig, cc_switch_kernel_eig

function sum_ll_curves(curves::Vector{Matrix{Float64}}, cache::ICMCacheState; process::Symbol = :gp, lambdas::Union{Nothing,Vector{Float64}} = nothing)
    total = 0.0
    if process == :tprocess
        lambdas === nothing && error("lambdas must be provided for tprocess")
        # For a shared cache (same cluster params), each curve uses its own λ
        for (i, y) in enumerate(curves)
            total += TProcess.loglik_residual_tp_matrix(y, cache, lambdas[i])
        end
    else
        for y in curves
            total += fast_icm_loglik_curve(y, cache)
        end
    end
    total
end

function clone_dict(d::Dict{Symbol,Float64})
    Dict{Symbol,Float64}(k => v for (k, v) in d)
end

function mh_update_kernel_eig(k::Int, params::Vector{Utils.ClusterParams}, kernels::Vector{Utils.KernelConfig}, t, Y_list, a_eta::Float64, b_eta::Float64, process::Symbol, lambdas_k::Vector{Float64})
    cp = params[k]
    kc = kernels[cp.kern_idx]
    kp = clone_dict(cp.thetas[cp.kern_idx])
    curves = [Matrix{Float64}(Yi - cp.mu_cached) for Yi in Y_list]
    for pn in kc.pnames
        cur = kp[pn]
        if pn == :period
            z_cur = Utils.logit_safe(cur)
            z_prp = Base.rand(Normal(z_cur, kc.prop_sd[pn]))
            prp = Utils.invlogit_safe(z_prp)
            kp_prop = clone_dict(kp); kp_prop[pn] = prp
            cache_cur = build_icm_cache(t, kc, kp, cp.L, cp.eta, cp.tau_B, cp.cache)
            ll_cur = sum_ll_curves(curves, cache_cur; process=process, lambdas=lambdas_k)
            cache_prp = build_icm_cache(t, kc, kp_prop, cp.L, cp.eta, cp.tau_B, ICMCacheState())
            ll_prp = sum_ll_curves(curves, cache_prp; process=process, lambdas=lambdas_k)
            lp_cur = kc.prior(kp)
            lp_prp = kc.prior(kp_prop)
            # Jacobian for logistic transform: period = logistic(z)
            jac = log(prp * (1 - prp)) - log(cur * (1 - cur))
            a = (ll_prp + lp_prp + jac) - (ll_cur + lp_cur)
            if isfinite(a) && log(Base.rand()) < a
                cp.thetas[cp.kern_idx] = kp_prop
                cp.cache = cache_prp
                acc = cp.acc.kernel[pn]
                acc.a += 1
            end
            cp.acc.kernel[pn].n += 1
            kp = clone_dict(cp.thetas[cp.kern_idx])
        elseif pn == :kappa
            # Bounded (0,2] param → propose in z-space via logit(kappa/2)
            z_cur = Utils.logit_safe(cur / 2)
            z_prp = Base.rand(Normal(z_cur, kc.prop_sd[pn]))
            prp   = 2 * Utils.invlogit_safe(z_prp)

            kp_prop = clone_dict(kp); kp_prop[pn] = prp

            cache_cur = build_icm_cache(t, kc, kp,      cp.L, cp.eta, cp.tau_B, cp.cache)
            ll_cur    = sum_ll_curves(curves, cache_cur; process=process, lambdas=lambdas_k)
            cache_prp = build_icm_cache(t, kc, kp_prop, cp.L, cp.eta, cp.tau_B, ICMCacheState())
            ll_prp    = sum_ll_curves(curves, cache_prp; process=process, lambdas=lambdas_k)

            lp_cur = kc.prior(kp)
            lp_prp = kc.prior(kp_prop)

            # Jacobian for κ via u = κ/2 = logistic(z): dκ/dz = 2 u (1-u); log 2 cancels
            u_cur = cur / 2
            u_prp = prp / 2
            jac = log(u_prp * (1 - u_prp)) - log(u_cur * (1 - u_cur))
            a = (ll_prp + lp_prp + jac) - (ll_cur + lp_cur)
            if isfinite(a) && log(Base.rand()) < a
                cp.thetas[cp.kern_idx] = kp_prop
                cp.cache = cache_prp
                cp.acc.kernel[pn].a += 1
                kp = clone_dict(kp_prop)
            end
            cp.acc.kernel[pn].n += 1
        elseif pn == :t0
            # Bounded param t0 lies in [tmin, tmax] → propose in logit space of its rescaling
            tmin, tmax = minimum(t), maximum(t)
            u_cur = (cur - tmin) / (tmax - tmin)
            z_cur = Utils.logit_safe(u_cur)
            z_prp = Base.rand(Normal(z_cur, kc.prop_sd[pn]))
            u_prp = Utils.invlogit_safe(z_prp)
            prp   = tmin + u_prp * (tmax - tmin)

            kp_prop = clone_dict(kp); kp_prop[pn] = prp

            cache_cur = build_icm_cache(t, kc, kp,      cp.L, cp.eta, cp.tau_B, cp.cache)
            ll_cur    = sum_ll_curves(curves, cache_cur; process=process, lambdas=lambdas_k)
            cache_prp = build_icm_cache(t, kc, kp_prop, cp.L, cp.eta, cp.tau_B, ICMCacheState())
            ll_prp    = sum_ll_curves(curves, cache_prp; process=process, lambdas=lambdas_k)

            lp_cur = kc.prior(kp)
            lp_prp = kc.prior(kp_prop)

            # Jacobian for t0 via u in (0,1): t0 = tmin + (tmax-tmin) * logistic(z)
            jac = log(u_prp * (1 - u_prp)) - log(u_cur * (1 - u_cur))
            a = (ll_prp + lp_prp + jac) - (ll_cur + lp_cur)
            if isfinite(a) && log(Base.rand()) < a
                cp.thetas[cp.kern_idx] = kp_prop
                cp.cache = cache_prp
                cp.acc.kernel[pn].a += 1
                kp = clone_dict(kp_prop)
            end
            cp.acc.kernel[pn].n += 1
        elseif pn == :delta
            # Positive param → log-RW
            z_cur = log(cur)
            z_prp = Base.rand(Normal(z_cur, kc.prop_sd[pn]))
            prp   = exp(z_prp)

            kp_prop = clone_dict(kp); kp_prop[pn] = prp

            cache_cur = build_icm_cache(t, kc, kp,      cp.L, cp.eta, cp.tau_B, cp.cache)
            ll_cur    = sum_ll_curves(curves, cache_cur; process=process, lambdas=lambdas_k)
            cache_prp = build_icm_cache(t, kc, kp_prop, cp.L, cp.eta, cp.tau_B, ICMCacheState())
            ll_prp    = sum_ll_curves(curves, cache_prp; process=process, lambdas=lambdas_k)

            lp_cur = kc.prior(kp)
            lp_prp = kc.prior(kp_prop)

            q_cgpr = logpdf(LogNormal(log(prp), kc.prop_sd[pn]), cur)
            q_prgc = logpdf(LogNormal(log(cur), kc.prop_sd[pn]), prp)

            a = (ll_prp + lp_prp + q_cgpr) - (ll_cur + lp_cur + q_prgc)
            if isfinite(a) && log(Base.rand()) < a
                cp.thetas[cp.kern_idx] = kp_prop
                cp.cache = cache_prp
                cp.acc.kernel[pn].a += 1
                kp = clone_dict(kp_prop)
            end
            cp.acc.kernel[pn].n += 1
        else
            prp = Base.rand(LogNormal(log(cur), kc.prop_sd[pn]))
            kp_prop = clone_dict(kp); kp_prop[pn] = prp
            cache_cur = build_icm_cache(t, kc, kp, cp.L, cp.eta, cp.tau_B, cp.cache)
            ll_cur = sum_ll_curves(curves, cache_cur; process=process, lambdas=lambdas_k)
            cache_prp = build_icm_cache(t, kc, kp_prop, cp.L, cp.eta, cp.tau_B, ICMCacheState())
            ll_prp = sum_ll_curves(curves, cache_prp; process=process, lambdas=lambdas_k)
            lp_cur = kc.prior(kp)
            lp_prp = kc.prior(kp_prop)
            q_cgpr = logpdf(LogNormal(log(prp), kc.prop_sd[pn]), cur)
            q_prgc = logpdf(LogNormal(log(cur), kc.prop_sd[pn]), prp)
            a = (ll_prp + lp_prp + q_cgpr) - (ll_cur + lp_cur + q_prgc)
            if isfinite(a) && log(Base.rand()) < a
                cp.thetas[cp.kern_idx] = kp_prop
                cp.cache = cache_prp
                cp.acc.kernel[pn].a += 1
                kp = clone_dict(kp_prop)
            end
            cp.acc.kernel[pn].n += 1
        end
    end
    params[k] = cp
    params
end

function mh_update_L_eig(k::Int, params::Vector{Utils.ClusterParams}, kernels::Vector{Utils.KernelConfig}, t, Y_list, mh_step_L::Float64, process::Symbol, lambdas_k::Vector{Float64})
    cp = params[k]
    th = Utils.pack_L(cp.L)
    thp = th .+ Base.rand(Normal(0, mh_step_L), length(th))
    Lp = Utils.unpack_L(thp, size(cp.L, 1))
    curves = [Matrix{Float64}(Yi - cp.mu_cached) for Yi in Y_list]
    kc = kernels[cp.kern_idx]; kp = cp.thetas[cp.kern_idx]
    cache_cur = build_icm_cache(t, kc, kp, cp.L, cp.eta, cp.tau_B, cp.cache)
    ll_cur = sum_ll_curves(curves, cache_cur; process=process, lambdas=lambdas_k)
    cache_prp = build_icm_cache(t, kc, kp, Lp, cp.eta, cp.tau_B, ICMCacheState())
    ll_prp = sum_ll_curves(curves, cache_prp; process=process, lambdas=lambdas_k)
    lp_cur = sum(logpdf(Normal(0, 1), th))
    lp_prp = sum(logpdf(Normal(0, 1), thp))
    a = (ll_prp + lp_prp) - (ll_cur + lp_cur)
    if isfinite(a) && log(Base.rand()) < a
        cp.L = Lp
        cp.cache = cache_prp
        cp.acc.L.a += 1
    end
    cp.acc.L.n += 1
    params[k] = cp
    params
end

function mh_update_eta_eig(k::Int, params::Vector{Utils.ClusterParams}, kernels::Vector{Utils.KernelConfig}, t, Y_list, mh_step_eta::Float64, a_eta::Float64, b_eta::Float64, process::Symbol, lambdas_k::Vector{Float64})
    cp = params[k]
    curves = [Matrix{Float64}(Yi - cp.mu_cached) for Yi in Y_list]
    kc = kernels[cp.kern_idx]; kp = cp.thetas[cp.kern_idx]
    for j in eachindex(cp.eta)
        cur = cp.eta[j]
        prp = Base.rand(LogNormal(log(cur), mh_step_eta))
        etap = copy(cp.eta); etap[j] = prp
        cache_cur = build_icm_cache(t, kc, kp, cp.L, cp.eta, cp.tau_B, cp.cache)
        cache_prp = build_icm_cache(t, kc, kp, cp.L, etap, cp.tau_B, ICMCacheState())
        ll_cur = sum_ll_curves(curves, cache_cur; process=process, lambdas=lambdas_k)
        ll_prp = sum_ll_curves(curves, cache_prp; process=process, lambdas=lambdas_k)
        dist = InverseGamma(a_eta, 1 / b_eta)
        lp_cur = logpdf(dist, cur)
        lp_prp = logpdf(dist, prp)
        q_cgpr = logpdf(LogNormal(log(prp), mh_step_eta), cur)
        q_prgc = logpdf(LogNormal(log(cur), mh_step_eta), prp)
        a = (ll_prp + lp_prp + q_cgpr) - (ll_cur + lp_cur + q_prgc)
        if isfinite(a) && log(Base.rand()) < a
            cp.eta = etap
            cp.cache = cache_prp
            cp.acc.eta[j].a += 1
        end
        cp.acc.eta[j].n += 1
    end
    params[k] = cp
    params
end

function mh_update_tauB_eig(k::Int, params::Vector{Utils.ClusterParams}, kernels::Vector{Utils.KernelConfig}, t, Y_list, mh_step_tauB::Float64, process::Symbol, lambdas_k::Vector{Float64})
    cp = params[k]
    curves = [Matrix{Float64}(Yi - cp.mu_cached) for Yi in Y_list]
    kc = kernels[cp.kern_idx]; kp = cp.thetas[cp.kern_idx]
    cur = cp.tau_B
    prp = Base.rand(LogNormal(log(cur), mh_step_tauB))
    cache_cur = build_icm_cache(t, kc, kp, cp.L, cp.eta, cur, cp.cache)
    cache_prp = build_icm_cache(t, kc, kp, cp.L, cp.eta, prp, ICMCacheState())
    ll_cur = sum_ll_curves(curves, cache_cur; process=process, lambdas=lambdas_k)
    ll_prp = sum_ll_curves(curves, cache_prp; process=process, lambdas=lambdas_k)
    lp_cur = -log1p(cur)
    lp_prp = -log1p(prp)
    q_cgpr = logpdf(LogNormal(log(prp), mh_step_tauB), cur)
    q_prgc = logpdf(LogNormal(log(cur), mh_step_tauB), prp)
    a = (ll_prp + lp_prp + q_cgpr) - (ll_cur + lp_cur + q_prgc)
    if isfinite(a) && log(Base.rand()) < a
        cp.tau_B = prp
        cp.cache = cache_prp
        cp.acc.tauB.a += 1
    end
    cp.acc.tauB.n += 1
    params[k] = cp
    params
end

function cc_switch_kernel_eig(k::Int, params::Vector{Utils.ClusterParams}, kernels::Vector{Utils.KernelConfig}, t, Y_list, process::Symbol, lambdas_k::Vector{Float64})
    cp = params[k]
    Mmod = length(kernels)
    p_m = fill(1.0 / Mmod, Mmod)
    theta_draws = [i == cp.kern_idx ? clone_dict(cp.thetas[i]) : kernels[i].pstar() for i in 1:Mmod]
    curves = [Matrix{Float64}(Yi - cp.mu_cached) for Yi in Y_list]
    logw = similar(p_m)
    for m in 1:Mmod
        kc_m = kernels[m]; kp_m = theta_draws[m]
        cache_m = build_icm_cache(t, kc_m, kp_m, cp.L, cp.eta, cp.tau_B, ICMCacheState())
        ll_m = sum_ll_curves(curves, cache_m; process=process, lambdas=lambdas_k)
        logw[m] = log(p_m[m]) + ll_m
        if m == cp.kern_idx
            cp.cache = cache_m
        end
    end
    w = exp.(logw .- maximum(logw))
    w ./= sum(w)
    new_idx = sample(1:Mmod, Weights(w))
    cp.kern_idx = new_idx
    cp.thetas = theta_draws
    kc = kernels[new_idx]; kp = cp.thetas[new_idx]
    cp.cache = build_icm_cache(t, kc, kp, cp.L, cp.eta, cp.tau_B, cp.cache)
    params[k] = cp
    params
end

end # module
