module ICMCache

using ..Utils
using LinearAlgebra

export ICMCacheState, build_icm_cache, fast_icm_loglik_curve

mutable struct ICMCacheState
    Ux_x::Union{Nothing,Matrix{Float64}}
    lam_x::Union{Nothing,Vector{Float64}}
    Bshape::Union{Nothing,Matrix{Float64}}
    chol_list::Union{Nothing,Vector{Cholesky{Float64,Matrix{Float64}}}}
    logdet_sum::Float64
    key_kx::Union{Nothing,String}
    key_B::Union{Nothing,String}
    tau::Union{Nothing,Float64}
    eta::Union{Nothing,Vector{Float64}}
    function ICMCacheState()
        new(nothing, nothing, nothing, nothing, 0.0, nothing, nothing, nothing, nothing)
    end
end

function kernel_key(kcfg::Utils.KernelConfig, kp::Dict{Symbol,Float64})
    vals = [string(kp[p]) for p in kcfg.pnames]
    kcfg.name * "::" * join(vals, "|")
end

function build_icm_cache(t, kern_cfg::Utils.KernelConfig, kp::Dict{Symbol,Float64}, L::Matrix{Float64}, eta::Vector{Float64}, tau_B::Float64, cache::Union{Nothing,ICMCacheState} = nothing)
    cache = cache === nothing ? ICMCacheState() : cache
    key_kx = kernel_key(kern_cfg, kp)
    key_B = join(vec(string.(round.(L; digits = 8))), "|")
    need_kx = cache.key_kx === nothing || cache.key_kx != key_kx
    need_B = cache.key_B === nothing || cache.key_B != key_B || cache.Bshape === nothing
    need_tau_eta = cache.tau === nothing || cache.tau != tau_B || cache.eta === nothing || any(cache.eta .!= eta)

    if need_kx
        Kx = kern_cfg.fun(t, kp)
        P_exp = Utils.nloc(t)
        size(Kx, 1) == P_exp && size(Kx, 2) == P_exp || error("Kernel matrix has wrong size")
        eig = eigen(Symmetric(Kx))
        cache.Ux_x = eig.vectors
        cache.lam_x = max.(eig.values, 1e-12)
        cache.key_kx = key_kx
    end

    if need_B
        Bshape = L * transpose(L)
        trB = tr(Bshape)
        if trB > 0
            Bshape .= Bshape .* (size(Bshape, 1) / trB)
        end
        cache.Bshape = Bshape
        cache.key_B = key_B
    end

    if need_kx || need_B || need_tau_eta || cache.chol_list === nothing
        cache.lam_x === nothing && error("Cache missing lam_x")
        P_need = length(cache.lam_x)
        M = length(eta)
        cache.chol_list = Vector{Cholesky{Float64,Matrix{Float64}}}(undef, P_need)
        logdet_sum = 0.0
        Deta = Diagonal(eta)
        for j in 1:P_need
            Sj = tau_B * cache.lam_x[j] * cache.Bshape + Deta
            ok = false
            jitter = 1e-8
            chol = nothing
            for _ in 1:6
                try
                    chol = cholesky(Symmetric(Sj + jitter * I); check = false)
                    ok = true
                    break
                catch
                    jitter *= 10
                end
            end
            ok || error("Cholesky failed for block")
            cache.chol_list[j] = chol
            logdet_sum += 2 * sum(log, diag(chol.U))
        end
        cache.logdet_sum = logdet_sum
        cache.tau = tau_B
        cache.eta = copy(eta)
    end
    cache
end

function fast_icm_loglik_curve(y_resid::AbstractMatrix, cache::ICMCacheState)
    cache.Ux_x === nothing && error("Cache not built: missing Ux_x")
    cache.chol_list === nothing && error("Cache not built: missing chol_list")
    P_y = size(y_resid, 1)
    P_ch = length(cache.chol_list)
    P_y == P_ch || error("Cache dimension mismatch")
    Ytil = transpose(cache.Ux_x) * Matrix{Float64}(y_resid)
    quad = 0.0
    for j in 1:P_ch
        chol = cache.chol_list[j]
        v = Ytil[j, :]
        x = chol \ v
        quad += dot(v, x)
    end
    -0.5 * (P_ch * size(y_resid, 2) * log(2 * pi) + cache.logdet_sum + quad)
end

end # module
