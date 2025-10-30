module TProcess

using LinearAlgebra, Random, Distributions
using ..ICMCache: ICMCacheState

# Compute quadratic form using cached eig/Cholesky blocks
function _quad_from_cache(y_resid::AbstractMatrix, cache::ICMCacheState)
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
    return quad
end

# Log-likelihood for t-process residual (matrix form): y ~ N(0, Σ/λ)
function loglik_residual_tp_matrix(e::AbstractMatrix, cache::ICMCacheState, λ::Float64)
    P = size(e, 1)
    M = size(e, 2)
    d = P * M
    quad = _quad_from_cache(e, cache)
    logdet_Sigma = cache.logdet_sum
    return -0.5 * (d * log(2π) + logdet_Sigma - d * log(λ) + λ * quad)
end

# Gibbs update for latent scales λ_n given residuals and caches
function sample_lambda!(λ::Vector{Float64}, ν::Float64, Z::Vector{Int}, residuals::Vector{Matrix{Float64}}, params)
    d = size(residuals[1], 1) * size(residuals[1], 2)
    a = 0.5 * (ν + d)
    @inbounds for n in eachindex(residuals)
        k = Z[n]
        cache = params[k].cache
        q = _quad_from_cache(residuals[n], cache)
        b = 0.5 * (ν + q)
        λ[n] = rand(Gamma(a, 1 / b))
    end
    return nothing
end

# Optional MH update for ν with log(ν-2) RW
function mh_update_nu!(ν::Float64, λ::Vector{Float64}, step::Float64=0.1)
    η_cur = log(ν - 2)
    η_prop = η_cur + step * randn()
    ν_prop = 2 + exp(η_prop)

    function lp(νloc)
        a = νloc / 2
        s = 0.0
        for λn in λ
            s += a * log(a) - loggamma(a) + (a - 1) * log(λn) - a * λn
        end
        return s
    end

    logacc = lp(ν_prop) - lp(ν) + (η_prop - η_cur)
    if log(rand()) < logacc
        return ν_prop
    else
        return ν
    end
end

end # module


