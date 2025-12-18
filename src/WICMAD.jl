"""
WICMAD ( Wavelet Intrinsic Coregionalization Model Anomaly Detector)

This module implements a Bayesian nonparametric clustering method for multivariate functional data.

Key features:
- Dirichlet Process (DP) prior for clustering
- Wavelet-based mean function estimation with Besov spike-and-slab priors
- Gaussian Process (GP) residual model with intrinsic coregionalization
- Multiple kernel types for flexible covariance modeling
- Semi-supervised learning with revealed normal samples
- Automatic wavelet selection

The algorithm uses MCMC sampling with:
- Slice sampling for cluster assignments
- Metropolis-Hastings updates for kernel parameters and ICM parameters
- Gibbs sampling for wavelet coefficients
- Carlin-Chib sampler for kernel selection
"""
module WICMAD

# Core dependencies
using LinearAlgebra      # Matrix operations, eigendecomposition
using Distributions      # Probability distributions
using StatsBase          # Statistical functions
using StatsFuns          # Statistical functions (logit, logistic)
using Random             # Random number generation
using Wavelets           # Wavelet transforms
using Wavelets: WT       # Wavelet types
using Distances          # Distance metrics (for kernel functions)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

"""
    init_diagnostics(diagnostics::Bool, keep::Int, n_iter::Int)

Initialize diagnostic tracking structures for MCMC monitoring.
Tracks: number of occupied clusters, alpha parameter, and log-likelihood.
"""
function init_diagnostics(diagnostics::Bool, keep::Int, n_iter::Int)
    diagnostics || return nothing
    Dict(
        :global => Dict(
            :K_occ => fill(Float64(NaN), keep),
            :alpha => fill(Float64(NaN), keep),
            :loglik => fill(Float64(NaN), keep),
            :K_occ_all => fill(Float64(NaN), n_iter)
        )
    )
end

# =============================================================================
# KERNEL FUNCTIONS
# =============================================================================

"""
Kernel configuration structure for Gaussian Process covariance functions.

Each kernel defines:
- name: String identifier
- fun: Function that computes the kernel matrix K(t, t')
- pnames: Parameter names (e.g., [:l_scale, :period])
- prior: Log-prior function for parameters
- pstar: Function to draw initial parameter values
- prop_sd: Proposal standard deviations for MH updates
"""
struct KernelConfig
    name::String
    fun::Function
    pnames::Vector{Symbol}
    prior::Function
    pstar::Function
    prop_sd::Dict{Symbol,Float64}
end

"""
    k_sqexp(t, l_scale)

Squared Exponential (RBF) kernel: k(t, t') = exp(-0.5 * ||t - t'||² / ℓ²)
Produces smooth, infinitely differentiable functions.
"""
function k_sqexp(t, l_scale)
    D2 = dist_rows(t).^2
    @. exp(-0.5 * D2 / (l_scale^2))
end

"""
    k_mat32(t, l_scale)

Matérn 3/2 kernel: k(t, t') = (1 + √3r) * exp(-√3r) where r = ||t - t'|| / ℓ
Produces once-differentiable functions (less smooth than SE).
"""
function k_mat32(t, l_scale)
    D = dist_rows(t)
    r = D ./ l_scale
    a = sqrt(3.0) .* r
    @. (1 + a) * exp(-a)
end

"""
    k_mat52(t, l_scale)

Matérn 5/2 kernel: k(t, t') = (1 + √5r + 5r²/3) * exp(-√5r)
Produces twice-differentiable functions (smoother than Mat32, less smooth than SE).
"""
function k_mat52(t, l_scale)
    D = dist_rows(t)
    r = D ./ l_scale
    a = sqrt(5.0) .* r
    @. (1 + a + 5 * r^2 / 3) * exp(-a)
end

"""
    k_periodic(t, l_scale, period)

Periodic kernel: k(t, t') = exp(-2 * sin²(π|t-t'|/p) / ℓ²)
Captures periodic patterns with period p.
"""
function k_periodic(t, l_scale, period)
    D = dist_rows(t)
    @. exp(-2 * sinpi(D / period)^2 / (l_scale^2))
end

"""
    k_rq(t, l_scale, alpha)

Rational Quadratic kernel: k(t, t') = (1 + ||t-t'||²/(2αℓ²))^(-α)
Generalizes SE kernel (α→∞) and allows varying smoothness.
"""
function k_rq(t, l_scale, alpha)
    D2 = dist_rows(t).^2
    @. (1 + D2 / (2 * alpha * l_scale^2))^(-alpha)
end

"""
    k_powexp(t, l_scale, kappa)

Powered Exponential kernel: k(t, t') = exp(-(||t-t'||/ℓ)^κ) where κ ∈ (0,2]
Allows control over smoothness via κ (κ=2 gives SE, κ=1 gives exponential).
"""
function k_powexp(t, l_scale, kappa)
    D = dist_rows(t)
    r = D ./ l_scale
    @. exp(-(r^kappa))
end

@inline function _ell_pc3(t::AbstractVector{<:Real},
                          ℓ1::Real, ℓ2::Real, ℓ3::Real)
    P = length(t)
    ℓ = similar(t, Float64)
    tmin, tmax = minimum(t), maximum(t)
    b1 = tmin + (tmax - tmin) / 3
    b2 = tmin + 2*(tmax - tmin) / 3
    @inbounds for i in 1:P
        ti = t[i]
        ℓ[i] = ti < b1 ? ℓ1 : (ti < b2 ? ℓ2 : ℓ3)
    end
    return ℓ
end

function k_gibbs_pc3(t::AbstractVector{<:Real}, ℓ1::Real, ℓ2::Real, ℓ3::Real)
    P = length(t)
    ℓ = _ell_pc3(t, ℓ1, ℓ2, ℓ3)
    K = Matrix{Float64}(undef, P, P)
    @inbounds for i in 1:P
        ℓi = ℓ[i]
        K[i,i] = 1.0
        for j in (i+1):P
            ℓj = ℓ[j]
            denom = ℓi^2 + ℓj^2
            pre   = sqrt( (2.0*ℓi*ℓj) / denom )
            r2    = (t[i]-t[j])^2
            val   = pre * exp( - r2/denom )
            K[i,j] = val
            K[j,i] = val
        end
    end
    return K
end

function _gate_sigma(t::AbstractVector{<:Real}, t0::Real, delta::Real)
    σ = similar(t, Float64)
    invδ = 1.0 / delta
    @inbounds for i in 1:length(t)
        σ[i] = logistic( (t[i] - t0) * invδ )
    end
    σ
end

@inline function _m52_corr(r::Float64, ℓ::Float64)
    a = sqrt(5.0) * r / ℓ
    (1 + a + (a^2)/3) * exp(-a)
end

function k_changepoint_m52(t::AbstractVector{<:Real}, t0::Real, delta::Real,
                           ℓ_left::Real, ℓ_right::Real)
    P = length(t)
    σ = _gate_sigma(t, t0, delta)
    K = Matrix{Float64}(undef, P, P)
    @inbounds for i in 1:P
        σi = σ[i]
        Ki = @view K[i, :]
        Ki[i] = 1.0
        for j in (i+1):P
            σj = σ[j]
            r  = abs(t[i]-t[j])
            kL = _m52_corr(r, ℓ_left)
            kR = _m52_corr(r, ℓ_right)
            val = (1-σi)*(1-σj)*kL + σi*σj*kR
            K[i,j] = val
            K[j,i] = val
        end
    end
    return K
end

function make_kernels()
    KernelConfig[
        KernelConfig(
            "SE",
            (t, par) -> k_sqexp(t, par[:l_scale]),
            [:l_scale],
            par -> logpdf(Gamma(2, 1 / 2), par[:l_scale]),
            () -> Dict(:l_scale => Base.rand(Gamma(2, 1 / 2))),
            Dict(:l_scale => 0.20),
        ),
        KernelConfig(
            "Mat32",
            (t, par) -> k_mat32(t, par[:l_scale]),
            [:l_scale],
            par -> logpdf(Gamma(2, 1 / 2), par[:l_scale]),
            () -> Dict(:l_scale => Base.rand(Gamma(2, 1 / 2))),
            Dict(:l_scale => 0.20),
        ),
        KernelConfig(
            "Mat52",
            (t, par) -> k_mat52(t, par[:l_scale]),
            [:l_scale],
            par -> logpdf(Gamma(2, 1 / 2), par[:l_scale]),
            () -> Dict(:l_scale => Base.rand(Gamma(2, 1 / 2))),
            Dict(:l_scale => 0.20),
        ),
        KernelConfig(
            "Periodic",
            (t, par) -> k_periodic(t, par[:l_scale], par[:period]),
            [:l_scale, :period],
            par -> logpdf(Gamma(3, 1 / 2), par[:l_scale]) + logpdf(Beta(5, 5), par[:period]),
            () -> Dict(:l_scale => Base.rand(Gamma(3, 1 / 2)), :period => Base.rand(Beta(5, 5))),
            Dict(:l_scale => 0.20, :period => 0.20),
        ),
        KernelConfig(
            "RQ",
            (t, par) -> k_rq(t, par[:l_scale], par[:alpha]),
            [:l_scale, :alpha],
            par -> (logpdf(Gamma(2, 1 / 2), par[:l_scale]) +
                    logpdf(Gamma(2, 1.0), par[:alpha])),
            () -> Dict(:l_scale => Base.rand(Gamma(2, 1 / 2)),
                        :alpha   => Base.rand(Gamma(2, 1.0))),
            Dict(:l_scale => 0.20, :alpha => 0.25),
        ),
        KernelConfig(
            "PowExp",
            (t, par) -> k_powexp(t, par[:l_scale], par[:kappa]),
            [:l_scale, :kappa],
            par -> (logpdf(Gamma(2, 1 / 2), par[:l_scale]) +
                    (logpdf(Beta(2, 2), par[:kappa] / 2) - log(2.0))),
            () -> Dict(:l_scale => Base.rand(Gamma(2, 1 / 2)),
                        :kappa   => 2 * Base.rand(Beta(2, 2))),
            Dict(:l_scale => 0.20, :kappa => 0.25),
        ),
        KernelConfig(
            "GibbsPC3",
            (t, par) -> k_gibbs_pc3(t, par[:ell1], par[:ell2], par[:ell3]),
            [:ell1, :ell2, :ell3],
            par -> ( logpdf(Gamma(2, 1/2), par[:ell1]) +
                     logpdf(Gamma(2, 1/2), par[:ell2]) +
                     logpdf(Gamma(2,  1/2), par[:ell3]) ),
            () -> Dict(:ell1 => Base.rand(Gamma(2, 1/2)),
                       :ell2 => Base.rand(Gamma(2, 1/2)),
                       :ell3 => Base.rand(Gamma(2, 1/2))),
            Dict(:ell1 => 0.20, :ell2 => 0.20, :ell3 => 0.20),
        ),
        KernelConfig(
            "CP_M52",
            (t, par) -> k_changepoint_m52(t, par[:t0], par[:delta], par[:ell_left], par[:ell_right]),
            [:t0, :delta, :ell_left, :ell_right],
            par -> (
                logpdf(Gamma(2, 1/2), par[:ell_left])  +
                logpdf(Gamma(2, 1/2), par[:ell_right]) +
                logpdf(Gamma(2, 1.0),  par[:delta])
            ),
            () -> Dict(:t0 => 0.5,
                       :delta => Base.rand(Gamma(2, 1.0)),
                       :ell_left  => Base.rand(Gamma(2, 1/2)),
                       :ell_right => Base.rand(Gamma(2, 1/2))),
            Dict(:t0 => 0.20, :delta => 0.20, :ell_left => 0.20, :ell_right => 0.20),
        ),
    ]
end

# =============================================================================
# BASIC UTILITY FUNCTIONS
# =============================================================================

"""
    scale_t01(t)

Scale time vector to [0, 1] interval.

This normalization is important for kernel functions which assume
the time domain is on a standard scale.
"""
function scale_t01(t::AbstractVector)
    mn = minimum(t); mx = maximum(t)
    rng = mx - mn
    rng <= 0 ? collect(t) : (collect(t) .- mn) ./ rng
end

function scale_t01(t::AbstractMatrix)
    T = Matrix{Float64}(t)
    for j in axes(T, 2)
        col = view(T, :, j)
        mn = minimum(col); mx = maximum(col)
        rng = mx - mn
        if rng > 0
            col .= (col .- mn) ./ rng
        end
    end
    T
end

scale_t01(t) = scale_t01(collect(t))

function dist_rows(t)
    if isa(t, AbstractVector)
        T = reshape(Float64.(t), :, 1)
    else
        T = Matrix{Float64}(t)
    end
    pairwise(Euclidean(), T; dims = 1)
end

nloc(t::AbstractVector) = length(t)
nloc(t::AbstractMatrix) = size(t, 1)

"""
    ensure_dyadic_J(P, J)

Ensure wavelet decomposition level J is compatible with signal length P.

If J is not provided, computes J = floor(log2(P)).
If P is not a power of 2, adjusts J to the closest compatible value.
"""
function ensure_dyadic_J(P::Integer, J::Union{Nothing, Integer, AbstractFloat})
    Jval = isnothing(J) ? log2(P) : Float64(J)
    Jint = round(Int, Jval)
    
    expected_P = 2^Jint
    if abs(P - expected_P) > eps(Float64) * max(1, P)
        Jint = floor(Int, log2(P))
        expected_P = 2^Jint
        if abs(P - expected_P) > eps(Float64) * max(1, P)
            Jint = round(Int, log2(P))
        end
    end
    Jint
end

logit_safe(x) = logit(clamp(x, eps(Float64), 1 - eps(Float64)))
invlogit_safe(z) = logistic(z)

"""
    stick_to_pi(v)

Convert stick-breaking weights to probabilities.

Given stick-breaking weights v = [v₁, v₂, ..., v_K], computes:
π₁ = v₁
π₂ = v₂(1-v₁)
π₃ = v₃(1-v₁)(1-v₂)
...
This implements the Dirichlet Process stick-breaking construction.
"""
function stick_to_pi(v::AbstractVector)
    K = length(v)
    pi = zeros(Float64, K)
    tail = 1.0  # Remaining stick length
    for k in 1:K
        pi[k] = v[k] * tail  # Probability for cluster k
        tail *= (1 - v[k])   # Update remaining stick length
    end
    pi
end

"""
    extend_sticks_until(v, alpha, threshold)

Extend stick-breaking weights until tail probability < threshold.

Used in slice sampling to ensure we have enough clusters to satisfy
the slice constraint. The tail probability is the probability of all
clusters beyond the current ones.
"""
function extend_sticks_until(v::Vector{Float64}, alpha::Float64, threshold::Float64)
    tail = prod(1 .- v)  # Current tail probability
    while tail > threshold
        v_new = Base.rand(Beta(1, alpha))  # Draw new stick weight
        push!(v, v_new)
        tail *= (1 - v_new)  # Update tail probability
    end
    v
end

"""
    update_v_given_z(v, z, alpha)

Update stick-breaking weights given cluster assignments (Gibbs update).

For each stick k, samples v_k ~ Beta(1 + n_k, α + n_{>k})
where n_k is the number of observations in cluster k and n_{>k} is
the number in clusters > k. This is the conjugate update for DP.
"""
function update_v_given_z(v::Vector{Float64}, z::Vector{Int}, alpha::Float64)
    K = length(v)
    counts = [count(==(k), z) for k in 1:K]  # Number of observations per cluster
    tail_counts = reverse(cumsum(reverse(counts)))  # Cumulative counts from the end
    for k in 1:K
        a = 1 + counts[k]  # Beta shape parameter
        b = alpha + (k < K ? tail_counts[k + 1] : 0)  # Beta rate parameter
        v[k] = Base.rand(Beta(a, b))
    end
    v
end

function pack_L(L::AbstractMatrix)
    m, n = size(L)
    @assert m == n "L must be square"
    out = Vector{Float64}(undef, div(m * (m + 1), 2))
    idx = 1
    for j in 1:m
        for i in j:m
            out[idx] = L[i, j]
            idx += 1
        end
    end
    out
end

function unpack_L(theta::AbstractVector, m::Integer)
    expected = div(m * (m + 1), 2)
    length(theta) == expected || error("theta length $(length(theta)) incompatible with m=$m")
    L = zeros(Float64, m, m)
    idx = 1
    for j in 1:m
        for i in j:m
            L[i, j] = theta[idx]
            idx += 1
        end
    end
    for i in 1:m
        L[i, i] = abs(L[i, i]) + 1e-8
    end
    L
end

as_num_mat(A) = Matrix{Float64}(A)

function eig_Kx(Kx::AbstractMatrix)
    ee = eigen(Symmetric(Matrix{Float64}(Kx)))
    (V = ee.vectors, s = max.(ee.values, 0.0))
end

function eig_Bshape(L::AbstractMatrix, M::Int)
    Bshape = Matrix(L) * transpose(Matrix(L))
    trB = tr(Bshape)
    if trB > 0
        Bshape .= Bshape .* (M / trB)
    end
    ee = eigen(Symmetric(Bshape))
    (U = ee.vectors, lam = max.(ee.values, 0.0))
end



"""
    normalize_t(t, P)

Normalize time vector to match signal length P.

Handles both vector and matrix inputs, ensuring the time dimension
matches the signal length P.
"""
function normalize_t(t, P::Integer)
    if t === nothing
        error("t is nothing")
    end
    if isa(t, AbstractVector)
        length(t) == P || error("t vector has length $(length(t)) but P=$P")
        return collect(Float64.(t))
    end
    T = Matrix{Float64}(t)
    if size(T, 1) == P
        return T
    elseif size(T, 2) == P
        return transpose(T)
    else
        error("t has incompatible shape: $(size(T,1)) x $(size(T,2)); need P x d or length-P (P=$P)")
    end
end

# =============================================================================
# INITIALIZATION STRUCTURES AND FUNCTIONS
# =============================================================================

mutable struct AcceptCount
    a::Int
    n::Int
    AcceptCount() = new(0, 0)
end

mutable struct AcceptanceTracker
    kernel::Dict{Symbol,AcceptCount}
    L::AcceptCount
    eta::Vector{AcceptCount}
    tauB::AcceptCount
end

"""
Wavelet parameter structure for Besov spike-and-slab prior.

Stores:
- pi_level: Probability of activation per detail level
- g_level: Shrinkage parameter per detail level
- gamma_ch: Binary indicators (0/1) for each coefficient per channel
- Hyperparameters: kappa_pi, c2, tau_pi (control sparsity)
- a_g, b_g: InverseGamma prior for g_level
- a_sig, b_sig, a_tau, b_tau: Priors for noise variance hierarchy
"""
mutable struct WaveletParams
    lev_names::Vector{String}
    pi_level::Dict{String,Float64}
    g_level::Dict{String,Float64}
    gamma_ch::Vector{Vector{Int}}

    kappa_pi::Float64
    c2::Float64
    tau_pi::Float64

    a_g::Float64
    b_g::Float64

    a_sig::Float64
    b_sig::Float64
    a_tau::Float64
    b_tau::Float64
end

"""
Cluster parameter structure.

Stores all parameters for a single cluster:
- wpar: Wavelet parameters (mean function)
- kern_idx: Current kernel type index
- thetas: Kernel hyperparameters (one dict per kernel type)
- L: Lower-triangular matrix for cross-channel covariance B = L*L'
- eta: Channel-specific noise variances
- tau_B: Scale parameter for cross-channel covariance
- beta_ch: Wavelet coefficients (per channel)
- sigma2: Noise variances (per channel)
- tau_sigma: Global scale for sigma2
- cache: ICM cache for efficient likelihood computation
- mu_cached: Cached mean function (reconstructed from beta_ch)
- acc: Acceptance tracking for MH updates
"""
mutable struct ClusterParams
    wpar::WaveletParams
    kern_idx::Int
    thetas::Vector{Dict{Symbol,Float64}}
    L::Matrix{Float64}
    eta::Vector{Float64}
    tau_B::Float64
    beta_ch::Vector{Vector{Float64}}
    sigma2::Vector{Float64}
    tau_sigma::Float64
    cache::Any
    mu_cached::Union{Nothing,Matrix{Float64}}
    mu_cached_iter::Int
    acc::AcceptanceTracker
    g_hyp::Union{Nothing,Dict{String,Dict{String,Float64}}}
end

function draw_empty_acc(M::Int, kernels::Vector{KernelConfig})
    pset = Set{Symbol}()
    for kc in kernels
        foreach(pn -> push!(pset, pn), kc.pnames)
    end
    kernel_counts = Dict{Symbol,AcceptCount}(pn => AcceptCount() for pn in pset)
    AcceptanceTracker(kernel_counts, AcceptCount(), [AcceptCount() for _ in 1:M], AcceptCount())
end

function force_scaling_active!(wpar::WaveletParams, maps_per_channel::Vector)
    for m in 1:length(maps_per_channel)
        map_m = maps_per_channel[m]
        for (sym, ur) in map_m.idx
            if startswith(String(sym), "s")
                wpar.gamma_ch[m][ur] .= 1
            end
        end
    end
end

"""
    draw_new_cluster_params(M, P, t, kernels; wf, J, boundary)

Initialize parameters for a new cluster.

Draws initial values for:
- Wavelet parameters (pi_level, g_level, gamma_ch, beta_ch)
- Kernel type and hyperparameters
- ICM parameters (L, eta, tau_B)
- Noise variances (sigma2, tau_sigma)

All parameters are drawn from their priors.
"""
function draw_new_cluster_params(M::Int, P::Int, t, kernels::Vector{KernelConfig}; wf::String = "sym8", J = nothing, boundary::String = "periodic")
    Jv = ensure_dyadic_J(P, J)
    zeros_mat = zeros(P, M)
    tmp = wt_forward_mat(zeros_mat; wf = wf, J = Jv, boundary = boundary)
    lev_names = [String(k) for k in keys(tmp[1].map.idx)]
    det_names = filter(name -> startswith(name, "d"), lev_names)
    ncoeff = length(tmp[1].coeff)

    kappa_pi = 1.0
    c2       = 1.0
    tau_pi   = 40.0
    a_g, b_g = 2.0, 2.0
    a_sig, b_sig = 2.0, 2.0
    a_tau, b_tau = 2.0, 2.0

    pi_level = Dict{String,Float64}()
    g_level  = Dict{String,Float64}()
    for name in det_names
        j = parse(Int, replace(name, "d"=>""))
        m_j = kappa_pi * 2.0^(-c2 * j)
        pi_level[name] = clamp(m_j, 1e-6, 1 - 1e-6)
        g0 = 1.0
        g_level[name] = g0
    end

    beta_ch  = [zeros(ncoeff) for _ in 1:M]
    gamma_ch = [zeros(Int, ncoeff) for _ in 1:M]

    wpar = WaveletParams(
        lev_names, pi_level, g_level, gamma_ch,
        kappa_pi, c2, tau_pi,
        a_g, b_g,
        a_sig, b_sig, a_tau, b_tau
    )
    thetas = [kc.pstar() for kc in kernels]
    ClusterParams(
        wpar,
        Base.rand(1:length(kernels)),
        thetas,
        Matrix{Float64}(I, M, M),
        fill(0.05, M),
        1.0,
        beta_ch,
        fill(0.05, M),
        1.0,
        nothing,
        nothing,
        0,
        draw_empty_acc(M, kernels),
        nothing,
    )
end

function ensure_complete_cache!(cp::ClusterParams, kernels::Vector{KernelConfig}, t, M::Int)
    if length(cp.eta) != M || any(.!isfinite.(cp.eta))
        cp.eta = fill(0.05, M)
    end
    if !isfinite(cp.tau_B) || cp.tau_B <= 0
        cp.tau_B = 1.0
    end
    if cp.kern_idx < 1 || cp.kern_idx > length(kernels)
        cp.kern_idx = 1
    end
    if length(cp.thetas) != length(kernels)
        cp.thetas = [kc.pstar() for kc in kernels]
    end
    if cp.thetas[cp.kern_idx] === nothing
        cp.thetas[cp.kern_idx] = kernels[cp.kern_idx].pstar()
    end
    cp
end

# =============================================================================
# WAVELET OPERATIONS
# =============================================================================

"""
Wavelet map structure storing information about wavelet decomposition.

Contains:
- J: Decomposition level
- wf: Wavelet family name
- boundary: Boundary condition
- P: Original signal length
- idx: Dictionary mapping level names (e.g., :d1, :d2, :sJ) to coefficient indices
"""
struct WaveletMap
    J::Int
    wf::String
    boundary::String
    P::Int
    idx::Dict{Symbol,UnitRange{Int}}
end

"""
Wavelet coefficients structure.

Stores the flattened coefficient vector and the map that describes
how coefficients are organized by level (detail vs approximation).
"""
struct WaveletCoefficients
    coeff::Vector{Float64}
    map::WaveletMap
end

const _WAVELET_ALIASES = Dict(
    "haar" => :haar,
    "db1" => :haar,
    "db2" => :db2,
    "db3" => :db3,
    "db4" => :db4,
    "db5" => :db5,
    "db6" => :db6,
    "db7" => :db7,
    "db8" => :db8,
    "coif2" => :coif2,
    "coif4" => :coif4,
    "coif6" => :coif6,
    "sym4" => :sym4,
    "sym5" => :sym5,
    "sym6" => :sym6,
    "sym7" => :sym7,
    "sym8" => :sym8,
    "batt2" => :batt2,
    "batt4" => :batt4,
    "batt6" => :batt6,
)

function wavelet_from_string(wf::String, boundary::String = "periodic")
    sym = Symbol(lowercase(wf))
    sym = get(_WAVELET_ALIASES, String(sym), sym)
    if isdefined(WT, sym)
        ext = boundary == "periodic" ? WT.Periodic : WT.Symmetric
        return wavelet(getfield(WT, sym), WT.Filter, ext)
    else
        error("Unsupported wavelet family $wf")
    end
end

function extension_from_string(boundary::String)
    boundary == "periodic" && return WT.Periodic
    boundary == "reflection" && return WT.Symmetric
    error("Unsupported boundary $boundary; use \"periodic\" or \"reflection\".")
end

"""
    wt_forward_1d(y; wf, J, boundary)

Forward 1D wavelet transform.

Decomposes a signal into wavelet coefficients organized by level.
Returns detail coefficients (d1, d2, ..., dJ) and approximation coefficients (sJ).
"""
function wt_forward_1d(y::AbstractVector; wf::String = "sym8", J::Union{Nothing,Int} = nothing, boundary::String = "periodic")
    P = length(y)
    Jv = ensure_dyadic_J(P, J)
    wave = wavelet_from_string(wf, boundary)
    wt = dwt(Float64.(y), wave, Jv)
    
    coeffs_vec = Float64[]
    idx = Dict{Symbol,UnitRange{Int}}()
    offset = 0
    
    for lev in 1:Jv
        if lev == 1
            start_idx = 1
            end_idx = detailindex(P, 1, Jv) - 1
        else
            start_idx = detailindex(P, lev-1, Jv)
            end_idx = detailindex(P, lev, Jv) - 1
        end
        
        if start_idx <= end_idx
            d = wt[start_idx:end_idx]
            append!(coeffs_vec, d)
            idx[Symbol("d" * string(lev))] = offset + 1:offset + length(d)
            offset += length(d)
        end
    end
    
    approx_start = detailindex(P, Jv, Jv)
    s = wt[approx_start:end]
    append!(coeffs_vec, s)
    idx[Symbol("s" * string(Jv))] = offset + 1:offset + length(s)
    
    WaveletCoefficients(coeffs_vec, WaveletMap(Jv, wf, boundary, P, idx))
end

"""
    wt_inverse_1d(coeff_vec, map)

Inverse 1D wavelet transform.

Reconstructs the original signal from wavelet coefficients.
"""
function wt_inverse_1d(coeff_vec::AbstractVector, map::WaveletMap)
    wave = wavelet_from_string(map.wf, map.boundary)
    
    wt_reconstructed = zeros(Float64, map.P)
    
    for lev in 1:map.J
        key = Symbol("d" * string(lev))
        if haskey(map.idx, key)
            ids = map.idx[key]
            detail_coeffs = collect(coeff_vec[ids])
            
            if lev == 1
                start_idx = 1
                end_idx = detailindex(map.P, 1, map.J) - 1
            else
                start_idx = detailindex(map.P, lev-1, map.J)
                end_idx = detailindex(map.P, lev, map.J) - 1
            end
            
            if start_idx <= end_idx && length(detail_coeffs) > 0
                wt_reconstructed[start_idx:end_idx] = detail_coeffs
            end
        end
    end
    
    approx_key = Symbol("s" * string(map.J))
    if haskey(map.idx, approx_key)
        ids = map.idx[approx_key]
        approx_coeffs = collect(coeff_vec[ids])
        approx_start = detailindex(map.P, map.J, map.J)
        if length(approx_coeffs) > 0
            wt_reconstructed[approx_start:end] = approx_coeffs
        end
    end
    
    idwt(wt_reconstructed, wave, map.J)
end

function wt_forward_mat(y_mat::AbstractMatrix; wf::String = "sym8", J::Union{Nothing,Int} = nothing, boundary::String = "periodic")
    M = size(y_mat, 2)
    [wt_forward_1d(view(y_mat, :, m); wf = wf, J = J, boundary = boundary) for m in 1:M]
end

function precompute_wavelets(Y_list, wf::String, J, boundary::String)
    [wt_forward_mat(mat; wf = wf, J = J, boundary = boundary) for mat in Y_list]
end

function stack_D_from_precomp(precomp, idx::Vector{Int}, M::Int)
    ncoeff = length(precomp[idx[1]][1].coeff)
    N = length(idx)
    D_arr = Array{Float64}(undef, ncoeff, N, M)
    for (jj, i) in enumerate(idx)
        for m in 1:M
            coeffs = precomp[i][m].coeff
            D_arr[:, jj, m] = coeffs
        end
    end
    maps = [precomp[idx[1]][m].map for m in 1:M]
    (; D_arr, maps)
end

"""
    compute_mu_from_beta(beta_ch, wf, J, boundary, P)

Reconstruct mean function from wavelet coefficients.

Takes the sampled wavelet coefficients (beta_ch) and reconstructs
the mean function μ(t) via inverse wavelet transform.
"""
function compute_mu_from_beta(beta_ch::Vector{Vector{Float64}}, wf::String, J::Int, boundary::String, P::Int)
    M = length(beta_ch)
    zeros_mat = zeros(P, M)
    tmpl = wt_forward_mat(zeros_mat; wf = wf, J = J, boundary = boundary)
    mu = zeros(P, M)
    for m in 1:M
        mu[:, m] = wt_inverse_1d(beta_ch[m], tmpl[m].map)
    end
    mu
end

function ensure_gamma_length!(wpar::WaveletParams, ncoeff::Int, M::Int)
    if length(wpar.gamma_ch) != M || any(length(g) != ncoeff for g in wpar.gamma_ch)
        wpar.gamma_ch = [Int.(Base.rand(Bernoulli(0.2), ncoeff)) for _ in 1:M]
    end
end

function update_cluster_wavelet_params_besov(idx::Vector{Int}, precomp, M::Int, wpar::WaveletParams,
    sigma2_m::Vector{Float64}, tau_sigma::Float64;
    kappa_pi::Float64 = 1.0, c2::Float64 = 1.0, tau_pi::Float64 = 40.0,
    g_hyp = nothing,
    a_sig::Float64 = 2.5, b_sig::Float64 = 0.02,
    a_tau::Float64 = 2.0, b_tau::Float64 = 2.0)

    if isempty(idx)
        return (wpar = wpar,
                beta_ch  = [Float64[] for _ in 1:M],
                sigma2_m = sigma2_m,
                tau_sigma = tau_sigma,
                maps = nothing)
    end

    length(sigma2_m) == M || error("sigma2_m must be length M")
    stk = stack_D_from_precomp(precomp, idx, M)
    D = stk.D_arr
    maps = stk.maps
    ncoeff = size(D, 1)
    N = size(D, 2)
    lev_names = sort([String(k) for k in keys(maps[1].idx)])
    det_names = sort(filter(name -> startswith(name, "d"), lev_names))
    s_name = sort(filter(name -> startswith(name, "s"), lev_names))

    if isempty(wpar.pi_level)
        wpar.pi_level = Dict(name => 0.5 for name in det_names)
    end
    if isempty(wpar.g_level)
        wpar.g_level = Dict(name => 2.0 for name in det_names)
    end
    ensure_gamma_length!(wpar, ncoeff, M)

    for m in 1:M
        Dm = view(D, :, :, m)
        gam = wpar.gamma_ch[m]
        σ2 = sigma2_m[m]
        for lev in det_names
            ids = maps[m].idx[Symbol(lev)]
            isempty(ids) && continue
            π = wpar.pi_level[lev]
            g = wpar.g_level[lev]
            S1 = vec(sum(Dm[ids, :]; dims=2))
            c1 = log(π) - log1p(-π) - 0.5 * log1p(g * N)
            c2 = 0.5 / σ2 * (g / (1.0 + g * N))
            logit = c1 .+ c2 .* (S1 .^ 2)
            p1 = 1.0 ./ (1.0 .+ exp.(-clamp.(logit, -35, 35)))
            gam[ids] = Int.(Base.rand.(Bernoulli.(p1)))
        end
        if length(s_name) == 1
            ids_s = maps[m].idx[Symbol(s_name[1])]
            if !isempty(ids_s)
                gam[ids_s] .= 1
            end
        end
        wpar.gamma_ch[m] = gam
    end

    for lev in det_names
        jnum = parse(Int, replace(lev, "d" => ""))
        m_j = clamp(kappa_pi * 2^(-c2 * jnum), 1e-6, 1 - 1e-6)
        a0 = tau_pi * m_j
        b0 = tau_pi * (1 - m_j)
        n1 = 0
        n0 = 0
        for m in 1:M
            ids = maps[m].idx[Symbol(lev)]
            isempty(ids) && continue
            gm = wpar.gamma_ch[m][ids]
            n1 += count(==(1), gm)
            n0 += count(==(0), gm)
        end
        wpar.pi_level[lev] = Base.rand(Beta(a0 + n1, b0 + n0))
    end

    beta_ch = [zeros(ncoeff) for _ in 1:M]
    for m in 1:M
        Dm = view(D, :, :, m)
        gam = wpar.gamma_ch[m]
        for lev in det_names
            ids = maps[m].idx[Symbol(lev)]
            isempty(ids) && continue
            g_j = wpar.g_level[lev]
            n = N
            Dbar = vec(mean(Dm[ids, :]; dims = 2))
            shrink = n / (n + 1 / g_j)
            mean_post = shrink .* Dbar
            var_post = sigma2_m[m] / (n + 1 / g_j)
            is_on = gam[ids] .== 1
            if any(is_on)
                dists = Normal.(mean_post[is_on], sqrt(var_post))
                beta_ch[m][ids[is_on]] = Base.rand.(dists)
            end
        end
        if length(s_name) == 1
            ids_s = maps[m].idx[Symbol(s_name[1])]
            if !isempty(ids_s)
                Dbar_s = vec(mean(Dm[ids_s, :]; dims = 2))
                dists_s = Normal.(Dbar_s, sqrt(sigma2_m[m] / N))
                beta_ch[m][ids_s] = Base.rand.(dists_s)
            end
        end
    end

    for lev in det_names
        shape0 = isnothing(g_hyp) ? 2.0 : g_hyp[lev]["shape"]
        rate0  = isnothing(g_hyp) ? 2.0 : g_hyp[lev]["rate"]

        ss_over_sigma = 0.0
        n_sel_total   = 0

        for m in 1:M
            ids = maps[m].idx[Symbol(lev)]
            isempty(ids) && continue

            act = wpar.gamma_ch[m][ids] .== 1
            any(act) || continue

            βm = beta_ch[m]
            act_ids = ids[act]
            ss_over_sigma += sum( (βm[act_ids].^2) ) / sigma2_m[m]
            n_sel_total   += length(act_ids)
        end

        shape_post = shape0 + 0.5 * n_sel_total
        rate_post  = rate0  + 0.5 * ss_over_sigma
        wpar.g_level[lev] = Base.rand(InverseGamma(shape_post, rate_post))
    end

    sigma2_m_new = copy(sigma2_m)
    n_eff_m = ncoeff * N
    for m in 1:M
        Dm = view(D, :, :, m)
        resid = Dm .- beta_ch[m]
        ss_m = sum(resid .^ 2)
        shape_post = a_sig + 0.5 * n_eff_m
        rate_post = b_sig * tau_sigma + 0.5 * ss_m
        sigma2_m_new[m] = Base.rand(InverseGamma(shape_post, rate_post))
    end
    a_post = a_tau + M * a_sig
    b_post = b_tau + b_sig * sum(1 ./ sigma2_m_new)
    tau_sigma_new = Base.rand(Gamma(a_post, 1 / b_post))

    (wpar = wpar, beta_ch = beta_ch, sigma2_m = sigma2_m_new, tau_sigma = tau_sigma_new, maps = maps)
end

"""
    update_cluster_wavelet_params_besov_fullbayes(idx, precomp, M, wpar, sigma2_m, tau_sigma)

Update wavelet parameters for a cluster using full Bayesian Besov spike-and-slab prior.

This function performs Gibbs sampling for:
1. gamma (spike indicators): which coefficients are active
2. beta (wavelet coefficients): sampled from posterior given gamma
3. pi_level (sparsity): probability of activation per level
4. g_level (shrinkage): precision parameter per level
5. sigma2 (noise variance): per channel
6. tau_sigma (global scale): for sigma2

The Besov prior encourages sparsity in wavelet coefficients, with level-dependent
sparsity controlled by pi_level and shrinkage controlled by g_level.
"""
function update_cluster_wavelet_params_besov_fullbayes(
    idx::Vector{Int}, precomp, M::Int, wpar::WaveletParams,
    sigma2_m::Vector{Float64}, tau_sigma::Float64
)
    if isempty(idx)
        return (wpar=wpar,
                beta_ch=[Float64[] for _ in 1:M],
                sigma2_m=sigma2_m,
                tau_sigma=tau_sigma,
                maps=nothing)
    end

    stk   = stack_D_from_precomp(precomp, idx, M)
    D     = stk.D_arr
    maps  = stk.maps
    ncoeff, N_k, _ = size(D)

    ensure_gamma_length!(wpar, ncoeff, M)
    force_scaling_active!(wpar, maps)

    det_names = filter(name -> startswith(name, "d"), wpar.lev_names)
    s_names   = filter(name -> startswith(name, "s"), wpar.lev_names)

    for m in 1:M
        Dm = view(D, :, :, m)
        gam = wpar.gamma_ch[m]
        σ2  = sigma2_m[m]

        for lev in det_names
            ids = maps[m].idx[Symbol(lev)]
            isempty(ids) && continue
            π  = wpar.pi_level[lev]
            g  = wpar.g_level[lev]

            S1 = vec(sum(Dm[ids, :]; dims=2))
            c1 = log(π) - log1p(-π) - 0.5 * log1p(g * N_k)
            c2 = 0.5/σ2 * (g / (1.0 + g * N_k))
            logit = c1 .+ c2 .* (S1 .^ 2)

            p1 = 1.0 ./ (1.0 .+ exp.(-clamp.(logit, -35, 35)))
            gam[ids] = Int.(Base.rand.(Bernoulli.(p1)))
        end
    end

    beta_ch = [zeros(ncoeff) for _ in 1:M]
    for m in 1:M
        Dm = view(D, :, :, m)
        σ2 = sigma2_m[m]
        gam = wpar.gamma_ch[m]

        for lev in det_names
            ids = maps[m].idx[Symbol(lev)]
            isempty(ids) && continue
            g = wpar.g_level[lev]

            ybar  = vec(mean(Dm[ids, :]; dims=2))
            vstar = σ2 / (N_k + 1.0/g)
            μstar = (N_k / (N_k + 1.0/g)) .* ybar

            on  = (gam[ids] .== 1)
            off = .!on
            if any(on)
                beta_ch[m][ids[on]] = Base.rand.(Normal.(μstar[on], sqrt(vstar)))
            end
            if any(off)
                beta_ch[m][ids[off]] .= 0.0
            end
        end

        for sname in s_names
            ids_s = maps[m].idx[Symbol(sname)]
            isempty(ids_s) && continue
            ybar_s = vec(mean(Dm[ids_s, :]; dims=2))
            v_s    = σ2 / N_k
            beta_ch[m][ids_s] = Base.rand.(Normal.(ybar_s, sqrt(v_s)))
            wpar.gamma_ch[m][ids_s] .= 1
        end
    end

    for lev in det_names
        a_g, b_g = wpar.a_g, wpar.b_g
        n_on = 0
        sum_term = 0.0
        for m in 1:M
            σ2 = sigma2_m[m]
            ids = maps[m].idx[Symbol(lev)]
            isempty(ids) && continue
            on_mask = (wpar.gamma_ch[m][ids] .== 1)
            if any(on_mask)
                β_on = beta_ch[m][ids[on_mask]]
                n_on += length(β_on)
                sum_term += sum(@. (β_on^2) / σ2)
            end
        end
        if n_on > 0
            shape = a_g + 0.5 * n_on
            rate  = b_g + 0.5 * sum_term
            rate = max(rate, 1e-10)
            wpar.g_level[lev] = 1.0 / Base.rand(Gamma(shape, 1.0/rate))
        else
            wpar.g_level[lev] = max(wpar.g_level[lev], 1e-6)
        end
    end

    for lev in det_names
        j = parse(Int, replace(lev, "d"=>""))
        m_j = wpar.kappa_pi * 2.0^(-wpar.c2 * j)
        n1, n0 = 0, 0
        for m in 1:M
            ids = maps[m].idx[Symbol(lev)]
            isempty(ids) && continue
            gvec = wpar.gamma_ch[m][ids]
            n1 += sum(gvec)
            n0 += length(gvec) - sum(gvec)
        end
        α = wpar.tau_pi * m_j + n1
        β = wpar.tau_pi * (1.0 - m_j) + n0
        wpar.pi_level[lev] = Base.rand(Beta(α, β))
    end

    sigma2_m_new = similar(sigma2_m)
    N_eff = N_k * ncoeff
    for m in 1:M
        Dm = view(D, :, :, m)
        resid = Dm .- beta_ch[m]
        ss_m = sum(resid.^2)
        a_post = wpar.a_sig + 0.5 * N_eff
        b_post = wpar.b_sig * tau_sigma + 0.5 * ss_m
        sigma2_m_new[m] = 1.0 / Base.rand(Gamma(a_post, 1.0 / b_post))
    end
    aτ = wpar.a_tau + M * wpar.a_sig
    bτ = wpar.b_tau + wpar.b_sig * sum(1.0 ./ sigma2_m_new)
    tau_sigma_new = Base.rand(Gamma(aτ, 1.0 / bτ))

    return (wpar=wpar, beta_ch=beta_ch, sigma2_m=sigma2_m_new, tau_sigma=tau_sigma_new, maps=maps)
end

# =============================================================================
# ICM CACHE
# =============================================================================

"""
ICM (Intrinsic Coregionalization Model) cache structure.

Caches expensive computations for efficient likelihood evaluation:
- Ux_x, lam_x: Eigendecomposition of kernel matrix K_x
- Bshape: Normalized cross-channel covariance matrix B
- chol_list: Cholesky decompositions for each eigencomponent
- logdet_sum: Cached log-determinant for likelihood computation
- key_kx, key_B: String keys for cache invalidation
- tau, eta: Cached parameter values

The cache is updated only when kernel parameters, L, eta, or tau_B change.
"""
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

function kernel_key(kcfg::KernelConfig, kp::Dict{Symbol,Float64})
    vals = [string(kp[p]) for p in kcfg.pnames]
    kcfg.name * "::" * join(vals, "|")
end

"""
    build_icm_cache(t, kern_cfg, kp, L, eta, tau_B, cache)

Build or update the ICM cache for efficient likelihood computation.

The cache stores:
1. Eigendecomposition of kernel matrix K_x (only recomputed if kernel params change)
2. Normalized cross-channel covariance B = L*L' (only recomputed if L changes)
3. Cholesky decompositions for each eigencomponent (recomputed if K_x, B, tau, or eta change)

This function uses string keys to detect when parameters change, avoiding
expensive recomputations when parameters are unchanged.
"""
function build_icm_cache(t, kern_cfg::KernelConfig, kp::Dict{Symbol,Float64}, L::Matrix{Float64}, eta::Vector{Float64}, tau_B::Float64, cache::Union{Nothing,ICMCacheState} = nothing)
    cache = cache === nothing ? ICMCacheState() : cache
    # Create keys to detect parameter changes
    key_kx = kernel_key(kern_cfg, kp)
    key_B = join(vec(string.(round.(L; digits = 8))), "|")
    # Check what needs to be recomputed
    need_kx = cache.key_kx === nothing || cache.key_kx != key_kx
    need_B = cache.key_B === nothing || cache.key_B != key_B || cache.Bshape === nothing
    need_tau_eta = cache.tau === nothing || cache.tau != tau_B || cache.eta === nothing || any(cache.eta .!= eta)

    # Recompute kernel eigendecomposition if kernel parameters changed
    if need_kx
        Kx = kern_cfg.fun(t, kp)
        P_exp = nloc(t)
        size(Kx, 1) == P_exp && size(Kx, 2) == P_exp || error("Kernel matrix has wrong size")
        eig = eigen(Symmetric(Kx))
        cache.Ux_x = eig.vectors      # Eigenvectors of K_x
        cache.lam_x = max.(eig.values, 1e-12)  # Eigenvalues (clamped to avoid numerical issues)
        cache.key_kx = key_kx
    end

    # Recompute cross-channel covariance B if L changed
    if need_B
        Bshape = L * transpose(L)  # B = L*L' (cross-channel covariance)
        trB = tr(Bshape)
        # Normalize so trace(B) = M (number of channels)
        if trB > 0
            Bshape .= Bshape .* (size(Bshape, 1) / trB)
        end
        cache.Bshape = Bshape
        cache.key_B = key_B
    end

    # Recompute Cholesky decompositions if anything changed
    # For ICM, the covariance is: Σ_j = τ_B * λ_j * B + diag(η) for each eigencomponent j
    if need_kx || need_B || need_tau_eta || cache.chol_list === nothing
        cache.lam_x === nothing && error("Cache missing lam_x")
        P_need = length(cache.lam_x)
        M = length(eta)
        cache.chol_list = Vector{Cholesky{Float64,Matrix{Float64}}}(undef, P_need)
        logdet_sum = 0.0
        Deta = Diagonal(eta)  # Channel-specific noise variances
        for j in 1:P_need
            # Covariance for eigencomponent j: τ_B * λ_j * B + diag(η)
            Sj = tau_B * cache.lam_x[j] * cache.Bshape + Deta
            ok = false
            jitter = 1e-8  # Add small jitter for numerical stability
            chol = nothing
            # Try Cholesky with increasing jitter if needed
            for _ in 1:6
                try
                    chol = cholesky(Symmetric(Sj + jitter * I); check = false)
                    ok = true
                    break
                catch
                    jitter *= 10  # Increase jitter if Cholesky fails
                end
            end
            ok || error("Cholesky failed for block")
            cache.chol_list[j] = chol
            # Accumulate log-determinant: log|Σ_j| = 2*sum(log(diag(U)))
            logdet_sum += 2 * sum(log, diag(chol.U))
        end
        cache.logdet_sum = logdet_sum
        cache.tau = tau_B
        cache.eta = copy(eta)
    end
    cache
end

"""
    fast_icm_loglik_curve(y_resid, cache)

Compute log-likelihood of residual curve using cached ICM computations.

Uses the eigendecomposition to efficiently compute:
log p(y_resid | parameters) = -0.5 * [P*M*log(2π) + log|Σ| + y'*Σ^(-1)*y]

The computation is done in the eigenbasis where the covariance is block-diagonal,
allowing efficient computation via Cholesky decompositions.
"""
function fast_icm_loglik_curve(y_resid::AbstractMatrix, cache::ICMCacheState)
    cache.Ux_x === nothing && error("Cache not built: missing Ux_x")
    cache.chol_list === nothing && error("Cache not built: missing chol_list")
    P_y = size(y_resid, 1)
    P_ch = length(cache.chol_list)
    P_y == P_ch || error("Cache dimension mismatch")
    # Transform to eigenbasis: Ytil = U_x' * y_resid
    Ytil = transpose(cache.Ux_x) * Matrix{Float64}(y_resid)
    # Compute quadratic form y'*Σ^(-1)*y in eigenbasis (block-diagonal)
    quad = 0.0
    for j in 1:P_ch
        chol = cache.chol_list[j]
        v = Ytil[j, :]  # j-th eigencomponent across channels
        x = chol \ v     # Solve: Σ_j * x = v, so x = Σ_j^(-1) * v
        quad += dot(v, x)  # v' * Σ_j^(-1) * v
    end
    # Log-likelihood: -0.5 * [constant + log|Σ| + quadratic form]
    -0.5 * (P_ch * size(y_resid, 2) * log(2 * pi) + cache.logdet_sum + quad)
end

# =============================================================================
# METROPOLIS-HASTINGS UPDATES
# =============================================================================

"""
Helper function to deep copy a dictionary (needed for MH proposals).
"""
function clone_dict(d::Dict{Symbol,Float64})
    Dict{Symbol,Float64}(k => v for (k, v) in d)
end

"""
    sum_ll_curves(curves, cache)

Sum log-likelihoods across multiple curves using the same cache.
Used in MH updates where we need the total log-likelihood for all curves in a cluster.
"""
function sum_ll_curves(curves::Vector{Matrix{Float64}}, cache::ICMCacheState)
    total = 0.0
    for y in curves
        total += fast_icm_loglik_curve(y, cache)
    end
    total
end

"""
    mh_update_kernel_eig(k, params, kernels, t, Y_list, a_eta, b_eta)

Metropolis-Hastings update for kernel hyperparameters.

For each kernel parameter, proposes a new value and accepts/rejects based on:
- Log-likelihood ratio (computed using cached ICM)
- Prior ratio
- Proposal ratio (with Jacobian for transformed parameters)

Handles different parameter types:
- Bounded parameters (period, kappa, t0): logit transform
- Positive parameters (l_scale, delta, etc.): log-normal proposal
"""
function mh_update_kernel_eig(k::Int, params::Vector{ClusterParams}, kernels::Vector{KernelConfig}, t, Y_list, a_eta::Float64, b_eta::Float64)
    cp = params[k]
    kc = kernels[cp.kern_idx]
    kp = clone_dict(cp.thetas[cp.kern_idx])
    curves = [Matrix{Float64}(Yi - cp.mu_cached) for Yi in Y_list]
    for pn in kc.pnames
        cur = kp[pn]
        if pn == :period
            z_cur = logit_safe(cur)
            z_prp = Base.rand(Normal(z_cur, kc.prop_sd[pn]))
            prp = invlogit_safe(z_prp)
            kp_prop = clone_dict(kp); kp_prop[pn] = prp
            cache_cur = build_icm_cache(t, kc, kp, cp.L, cp.eta, cp.tau_B, cp.cache)
            ll_cur = sum_ll_curves(curves, cache_cur)
            cache_prp = build_icm_cache(t, kc, kp_prop, cp.L, cp.eta, cp.tau_B, ICMCacheState())
            ll_prp = sum_ll_curves(curves, cache_prp)
            lp_cur = kc.prior(kp)
            lp_prp = kc.prior(kp_prop)
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
            z_cur = logit_safe(cur / 2)
            z_prp = Base.rand(Normal(z_cur, kc.prop_sd[pn]))
            prp   = 2 * invlogit_safe(z_prp)

            kp_prop = clone_dict(kp); kp_prop[pn] = prp

            cache_cur = build_icm_cache(t, kc, kp,      cp.L, cp.eta, cp.tau_B, cp.cache)
            ll_cur    = sum_ll_curves(curves, cache_cur)
            cache_prp = build_icm_cache(t, kc, kp_prop, cp.L, cp.eta, cp.tau_B, ICMCacheState())
            ll_prp    = sum_ll_curves(curves, cache_prp)

            lp_cur = kc.prior(kp)
            lp_prp = kc.prior(kp_prop)

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
            tmin, tmax = minimum(t), maximum(t)
            u_cur = (cur - tmin) / (tmax - tmin)
            z_cur = logit_safe(u_cur)
            z_prp = Base.rand(Normal(z_cur, kc.prop_sd[pn]))
            u_prp = invlogit_safe(z_prp)
            prp   = tmin + u_prp * (tmax - tmin)

            kp_prop = clone_dict(kp); kp_prop[pn] = prp

            cache_cur = build_icm_cache(t, kc, kp,      cp.L, cp.eta, cp.tau_B, cp.cache)
            ll_cur    = sum_ll_curves(curves, cache_cur)
            cache_prp = build_icm_cache(t, kc, kp_prop, cp.L, cp.eta, cp.tau_B, ICMCacheState())
            ll_prp    = sum_ll_curves(curves, cache_prp)

            lp_cur = kc.prior(kp)
            lp_prp = kc.prior(kp_prop)

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
            z_cur = log(cur)
            z_prp = Base.rand(Normal(z_cur, kc.prop_sd[pn]))
            prp   = exp(z_prp)

            kp_prop = clone_dict(kp); kp_prop[pn] = prp

            cache_cur = build_icm_cache(t, kc, kp,      cp.L, cp.eta, cp.tau_B, cp.cache)
            ll_cur    = sum_ll_curves(curves, cache_cur)
            cache_prp = build_icm_cache(t, kc, kp_prop, cp.L, cp.eta, cp.tau_B, ICMCacheState())
            ll_prp    = sum_ll_curves(curves, cache_prp)

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
            ll_cur = sum_ll_curves(curves, cache_cur)
            cache_prp = build_icm_cache(t, kc, kp_prop, cp.L, cp.eta, cp.tau_B, ICMCacheState())
            ll_prp = sum_ll_curves(curves, cache_prp)
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

"""
    mh_update_L_eig(k, params, kernels, t, Y_list, mh_step_L)

Metropolis-Hastings update for L matrix (lower-triangular Cholesky factor of B).

L determines the cross-channel covariance structure.
Proposes new L by adding Gaussian noise to packed L parameters.
"""
function mh_update_L_eig(k::Int, params::Vector{ClusterParams}, kernels::Vector{KernelConfig}, t, Y_list, mh_step_L::Float64)
    cp = params[k]
    th = pack_L(cp.L)
    thp = th .+ Base.rand(Normal(0, mh_step_L), length(th))
    Lp = unpack_L(thp, size(cp.L, 1))
    curves = [Matrix{Float64}(Yi - cp.mu_cached) for Yi in Y_list]
    kc = kernels[cp.kern_idx]; kp = cp.thetas[cp.kern_idx]
    cache_cur = build_icm_cache(t, kc, kp, cp.L, cp.eta, cp.tau_B, cp.cache)
    ll_cur = sum_ll_curves(curves, cache_cur)
    cache_prp = build_icm_cache(t, kc, kp, Lp, cp.eta, cp.tau_B, ICMCacheState())
    ll_prp = sum_ll_curves(curves, cache_prp)
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

"""
    mh_update_eta_eig(k, params, kernels, t, Y_list, mh_step_eta, a_eta, b_eta)

Metropolis-Hastings update for eta (channel-specific noise variances).

Updates each channel's noise variance independently using log-normal proposals.
Prior: InverseGamma(a_eta, 1/b_eta)
"""
function mh_update_eta_eig(k::Int, params::Vector{ClusterParams}, kernels::Vector{KernelConfig}, t, Y_list, mh_step_eta::Float64, a_eta::Float64, b_eta::Float64)
    cp = params[k]
    curves = [Matrix{Float64}(Yi - cp.mu_cached) for Yi in Y_list]
    kc = kernels[cp.kern_idx]; kp = cp.thetas[cp.kern_idx]
    for j in eachindex(cp.eta)
        cur = cp.eta[j]
        prp = Base.rand(LogNormal(log(cur), mh_step_eta))
        etap = copy(cp.eta); etap[j] = prp
        cache_cur = build_icm_cache(t, kc, kp, cp.L, cp.eta, cp.tau_B, cp.cache)
        cache_prp = build_icm_cache(t, kc, kp, cp.L, etap, cp.tau_B, ICMCacheState())
        ll_cur = sum_ll_curves(curves, cache_cur)
        ll_prp = sum_ll_curves(curves, cache_prp)
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

"""
    mh_update_tauB_eig(k, params, kernels, t, Y_list, mh_step_tauB)

Metropolis-Hastings update for tau_B (scale parameter for cross-channel covariance).

tau_B controls the overall strength of the cross-channel correlation.
Prior: p(τ_B) ∝ 1/(1+τ_B) (improper prior)
"""
function mh_update_tauB_eig(k::Int, params::Vector{ClusterParams}, kernels::Vector{KernelConfig}, t, Y_list, mh_step_tauB::Float64)
    cp = params[k]
    curves = [Matrix{Float64}(Yi - cp.mu_cached) for Yi in Y_list]
    kc = kernels[cp.kern_idx]; kp = cp.thetas[cp.kern_idx]
    cur = cp.tau_B
    prp = Base.rand(LogNormal(log(cur), mh_step_tauB))
    cache_cur = build_icm_cache(t, kc, kp, cp.L, cp.eta, cur, cp.cache)
    cache_prp = build_icm_cache(t, kc, kp, cp.L, cp.eta, prp, ICMCacheState())
    ll_cur = sum_ll_curves(curves, cache_cur)
    ll_prp = sum_ll_curves(curves, cache_prp)
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

"""
    cc_switch_kernel_eig(k, params, kernels, t, Y_list)

Component-wise MH update for kernel type selection.

Samples a new kernel type from the available kernels based on log-likelihood.
This allows the model to adaptively choose the best kernel for each cluster.
"""
function cc_switch_kernel_eig(k::Int, params::Vector{ClusterParams}, kernels::Vector{KernelConfig}, t, Y_list)
    cp = params[k]
    Mmod = length(kernels)
    p_m = fill(1.0 / Mmod, Mmod)
    theta_draws = [i == cp.kern_idx ? clone_dict(cp.thetas[i]) : kernels[i].pstar() for i in 1:Mmod]
    curves = [Matrix{Float64}(Yi - cp.mu_cached) for Yi in Y_list]
    logw = similar(p_m)
    for m in 1:Mmod
        kc_m = kernels[m]; kp_m = theta_draws[m]
        cache_m = build_icm_cache(t, kc_m, kp_m, cp.L, cp.eta, cp.tau_B, ICMCacheState())
        ll_m = sum_ll_curves(curves, cache_m)
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

# =============================================================================
# MAIN WICMAD FUNCTION
# =============================================================================

"""
    wicmad(Y, t; kwargs...)

Main WICMAD clustering function using ICM (Intrinsic Coregionalization Model).

Performs Bayesian nonparametric clustering of multivariate functional data using:
- Dirichlet Process (DP) prior for number of clusters
- Wavelet-based mean functions with Besov spike-and-slab priors
- Gaussian Process residuals with intrinsic coregionalization
- MCMC sampling for posterior inference

# Arguments
- `Y`: Vector of curves (each element is a P×M matrix or P-length vector)
- `t`: Time points (vector of length P or P×d matrix)

# Key Parameters
- `n_iter`: Total MCMC iterations
- `burn`: Burn-in iterations (discarded)
- `thin`: Thinning interval (keep every thin-th sample)
- `alpha_prior`: DP concentration parameter prior (shape, rate)
- `wf`: Wavelet family (e.g., "sym8", "db4")
- `revealed_idx`: Indices of revealed normal samples (for semi-supervised learning)
- `K_init`: Initial number of clusters
- `mh_step_*`: Proposal step sizes for MH updates

# Returns
Named tuple with:
- `Z`: Matrix of cluster assignments (samples × observations)
- `alpha`: Vector of DP concentration parameter samples
- `kern`: Vector of kernel indices (most frequent per sample)
- `params`: Final cluster parameters
- `K_occ`: Number of occupied clusters per sample
- `loglik`: Log-likelihood values
- `diagnostics`: Diagnostic information (if enabled)
"""
function wicmad(
    Y::Vector,
    t;
    n_iter::Int = 6000,              # Total MCMC iterations
    burn::Int = 3000,                # Burn-in period
    thin::Int = 5,                  # Thinning interval
    alpha_prior::Tuple{Float64,Float64} = (10.0, 1.0),  # DP concentration prior (shape, rate)
    wf::String = "sym8",             # Wavelet family
    J = nothing,                     # Wavelet decomposition level (auto if nothing)
    boundary::String = "periodic",   # Boundary condition: "periodic" or "reflection"
    mh_step_L::Float64 = 0.03,      # MH step size for L matrix
    mh_step_eta::Float64 = 0.10,    # MH step size for eta (noise variances)
    mh_step_tauB::Float64 = 0.15,   # MH step size for tau_B
    revealed_idx::Vector{Int} = Int[],  # Indices of revealed normal samples
    K_init::Int = 5,                # Initial number of clusters
    warmup_iters::Int = 100,        # Warmup iterations where revealed samples are pinned
    unpin::Bool = false,             # Whether to unpin revealed samples after warmup
    kappa_pi::Float64 = 1.0,        # Wavelet prior: kappa_pi parameter
    c2::Float64 = 1.0,              # Wavelet prior: c2 parameter
    tau_pi::Float64 = 40.0,         # Wavelet prior: tau_pi parameter
    a_sig::Float64 = 2.5,           # Wavelet prior: InverseGamma shape for sigma²
    b_sig::Float64 = 0.02,          # Wavelet prior: InverseGamma rate multiplier for sigma²
    a_tau::Float64 = 2.0,           # Wavelet prior: Gamma shape for tau_sigma
    b_tau::Float64 = 2.0,           # Wavelet prior: Gamma rate for tau_sigma
    a_eta::Float64 = 2.0,           # ICM prior: InverseGamma shape for eta
    b_eta::Float64 = 0.1,           # ICM prior: InverseGamma rate for eta
    diagnostics::Bool = true,       # Whether to track diagnostics
    track_ids = nothing,            # (unused, kept for compatibility)
    monitor_levels = nothing,        # (unused, kept for compatibility)
    wf_candidates = nothing,         # Wavelet candidates for automatic selection
    parallel::Bool = false,          # (unused, kept for compatibility)
    rng::AbstractRNG = Random.default_rng(),  # Random number generator
    z_init::Union{Nothing,Vector{Int}} = nothing,  # Initial cluster assignments
    verbose::Bool = true,           # Whether to print progress
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
    Jv = ensure_dyadic_J(P, J)
    t_norm = normalize_t(t, P)
    t_scaled = scale_t01(t_norm)

    # Automatic wavelet selection if revealed_idx is provided and using default wavelet
    # Note: We only auto-select if wf is exactly "sym8" (the default), not if it was explicitly set to "sym8"
    if !isempty(revealed_idx) && wf == "sym8" && wf_candidates === nothing
        verbose && println("\n" * "="^60)
        verbose && println("AUTOMATIC WAVELET SELECTION")
        verbose && println("="^60)
        try
            sel = select_wavelet(Y_mats, t, revealed_idx; 
                               wf_candidates=wf_candidates, 
                               J=Jv, 
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

    kernels = make_kernels()
    alpha = rand(rng, Gamma(alpha_prior[1], 1 / alpha_prior[2]))
    K_from_init = z_init === nothing ? K_init : max(K_init, maximum(relabel_to_consecutive(z_init)))
    v = [rand(rng, Beta(1, alpha)) for _ in 1:K_from_init]
    params = ClusterParams[]
    for _ in 1:length(v)
        cp = draw_new_cluster_params(M, P, t_scaled, kernels; wf = wf, J = Jv, boundary = boundary)
        cp.cache = ICMCacheState()
        cp = ensure_complete_cache!(cp, kernels, t_scaled, M)
        cp.cache = build_icm_cache(t_scaled, kernels[cp.kern_idx], cp.thetas[cp.kern_idx], cp.L, cp.eta, cp.tau_B, cp.cache)
        push!(params, cp)
    end

    if z_init !== nothing
        z = relabel_to_consecutive(Vector{Int}(z_init))
        length(z) == N || error("z_init length must match number of observations")
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
    
    print_interval = max(1, n_iter ÷ 20)
    mode_tag = parallel ? "parallel" : "single-thread"
    verbose && println("Starting WICMAD MCMC with $n_iter iterations ($mode_tag mode)...")

    # =========================================================================
    # MAIN MCMC LOOP
    # =========================================================================
    for iter in 1:n_iter
        # --- Step 1: Slice sampling for cluster assignments ---
        # Convert stick-breaking weights to probabilities
        pi = stick_to_pi(v)
        # Sample slice variables u_i ~ Uniform(0, π_{z_i})
        u = [rand(rng, Uniform(0, pi[z[i]])) for i in 1:N]
        u_star = minimum(u)
        # Extend sticks until tail probability < u_star (slice sampling requirement)
        v = extend_sticks_until(v, alpha, u_star)
        pi = stick_to_pi(v)
        K = length(v)
        # Ensure we have enough cluster parameters for all active clusters
        while length(params) < K
            cp = draw_new_cluster_params(M, P, t_scaled, kernels; wf = wf, J = Jv, boundary = boundary)
            cp.cache = ICMCacheState()
            cp = ensure_complete_cache!(cp, kernels, t_scaled, M)
            cp.cache = build_icm_cache(t_scaled, kernels[cp.kern_idx], cp.thetas[cp.kern_idx], cp.L, cp.eta, cp.tau_B, cp.cache)
            push!(params, cp)
        end
        # Update caches for all clusters (in case parameters changed)
        for k in 1:length(params)
            params[k] = ensure_complete_cache!(params[k], kernels, t_scaled, M)
            params[k].cache = build_icm_cache(t_scaled, kernels[params[k].kern_idx], params[k].thetas[params[k].kern_idx], params[k].L, params[k].eta, params[k].tau_B, params[k].cache)
        end

        # Store previous assignments for ARI computation (diagnostics)

        # --- Step 2: Update cluster assignments (slice sampling) ---
        for i in 1:N
            # Pin revealed normal samples to cluster 1 (if in warmup or unpin=false)
            if !isempty(revealed_idx) && (i in revealed_idx) && (!unpin || (warmup_iters > 0 && iter <= warmup_iters))
                z[i] = 1
                continue
            end
            # Find clusters with π_k > u_i (slice sampling)
            S = findall(x -> x > u[i], pi)
            isempty(S) && (S = [1])
            # Compute log-weights: log(π_k) + log-likelihood
            logw = Vector{Float64}(undef, length(S))
            for (idx_s, k) in enumerate(S)
                ensure_mu_cached!(k, iter)  # Ensure mean function is cached
                ll = ll_curve_k(k, Y_mats[i], params[k].mu_cached)  # Log-likelihood
                logw[idx_s] = log(pi[k]) + ll
            end
            # Numerical stability: subtract max before exponentiating
            logw .-= maximum(logw)
            w = exp.(logw)
            w ./= sum(w)
            # Sample cluster assignment from categorical distribution
            z[i] = S[sample(1:length(S), Weights(w))]
        end

        # --- Step 3: Update stick-breaking weights given assignments ---
        v = update_v_given_z(v, z, alpha)
        pi = stick_to_pi(v)
        K = length(v)

        # --- Step 4: Update cluster-specific parameters ---
        for k in 1:K
            idx = findall(==(k), z)  # Indices of observations in cluster k
            isempty(idx) && continue  # Skip empty clusters
            
            # Update wavelet parameters (mean function) via Gibbs sampling
            # This samples: beta (wavelet coefficients), gamma (spike indicators),
            # pi_level, g_level, sigma2, tau_sigma
            upd = update_cluster_wavelet_params_besov_fullbayes(
                idx, precomp_all, M, params[k].wpar, params[k].sigma2, params[k].tau_sigma
            )
            params[k].wpar      = upd.wpar
            params[k].beta_ch   = upd.beta_ch
            params[k].sigma2    = upd.sigma2_m
            params[k].tau_sigma = upd.tau_sigma

            # Reconstruct mean function from wavelet coefficients
            mu_k = compute_mu_from_beta(params[k].beta_ch, wf, Jv, boundary, P)
            params[k].mu_cached = mu_k
            params[k].mu_cached_iter = iter
            
            # Update ICM parameters (residual covariance) via MH
            Yk = [Y_mats[ii] for ii in idx]  # Curves in cluster k
            params = cc_switch_kernel_eig(k, params, kernels, t_scaled, Yk)  # Kernel type
            params = mh_update_kernel_eig(k, params, kernels, t_scaled, Yk, a_eta, b_eta)  # Kernel hyperparameters
            params = mh_update_L_eig(k, params, kernels, t_scaled, Yk, mh_step_L)  # Cross-channel covariance
            params = mh_update_eta_eig(k, params, kernels, t_scaled, Yk, mh_step_eta, a_eta, b_eta)  # Channel noise
            params = mh_update_tauB_eig(k, params, kernels, t_scaled, Yk, mh_step_tauB)  # Scale parameter
        end

        # --- Step 5: Update DP concentration parameter alpha ---
        Kocc = length(unique(z))  # Number of occupied clusters
        if diagnostics && diag !== nothing
            diag[:global][:K_occ_all][iter] = Kocc
        end

        # Update alpha using auxiliary variable method (Escobar & West 1995)
        eta_aux = rand(rng, Beta(alpha + 1, N))
        mix = (alpha_prior[1] + Kocc - 1) / (N * (alpha_prior[2] - log(eta_aux)) + alpha_prior[1] + Kocc - 1)
        if rand(rng) < mix
            alpha = rand(rng, Gamma(alpha_prior[1] + Kocc, 1 / (alpha_prior[2] - log(eta_aux))))
        else
            alpha = rand(rng, Gamma(alpha_prior[1] + Kocc - 1, 1 / (alpha_prior[2] - log(eta_aux))))
        end

        if iter % print_interval == 0 || iter == n_iter
            percentage = round(100 * iter / n_iter, digits=1)
            verbose && println("$iter/$n_iter ($percentage%) - Clusters: $Kocc")
        end

        # --- Step 6: Store samples (after burn-in, with thinning) ---
        if keep > 0 && iter > burn && ((iter - burn) % thin == 0)
            sidx += 1
            Z_s[sidx, :] = z  # Cluster assignments
            alpha_s[sidx] = alpha  # DP concentration parameter
            # Track kernel of largest cluster
            counts = countmap(z)
            sorted_keys = sort(collect(keys(counts)); by = k -> -counts[k])
            k_big = sorted_keys[1]
            kern_s[sidx] = params[k_big].kern_idx
            K_s[sidx] = Kocc  # Number of occupied clusters
            # Compute total log-likelihood
            totll = 0.0
            for i in 1:N
                ki = z[i]
                ensure_mu_cached!(ki, iter)
                totll += ll_curve_k(ki, Y_mats[i], params[ki].mu_cached)
            end
            loglik_s[sidx] = totll
            # Store diagnostics
            if diagnostics && diag !== nothing
                diag[:global][:K_occ][sidx] = Kocc
                diag[:global][:alpha][sidx] = alpha
                diag[:global][:loglik][sidx] = totll
            end
        end
    end
    
    final_Kocc = length(unique(z))
    if verbose
        println("\nMCMC completed! Final clusters: $final_Kocc, Samples collected: $sidx")
    end

    (; Z = Z_s, alpha = alpha_s, kern = kern_s, params = params, v = v, pi = stick_to_pi(v),
        revealed_idx = revealed_idx, K_occ = K_s, loglik = loglik_s, diagnostics = diag)
end

function relabel_to_consecutive(z::Vector{Int})
    unique_labels = unique(z)
    label_map = Dict(old_label => new_label for (new_label, old_label) in enumerate(unique_labels))
    return [label_map[label] for label in z]
end

# =============================================================================
# POSTPROCESSING (for MCMC output) - moved to PostProcessing submodule below
# =============================================================================

# =============================================================================
# KERNEL SELECTION ALGORITHM
# =============================================================================

"""
    wavelet_smooth_mean_function(mean_function, wf, J, boundary)

Apply MCMC-based wavelet smoothing to the mean function using Besov spike-and-slab priors.
Runs a separate MCMC algorithm with iter=3000, burnin=1000, thin=1.
"""
function wavelet_smooth_mean_function(Y_mats::Vector{Matrix{Float64}}, revealed_idx::Vector{Int}, wf::String, J::Int, boundary::String;
                                      return_beta::Bool=false, n_iter_selection::Int=3000, burnin_selection::Int=1000, thin_selection::Int=1, verbose::Bool=false)
    P, M = size(Y_mats[1])
    cluster_mean = zeros(P, M)
    
    for idx in revealed_idx
        cluster_mean .+= Y_mats[idx]
    end
    cluster_mean ./= length(revealed_idx)
    
    Y_list = [cluster_mean]
    
    test_wt = wt_forward_1d(cluster_mean[:, 1]; wf=wf, J=J, boundary=boundary)
    lev_names = [String(k) for k in keys(test_wt.map.idx)]
    det_names = filter(name -> startswith(name, "d"), lev_names)
    ncoeff = length(test_wt.coeff)
    
    pi_level = Dict(name => 0.5 for name in det_names)
    g_level = Dict(name => 2.0 for name in det_names)
    gamma_ch = [Int.(Base.rand(Bernoulli(0.2), ncoeff)) for _ in 1:M]
    
    kappa_pi = 1.0
    c2 = 1.0
    tau_pi = 40.0
    a_g = 2.0
    b_g = 2.0
    a_sig = 2.5
    b_sig = 0.02
    a_tau = 2.0
    b_tau = 2.0
    
    wpar = WaveletParams(lev_names, pi_level, g_level, gamma_ch,
                         kappa_pi, c2, tau_pi,
                         a_g, b_g,
                         a_sig, b_sig, a_tau, b_tau)
    
    sigma2_m = [1.0 for _ in 1:M]
    tau_sigma = 1.0
    
    precomp = precompute_wavelets(Y_list, wf, J, boundary)
    
    verbose && println("    Running MCMC for wavelet $wf (iter=$n_iter_selection, burnin=$burnin_selection)...")
    
    keep = max(0, Int(floor((n_iter_selection - burnin_selection) / thin_selection)))
    beta_samples = []
    
    for iter in 1:n_iter_selection
        upd = update_cluster_wavelet_params_besov(
            [1], precomp, M, wpar, sigma2_m, tau_sigma;
            kappa_pi = 1.0, c2 = 1.0, tau_pi = 40.0,
            g_hyp = nothing,
            a_sig = 2.5, b_sig = 0.02,
            a_tau = 2.0, b_tau = 2.0
        )
        
        wpar = upd.wpar
        sigma2_m = upd.sigma2_m
        tau_sigma = upd.tau_sigma
        
        if iter > burnin_selection && ((iter - burnin_selection) % thin_selection == 0)
            push!(beta_samples, copy(upd.beta_ch))
        end
    end
    
    if isempty(beta_samples)
        verbose && println("    Warning: No samples collected for wavelet $wf")
        return cluster_mean
    end
    
    n_samples = length(beta_samples)
    posterior_mean_beta = [zeros(length(beta_samples[1][m])) for m in 1:M]
    
    for m in 1:M
        for sample_idx in 1:n_samples
            posterior_mean_beta[m] .+= beta_samples[sample_idx][m]
        end
        posterior_mean_beta[m] ./= n_samples
    end
    
    smoothed = compute_mu_from_beta(posterior_mean_beta, wf, J, boundary, P)
    
    verbose && println("    MCMC completed for wavelet $wf ($n_samples samples)")
    
    if return_beta
        return (smoothed_mean = smoothed, beta_summaries = (beta_mean = posterior_mean_beta, gamma_last = wpar.gamma_ch))
    else
        return smoothed
    end
end

"""
    score_wavelet_candidate(Y_mats::Vector{Matrix{Float64}}, revealed_idx::Vector{Int}, t::AbstractVector,
                           wf::String, J, boundary::String; mcmc = (n_iter = 3000, burnin = 1000, thin = 1))

Score a wavelet candidate using MCMC-based metrics.
"""
function score_wavelet_candidate(Y_mats::Vector{Matrix{Float64}}, revealed_idx::Vector{Int}, t::AbstractVector,
                                 wf::String, J, boundary::String;
                                 mcmc = (n_iter = 3000, burnin = 1000, thin = 1),
                                 verbose::Bool = true)

    P, M = size(Y_mats[1])

    mean_function = zeros(P, M); cnt = 0
    for i in revealed_idx
        if 1 <= i <= length(Y_mats)
            mean_function .+= Y_mats[i]; cnt += 1
        end
    end
    cnt == 0 && error("No indices in revealed_idx")
    mean_function ./= cnt

    Jv = isnothing(J) ? ensure_dyadic_J(P, nothing) : J
    
    smoothed_pack = wavelet_smooth_mean_function(Y_mats, revealed_idx, wf, Jv, boundary;
                                                 n_iter_selection = mcmc.n_iter,
                                                 burnin_selection = mcmc.burnin,
                                                 thin_selection   = mcmc.thin,
                                                 return_beta      = true,
                                                 verbose          = verbose)
    smoothed = smoothed_pack.smoothed_mean
    βsumm    = smoothed_pack.beta_summaries
    β̄_post  = βsumm.beta_mean
    γ_last   = βsumm.gamma_last

    mse_time = mean((mean_function .- smoothed).^2)

    precomp = precompute_wavelets(Y_mats, wf, Jv, boundary)
    stk     = stack_D_from_precomp(precomp, revealed_idx, M)
    D       = stk.D_arr
    Dbar    = [vec(mean(view(D, :, :, m); dims = 2)) for m in 1:M]
    @assert all(length(β̄_post[m]) == length(Dbar[m]) for m in 1:M)
    mse_coeff = mean(vcat([(@. (β̄_post[m] - Dbar[m])^2) for m=1:M]...))

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
                      mcmc = (n_iter = 3000, burnin = 1000, thin = 1),
                      verbose::Bool = true)

    Y_mats = Vector{Matrix{Float64}}(undef, length(Y))
    for i in 1:length(Y)
        Y_mats[i] = isa(Y[i], AbstractVector) ? reshape(Float64.(Y[i]), :, 1) : Matrix{Float64}(Y[i])
    end

    if wf_candidates === nothing
        wf_candidates = ["haar", "db2", "db4", "db6", "db8",
                         "coif2", "coif4", "coif6",
                         "sym4", "sym6", "sym8",
                         "batt2", "batt4", "batt6"]
    end

    results = NamedTuple[]
    best = (wf = "", score = Inf)

    verbose && println("Testing $(length(wf_candidates)) wavelet candidates with MCMC scoring...")
    verbose && println("MCMC parameters: $(mcmc.n_iter) iterations, $(mcmc.burnin) burn-in, thin=$(mcmc.thin)")

    for wf in wf_candidates
        try
            verbose && println("  Testing wavelet: $wf")
            met = score_wavelet_candidate(Y_mats, revealed_idx, t, wf, J, boundary; mcmc = mcmc, verbose=verbose)
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
            
            verbose && println("    MSE_time: $(round(met.mse_time, digits=6)), MSE_coeff: $(round(met.mse_coeff, digits=6)), Sparsity: $(round(met.sparsity, digits=3)), Score: $(round(score, digits=6))")
        catch e
            @warn "Wavelet $wf failed during selection: $e"
        end
    end

    if isempty(results)
        verbose && @warn "All wavelet candidates failed during selection, defaulting to 'sym8'"
        return (selected_wf = "sym8", table = NamedTuple[])
    end

    results_sorted = sort(results; by = r -> r.score)
    best_result = results_sorted[1]
    
    verbose && println("\nWavelet Selection Results:")
    verbose && println("wf        MSE_time    MSE_coeff   Sparsity   Score")
    verbose && println("-" ^ 70)
    if verbose
        for r in results_sorted
            println("$(lpad(r.wf, 8)) $(lpad(round(r.mse_time, digits=6), 10)) $(lpad(round(r.mse_coeff, digits=6), 10)) $(lpad(round(r.sparsity, digits=3), 9)) $(lpad(round(r.score, digits=6), 10))")
        end
    end
    verbose && println("Selected wavelet: $(best_result.wf) (score: $(round(best_result.score, digits=6)))")
    
    return (selected_wf = best_result.wf, table = results_sorted)
end

# =============================================================================
# POSTPROCESSING SUBMODULE
# =============================================================================

module PostProcessing

using StatsBase

export dahl_partition, dahl_from_res, map_partition, map_from_res

function canonical_labels(z::AbstractVector{<:Integer})
    uniq = unique(z)
    mapping = Dict(u => i for (i, u) in enumerate(uniq))
    [mapping[val] for val in z]
end

function dahl_partition(Z::AbstractMatrix{<:Integer})
    S, N = size(Z)
    PSM = zeros(Float64, N, N)
    for s in 1:S
        zs = Z[s, :]
        A = zs .== permutedims(zs)
        PSM .+= Float64.(A)
    end
    PSM ./= S
    score = zeros(Float64, S)
    for s in 1:S
        zs = Z[s, :]
        A = zs .== permutedims(zs)
        score[s] = sum((Float64.(A) .- PSM) .^ 2)
    end
    s_hat = argmin(score)
    z_hat = canonical_labels(vec(Z[s_hat, :]))
    K_hat = length(unique(z_hat))
    (; z_hat, K_hat, s_hat, PSM, score)
end

function dahl_from_res(res::NamedTuple)
    haskey(res, :Z) || error("res[:Z] not found")
    dahl_partition(res.Z)
end

function map_partition(Z::AbstractMatrix{<:Integer})
    S, N = size(Z)
    keys = Vector{String}(undef, S)
    for s in 1:S
        keys[s] = join(canonical_labels(vec(Z[s, :])), "-")
    end
    tab = countmap(keys)
    key_hat = first(keys)
    freq = tab[key_hat]
    for (k, v) in tab
        if v > freq
            key_hat = k
            freq = v
        end
    end
    s_hat = findfirst(==(key_hat), keys)
    z_hat = canonical_labels(vec(Z[s_hat, :]))
    K_hat = length(unique(z_hat))
    (; z_hat, K_hat, s_hat, key_hat, freq)
end

function map_from_res(res::NamedTuple)
    haskey(res, :Z) || error("res[:Z] not found")
    map_partition(res.Z)
end

end # PostProcessing

# =============================================================================
# UTILS SUBMODULE
# =============================================================================

module Utils

export load_ucr_dataset, LoadedDataset

struct LoadedDataset
    series::Vector{Matrix{Float64}}
    labels::Vector{String}
end

function read_ts_file(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && error("$(path) appears to be empty")
    idx = findfirst(l -> strip(lowercase(strip(l))) == "@data", lines)
    if idx === nothing
        idx = findfirst(l -> startswith(lowercase(strip(l)), "@data"), lines)
    end
    idx === nothing && error("@data section not found in $(path)")
    data_lines = [strip(l) for l in lines[idx+1:end] if !isempty(strip(l)) && !startswith(lowercase(strip(l)), "@")]
    series = Vector{Matrix{Float64}}()
    labels = String[]
    for (ln, line) in enumerate(data_lines)
        parts = split(line, ":")
        length(parts) >= 2 || error("Invalid line $(ln) in $(path): expected at least one dimension and a label")
        dims = parts[1:end-1]
        label = strip(parts[end])
        dim_vals = []
        for dim in dims
            vals = split(strip(dim), ",")
            parsed_vals = Float64[]
            for val in vals
                val = strip(val)
                if isempty(val)
                    continue
                elseif val == "?" || lowercase(val) == "nan"
                    push!(parsed_vals, NaN)
                else
                    push!(parsed_vals, parse(Float64, val))
                end
            end
            push!(dim_vals, parsed_vals)
        end
        lengths = length.(dim_vals)
        all(lengths .== lengths[1]) || error("Inconsistent dimension lengths in line $(ln) of $(path)")
        n_time = lengths[1]
        n_dim = length(dim_vals)
        mat = Array{Float64}(undef, n_time, n_dim)
        for j in 1:n_dim
            mat[:, j] = dim_vals[j]
        end
        push!(series, mat)
        push!(labels, label)
    end
    LoadedDataset(series, labels)
end

function load_ucr_dataset(dataset::AbstractString, data_dir::AbstractString)
    base = joinpath(data_dir, dataset)
    train_path = base * "_TRAIN.ts"
    test_path = base * "_TEST.ts"
    isfile(train_path) || error("Training file not found: " * train_path)
    train = read_ts_file(train_path)
    test = isfile(test_path) ? read_ts_file(test_path) : LoadedDataset(Matrix{Float64}[], String[])
    LoadedDataset(vcat(train.series, test.series), vcat(train.labels, test.labels))
end

end # Utils

# Re-export submodules
using .PostProcessing
using .Utils

# Export main function and utilities
export wicmad, init_diagnostics,
       dahl_from_res, map_from_res, dahl_partition, map_partition,
       select_wavelet, score_wavelet_candidate, wavelet_smooth_mean_function

end # module

