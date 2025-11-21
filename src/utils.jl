module Utils

using LinearAlgebra
using StatsBase
using StatsFuns: logit, logistic
using Distances
using Distributions
using Random
using Statistics: mean
using Wavelets
using Wavelets: WT

# Re-export from ICMCache for convenience
# Note: ICMCache will be available after this module is loaded

export scale_t01, dist_rows, nloc, ensure_dyadic_J, logit_safe, invlogit_safe,
       stick_to_pi, extend_sticks_until, update_v_given_z, pack_L, unpack_L,
       as_num_mat, normalize_t, eig_Kx, eig_Bshape, project_curve,
       # Dataset loading exports
       load_ucr_dataset, prepare_anomaly_dataset, summarize_dataset, default_time_index,
       LoadedDataset, PreparedDataset,
       # Initialization exports
       AcceptCount, AcceptanceTracker, WaveletParams, ClusterParams,
       draw_empty_acc, draw_new_cluster_params, ensure_complete_cache!,
       # Kernel exports
       KernelConfig, make_kernels,
       # Process config for residual model
       ProcessConfig

# =============================================================================
# KERNEL FUNCTIONS (from kernels.jl) - MOVED TO TOP TO AVOID FORWARD REFERENCE ISSUES
# =============================================================================

struct KernelConfig
    name::String
    fun::Function
    pnames::Vector{Symbol}
    prior::Function
    pstar::Function
    prop_sd::Dict{Symbol,Float64}
end

# =============================================================================
# PROCESS CONFIGURATION
# =============================================================================

struct ProcessConfig
    process::Symbol   # :gp or :tprocess
    nu_df::Float64    # degrees of freedom for t-process
    learn_nu::Bool    # toggle MH update for ν
end

function k_sqexp(t, l_scale)
    D2 = dist_rows(t).^2
    @. exp(-0.5 * D2 / (l_scale^2))
end

function k_mat32(t, l_scale)
    D = dist_rows(t)
    r = D ./ l_scale
    a = sqrt(3.0) .* r
    @. (1 + a) * exp(-a)
end

function k_mat52(t, l_scale)
    D = dist_rows(t)
    r = D ./ l_scale
    a = sqrt(5.0) .* r
    @. (1 + a + 5 * r^2 / 3) * exp(-a)
end

function k_periodic(t, l_scale, period)
    D = dist_rows(t)
    @. exp(-2 * sinpi(D / period)^2 / (l_scale^2))
end

# Rational Quadratic (unit variance)
# k_RQ(r) = (1 + r^2 / (2 α ℓ^2))^{-α}
function k_rq(t, l_scale, alpha)
    D2 = dist_rows(t).^2
    @. (1 + D2 / (2 * alpha * l_scale^2))^(-alpha)
end

# Powered Exponential (a.k.a. generalized exponential, unit variance)
# k_PE(r) = exp(-(r/ℓ)^κ), with κ ∈ (0, 2]
function k_powexp(t, l_scale, kappa)
    D = dist_rows(t)
    r = D ./ l_scale
    @. exp(-(r^kappa))
end

# =============================================================================
# NON-STATIONARY KERNELS
# =============================================================================

# Piecewise-constant 3-bin mapping for ell(t)
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

# Gibbs kernel with piecewise-constant ℓ(t) in 3 bins (unit variance)
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

# Matérn-5/2 correlation (unit variance)
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
            # left/right regime correlations
            kL = _m52_corr(r, ℓ_left)
            kR = _m52_corr(r, ℓ_right)
            val = (1-σi)*(1-σj)*kL + σi*σj*kR
            K[i,j] = val
            K[j,i] = val
        end
    end
    return K
end

function make_kernels(; add_bias_variants::Bool = false)
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
            # Prior: l_scale ~ Gamma(2, 1/2); kappa ∈ (0,2] via Beta on kappa/2 with Jacobian
            par -> (logpdf(Gamma(2, 1 / 2), par[:l_scale]) +
                    (logpdf(Beta(2, 2), par[:kappa] / 2) - log(2.0))),
            () -> Dict(:l_scale => Base.rand(Gamma(2, 1 / 2)),
                        :kappa   => 2 * Base.rand(Beta(2, 2))),
            Dict(:l_scale => 0.20, :kappa => 0.25),
        ),
        # Gibbs with piecewise-constant 3-bin lengthscale
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
        # Change-point Matérn-5/2
        KernelConfig(
            "CP_M52",
            (t, par) -> k_changepoint_m52(t, par[:t0], par[:delta], par[:ell_left], par[:ell_right]),
            [:t0, :delta, :ell_left, :ell_right],
            par -> (
                # t0 prior uniform over domain via logistic transform below; here we assume flat in [tmin,tmax]
                # delta prior prefers moderate transitions
                logpdf(Gamma(2, 1/2), par[:ell_left])  +
                logpdf(Gamma(2, 1/2), par[:ell_right]) +
                logpdf(Gamma(2, 1.0),  par[:delta])    # shape=2 scale=1 encourages δ ~ O(1)
            ),
            () -> Dict(:t0 => 0.5,                 # will be moved by MH
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

function ensure_dyadic_J(P::Integer, J::Union{Nothing, Integer, AbstractFloat})
    Jval = isnothing(J) ? log2(P) : Float64(J)
    Jint = round(Int, Jval)
    
    # Check if P is close to 2^Jint (within tolerance)
    expected_P = 2^Jint
    if abs(P - expected_P) > eps(Float64) * max(1, P)
        # If not dyadic, find the closest dyadic J
        Jint = floor(Int, log2(P))
        expected_P = 2^Jint
        if abs(P - expected_P) > eps(Float64) * max(1, P)
            # Still not dyadic, use the closest J
            Jint = round(Int, log2(P))
        end
    end
    Jint
end

logit_safe(x) = logit(clamp(x, eps(Float64), 1 - eps(Float64)))
invlogit_safe(z) = logistic(z)

function stick_to_pi(v::AbstractVector)
    K = length(v)
    pi = zeros(Float64, K)
    tail = 1.0
    for k in 1:K
        pi[k] = v[k] * tail
        tail *= (1 - v[k])
    end
    pi
end

function extend_sticks_until(v::Vector{Float64}, alpha::Float64, threshold::Float64)
    tail = prod(1 .- v)
    while tail > threshold
        v_new = Base.rand(Beta(1, alpha))
        push!(v, v_new)
        tail *= (1 - v_new)
    end
    v
end

function update_v_given_z(v::Vector{Float64}, z::Vector{Int}, alpha::Float64)
    K = length(v)
    counts = [count(==(k), z) for k in 1:K]
    tail_counts = reverse(cumsum(reverse(counts)))
    for k in 1:K
        a = 1 + counts[k]
        b = alpha + (k < K ? tail_counts[k + 1] : 0)
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

function project_curve(Yi::AbstractMatrix, mu::AbstractMatrix, V::AbstractMatrix)
    V === nothing && error("Eigenvectors V are nothing")
    transpose(V) * (Matrix{Float64}(Yi) - Matrix{Float64}(mu))
end

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
# DATASET LOADING FUNCTIONS (from common.jl)
# =============================================================================

struct LoadedDataset
    series::Vector{Matrix{Float64}}
    labels::Vector{String}
end

struct PreparedDataset
    series::Vector{Matrix{Float64}}
    binary_labels::Vector{Int}
    original_labels::Vector{String}
    normal_label::String
    anomaly_labels::Vector{String}
    index_map::Dict{String,Int}
end

function read_ts_file(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && error("$(path) appears to be empty")
    # Look for @data on its own line (not @data@problemName)
    idx = findfirst(l -> strip(lowercase(strip(l))) == "@data", lines)
    if idx === nothing
        # Fallback: look for any line starting with @data
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
        # Parse dimensions, handling "?" as missing values (NaN)
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

function majority_label_id(counts::Dict{Int,Int})
    max_lab = first(keys(counts))
    max_count = counts[max_lab]
    for (lab, cnt) in counts
        if cnt > max_count
            max_count = cnt
            max_lab = lab
        end
    end
    max_lab
end

function prepare_anomaly_dataset(data::LoadedDataset; anomaly_ratio::Float64 = 0.15, rng::AbstractRNG = Random.default_rng())
    length(data.series) == length(data.labels) || error("Series and labels length mismatch")
    isempty(data.series) && error("Dataset contains no series")
    label_map = Dict{String,Int}()
    mapped = Vector{Int}(undef, length(data.labels))
    next_id = 1
    for (i, lab) in enumerate(data.labels)
        lab = strip(lab)
        if !haskey(label_map, lab)
            label_map[lab] = next_id
            next_id += 1
        end
        mapped[i] = label_map[lab]
    end
    counts = countmap(mapped)
    
    # Sort clusters by size to find the largest (most normal) and other clusters
    sorted_clusters = sort(collect(counts), by=x->x[2], rev=true)
    
    if length(sorted_clusters) < 2
        @warn "Dataset has only one class; cannot create anomalies"
        # Return all data as normal
        binary = zeros(Int, length(mapped))
        normal_idx = collect(1:length(mapped))
        anomaly_idx = Int[]
    else
        # Use the largest cluster as normal (most normal observations)
        maj_id = sorted_clusters[1][1]
        maj_label = ""
        for (lab, idx) in label_map
            if idx == maj_id
                maj_label = lab
                break
            end
        end
        
        # Use the OTHER anomaly cluster (not the second largest, but a different one)
        # Find the smallest non-majority cluster to use as anomalies
        other_clusters = sorted_clusters[2:end]  # All clusters except the largest
        if length(other_clusters) > 1
            # Use the smallest cluster (last in sorted list) as anomalies
            anomaly_id = other_clusters[end][1]
        else
            # Only one other cluster available, use it
            anomaly_id = other_clusters[1][1]
        end
        
        anomaly_label = ""
        for (lab, idx) in label_map
            if idx == anomaly_id
                anomaly_label = lab
                break
            end
        end
        
        binary = [mapped[i] == maj_id ? 0 : 1 for i in eachindex(mapped)]
        normal_idx = findall(==(0), binary)
        anomaly_idx = findall(==(1), binary)
        
        # Calculate how many anomalies we need based on the ratio
        n_normal = length(normal_idx)
        n_anom_available = length(anomaly_idx)
        n_anom_needed = round(Int, anomaly_ratio * n_normal / (1 - anomaly_ratio))
        n_anom = min(n_anom_needed, n_anom_available)
        
        if n_anom > 0
            # Shuffle and select anomalies from the same group
            shuffled_anoms = Random.shuffle(rng, anomaly_idx)
            anomaly_idx = shuffled_anoms[1:n_anom]
        else
            anomaly_idx = Int[]
        end
    end
    
    # Combine normal and selected anomalies
    selected_idx = vcat(normal_idx, anomaly_idx)
    shuffled = Random.shuffle(rng, selected_idx)
    sel_series = [data.series[i] for i in shuffled]
    sel_labels = [binary[i] for i in shuffled]
    anomaly_labels = String[]
    if length(sorted_clusters) >= 2
        # Only include the single anomaly group we selected
        other_clusters = sorted_clusters[2:end]
        if length(other_clusters) > 1
            anomaly_id = other_clusters[end][1]  # Smallest cluster
        else
            anomaly_id = other_clusters[1][1]     # Only other cluster
        end
        for (lab, idx) in label_map
            if idx == anomaly_id
                push!(anomaly_labels, lab)
                break
            end
        end
    end
    
    # Get the normal label
    maj_label = ""
    if length(sorted_clusters) >= 1
        maj_id = sorted_clusters[1][1]
        for (lab, idx) in label_map
            if idx == maj_id
                maj_label = lab
                break
            end
        end
    end
    
    PreparedDataset(sel_series, sel_labels, data.labels, maj_label, anomaly_labels, label_map)
end

function summarize_dataset(prepped::PreparedDataset)
    n = length(prepped.series)
    n == 0 && return "(empty dataset)"
    n_time = size(prepped.series[1], 1)
    n_dim = size(prepped.series[1], 2)
    anomaly_pct = mean(prepped.binary_labels) * 100
    "Curves: $(n) | Length: $(n_time) | Dimensions: $(n_dim) | Anomaly %: $(round(anomaly_pct; digits=2))"
end

function default_time_index(series::Vector{Matrix{Float64}})
    isempty(series) && error("Cannot build time index for empty dataset")
    n_time = size(series[1], 1)
    collect(1:n_time)
end

# =============================================================================
# INITIALIZATION STRUCTURES AND FUNCTIONS (from init.jl)
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

mutable struct WaveletParams
    lev_names::Vector{String}
    pi_level::Dict{String,Float64}     # to be SAMPLED (current value per detail level)
    g_level::Dict{String,Float64}      # to be SAMPLED (current value per detail level)
    gamma_ch::Vector{Vector{Int}}      # per-channel indicators (0/1) for every coefficient

    # NEW hyperparameters (per cluster, match write-up)
    kappa_pi::Float64                  # κ_π
    c2::Float64                        # c_2
    tau_pi::Float64                    # τ_π

    a_g::Float64                       # IG prior on g: shape
    b_g::Float64                       # IG prior on g: rate

    a_sig::Float64                     # IG prior on σ²: shape
    b_sig::Float64                     # IG prior on σ²: rate multiplier (used as b_sig * tau_sigma)
    a_tau::Float64                     # Gamma prior on τ_σ: shape
    b_tau::Float64                     # Gamma prior on τ_σ: rate
end

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
    cache::Any  # Will be ICMCacheState when ICMCache is loaded
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

function draw_new_cluster_params(M::Int, P::Int, t, kernels::Vector{KernelConfig}; wf::String = "sym8", J = nothing, boundary::String = "periodic")
    Jv = ensure_dyadic_J(P, J)
    zeros_mat = zeros(P, M)
    tmp = wt_forward_mat(zeros_mat; wf = wf, J = Jv, boundary = boundary)
    lev_names = [String(k) for k in keys(tmp[1].map.idx)]
    det_names = filter(name -> startswith(name, "d"), lev_names)
    ncoeff = length(tmp[1].coeff)

    # Prior hyperparameters: match thesis defaults (use 2's; tau_pi moderately informative)
    kappa_pi = 1.0
    c2       = 1.0
    tau_pi   = 40.0
    a_g, b_g = 2.0, 2.0
    a_sig, b_sig = 2.0, 2.0
    a_tau, b_tau = 2.0, 2.0

    # Initialize per-level π and g at reasonable prior locations
    pi_level = Dict{String,Float64}()
    g_level  = Dict{String,Float64}()
    for name in det_names
        j = parse(Int, replace(name, "d"=>""))
        m_j = kappa_pi * 2.0^(-c2 * j)
        pi_level[name] = clamp(m_j, 1e-6, 1 - 1e-6)
        # IG mode for g when a_g > 1: mode = rate / (shape + 1) in variance-IG; here we store expectation of g as positive value.
        # We will sample g later; any positive seed is fine.
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
        nothing,  # Will be set to ICMCacheState() when ICMCache is loaded
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
    # Note: cache will be built later when ICMCache module is available
    cp
end

# =============================================================================
# WAVELET OPERATIONS (needed for initialization)
# =============================================================================

struct WaveletMap
    J::Int
    wf::String
    boundary::String
    P::Int
    idx::Dict{Symbol,UnitRange{Int}}
end

struct WaveletCoefficients
    coeff::Vector{Float64}
    map::WaveletMap
end

const _WAVELET_ALIASES = Dict(
    # Haar wavelets
    "haar" => :haar,
    
    # Daubechies wavelets (db1:Inf)
    "db1" => :haar,    # db1 is equivalent to haar
    "db2" => :db2,
    "db3" => :db3,
    "db4" => :db4,
    "db5" => :db5,
    "db6" => :db6,
    "db7" => :db7,
    "db8" => :db8,
    "db9" => :db9,
    "db10" => :db10,
    "db11" => :db11,
    "db12" => :db12,
    "db13" => :db13,
    "db14" => :db14,
    "db15" => :db15,
    "db16" => :db16,
    "db17" => :db17,
    "db18" => :db18,
    "db19" => :db19,
    "db20" => :db20,
    
    # Coiflet wavelets (coif2:2:8)
    "coif2" => :coif2,
    "coif4" => :coif4,
    "coif6" => :coif6,
    "coif8" => :coif8,
    
    # Symlet wavelets (sym4:10)
    "sym4" => :sym4,
    "sym5" => :sym5,
    "sym6" => :sym6,
    "sym7" => :sym7,
    "sym8" => :sym8,
    "sym9" => :sym9,
    "sym10" => :sym10,
    
    # Battle wavelets (batt2:2:6)
    "batt2" => :batt2,
    "batt4" => :batt4,
    "batt6" => :batt6,
    
    # Beylkin wavelet
    "beyl" => :beyl,
    
    # Vaidyanathan wavelet
    "vaid" => :vaid,
)

function wt_forward_1d(y::AbstractVector; wf::String = "sym8", J::Int = nothing, boundary::String = "periodic")
    P = length(y)
    Jv = ensure_dyadic_J(P, J)
    wf_sym = get(_WAVELET_ALIASES, wf, Symbol(wf))
    if isdefined(WT, wf_sym)
        ext = boundary == "periodic" ? WT.Periodic : WT.Symmetric
        wave = wavelet(getfield(WT, wf_sym), WT.Filter, ext)
        wt = dwt(Float64.(y), wave, Jv)
        
        # Organize coefficients into detail and approximation coefficients
        coeffs_vec = Float64[]
        idx = Dict{Symbol,UnitRange{Int}}()
        offset = 0
        
        # Extract detail coefficients for each level
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
        
        # Extract approximation coefficients
        approx_start = detailindex(P, Jv, Jv)
        s = wt[approx_start:end]
        append!(coeffs_vec, s)
        idx[Symbol("s" * string(Jv))] = offset + 1:offset + length(s)
        
        WaveletCoefficients(coeffs_vec, WaveletMap(Jv, wf, boundary, P, idx))
    else
        error("Unsupported wavelet family $wf")
    end
end

function wt_inverse_1d(coeff_vec::AbstractVector, map::WaveletMap)
    wf_sym = get(_WAVELET_ALIASES, map.wf, Symbol(map.wf))
    if isdefined(WT, wf_sym)
        ext = map.boundary == "periodic" ? WT.Periodic : WT.Symmetric
        wave = wavelet(getfield(WT, wf_sym), WT.Filter, ext)
        
        # Reconstruct the flat coefficient vector in the format expected by idwt
        wt_reconstructed = zeros(Float64, map.P)
        
        # Reconstruct detail coefficients
        for lev in 1:map.J
            key = Symbol("d" * string(lev))
            if haskey(map.idx, key)
                ids = map.idx[key]
                detail_coeffs = collect(coeff_vec[ids])
                
                # Place detail coefficients in the correct positions
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
        
        # Reconstruct approximation coefficients
        approx_key = Symbol("s" * string(map.J))
        if haskey(map.idx, approx_key)
            ids = map.idx[approx_key]
            approx_coeffs = collect(coeff_vec[ids])
            approx_start = detailindex(map.P, map.J, map.J)
            if length(approx_coeffs) > 0
                wt_reconstructed[approx_start:end] = approx_coeffs
            end
        end
        
        # Perform inverse transform
        idwt(wt_reconstructed, wave, map.J)
    else
        error("Unsupported wavelet family $(map.wf)")
    end
end

function wt_forward_mat(Y::AbstractMatrix; wf::String = "sym8", J::Int = nothing, boundary::String = "periodic")
    [wt_forward_1d(Y[:, m]; wf = wf, J = J, boundary = boundary) for m in axes(Y, 2)]
end

end # module