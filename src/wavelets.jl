module WaveletOps

using ..Utils
using Wavelets
using Wavelets: WT
using LinearAlgebra
using Distributions
using StatsFuns: logistic

export WaveletMap, WaveletCoefficients, wt_forward_1d, wt_inverse_1d, wt_forward_mat,
       precompute_wavelets, stack_D_from_precomp, compute_mu_from_beta,
       update_cluster_wavelet_params_besov

# =============================================================================
# WAVELET STRUCTURES AND BASIC OPERATIONS
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
    
    # Daubechies wavelets (db1:db8 - only up to db8 for compatibility)
    "db1" => :haar,    # db1 is equivalent to haar
    "db2" => :db2,
    "db3" => :db3,
    "db4" => :db4,
    "db5" => :db5,
    "db6" => :db6,
    "db7" => :db7,
    "db8" => :db8,
    
    # Coiflet wavelets (coif2:coif6 - only up to coif6 for compatibility)
    "coif2" => :coif2,
    "coif4" => :coif4,
    "coif6" => :coif6,
    
    # Symlet wavelets (sym4:sym8 - only up to sym8 for compatibility)
    "sym4" => :sym4,
    "sym5" => :sym5,
    "sym6" => :sym6,
    "sym7" => :sym7,
    "sym8" => :sym8,
    
    # Battle wavelets (batt2:batt6 - only up to batt6 for compatibility)
    "batt2" => :batt2,
    "batt4" => :batt4,
    "batt6" => :batt6,
)

function wavelet_from_string(wf::String, boundary::String = "periodic")
    sym = Symbol(lowercase(wf))
    sym = get(_WAVELET_ALIASES, String(sym), sym)
    if isdefined(WT, sym)
        ext = extension_from_string(boundary)
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

function wt_forward_1d(y::AbstractVector; wf::String = "sym8", J::Union{Nothing,Int} = nothing, boundary::String = "periodic")
    P = length(y)
    Jv = Utils.ensure_dyadic_J(P, J)
    wave = wavelet_from_string(wf, boundary)
    wt = dwt(Float64.(y), wave, Jv)
    
    # The dwt function returns a flat vector of coefficients
    # We need to organize them into detail and approximation coefficients
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
end

function wt_inverse_1d(coeff_vec::AbstractVector, map::WaveletMap)
    wave = wavelet_from_string(map.wf, map.boundary)
    
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

# =============================================================================
# WAVELET BLOCK SAMPLING FUNCTIONS
# =============================================================================

function ensure_gamma_length!(wpar::Utils.WaveletParams, ncoeff::Int, M::Int)
    if length(wpar.gamma_ch) != M || any(length(g) != ncoeff for g in wpar.gamma_ch)
        wpar.gamma_ch = [Int.(Base.rand(Bernoulli(0.2), ncoeff)) for _ in 1:M]
    end
end

function update_cluster_wavelet_params_besov(idx::Vector{Int}, precomp, M::Int, wpar::Utils.WaveletParams,
    sigma2_m::Vector{Float64}, tau_sigma::Float64;
    kappa_pi::Float64 = 0.6, c2::Float64 = 1.0, tau_pi::Float64 = 40.0,
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
    det_names = sort(filter(name -> startswith(name, "d"), lev_names))  # "d1","d2",...
    s_name = sort(filter(name -> startswith(name, "s"), lev_names))  # usually ["sJ"]

    if isempty(wpar.pi_level)
        wpar.pi_level = Dict(name => 0.5 for name in det_names)
    end
    if isempty(wpar.g_level)
        wpar.g_level = Dict(name => 2.0 for name in det_names)
    end
    ensure_gamma_length!(wpar, ncoeff, M)

    # 1) gamma updates
    for m in 1:M
        Dm = view(D, :, :, m)
        gam = wpar.gamma_ch[m]
        for lev in det_names
            ids = maps[m].idx[Symbol(lev)]
            isempty(ids) && continue
            pi_j = wpar.pi_level[lev]
            g_j = wpar.g_level[lev]
            Dsub = Dm[ids, :]
            v_spike = sigma2_m[m]
            v_slab = (1 + g_j) * sigma2_m[m]
            ll_spike = -0.5 .* sum(log.(2π * v_spike) .+ (Dsub .^ 2) ./ v_spike; dims = 2)
            ll_slab = -0.5 .* sum(log.(2π * v_slab) .+ (Dsub .^ 2) ./ v_slab; dims = 2)
            logit_val = log(pi_j) .+ ll_slab .- (log(1 - pi_j) .+ ll_spike)
            p1 = logistic.(clamp.(logit_val, -35, 35))
            gam[ids] = Int.(Base.rand.(Bernoulli.(vec(p1))))
        end
        if length(s_name) == 1
            ids_s = maps[m].idx[Symbol(s_name[1])]
            if !isempty(ids_s)
                gam[ids_s] .= 1
            end
        end
        wpar.gamma_ch[m] = gam
    end

    # 2) pi_level updates
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

    # 3) beta sampling
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

    # 4) g_level updates
    for lev in det_names
        shape0 = isnothing(g_hyp) ? 2.0 : g_hyp[lev]["shape"]
        rate0  = isnothing(g_hyp) ? 2.0 : g_hyp[lev]["rate"]

        ss_over_sigma = 0.0
        n_sel_total   = 0

        for m in 1:M
            ids = maps[m].idx[Symbol(lev)]
            isempty(ids) && continue

            # Active coefficient indices at this level for channel m
            act = wpar.gamma_ch[m][ids] .== 1
            any(act) || continue

            # Use sampled beta, not data
            βm = beta_ch[m]
            act_ids = ids[act]
            ss_over_sigma += sum( (βm[act_ids].^2) ) / sigma2_m[m]
            n_sel_total   += length(act_ids)
        end

        shape_post = shape0 + 0.5 * n_sel_total
        rate_post  = rate0  + 0.5 * ss_over_sigma
        wpar.g_level[lev] = Base.rand(InverseGamma(shape_post, rate_post))
    end

    # 5) sigma2 updates & 6) tau_sigma
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

end # module