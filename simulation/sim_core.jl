#!/usr/bin/env julia

# Ensure we use the repository root project and have deps available
begin
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
end

# Report threads available
using Base.Threads
@info "Available Julia threads: $(nthreads())"

# ------------------------------------------------------------
# WICMAD simulation experiment runner (NO FPCA)
# Shared core (helpers, dataset registry, and runner function)
# ------------------------------------------------------------

using Random, LinearAlgebra, Statistics, Printf, Dates, DelimitedFiles
using StatsBase: countmap
using DataFrames, CSV
using Plots
using WICMAD
using Logging

# Per-MC-run prefixed logger so all messages printed from a given thread/run
# include the Monte Carlo index for easier tracking in parallel output
struct PrefixLogger <: AbstractLogger
    parent::AbstractLogger
    prefix::String
end

Logging.min_enabled_level(logger::PrefixLogger) = Logging.min_enabled_level(logger.parent)
Logging.shouldlog(logger::PrefixLogger, level, _module, group, id) = Logging.shouldlog(logger.parent, level, _module, group, id)
Logging.catch_exceptions(logger::PrefixLogger) = Logging.catch_exceptions(logger.parent)

function Logging.handle_message(logger::PrefixLogger, level, message, _module, group, id, file, line; kwargs...)
    prefixed = string(logger.prefix, message)
    return Logging.handle_message(logger.parent, level, prefixed, _module, group, id, file, line; kwargs...)
end

with_mc_logger(mc_idx::Integer, f::Function) = Logging.with_logger(PrefixLogger(Logging.current_logger(), string("(mc ", mc_idx, ") "))) do
    f()
end

# Accept (function, mc_idx) ordering too, to be resilient to macro expansions
with_mc_logger(f::Function, mc_idx::Integer) = Logging.with_logger(PrefixLogger(Logging.current_logger(), string("(mc ", mc_idx, ") "))) do
    f()
end

# -----------------------------
# Global controls
# -----------------------------
const P_use        = 32
const t_grid       = range(0.0, 1.0; length=P_use) |> collect
const Δ            = 1/(P_use - 1)
const σ_noise      = 0.05

const mc_runs      = 1
const base_seed    = 0x000000000135CFF1 % UInt64

# Sampler controls
const n_iter       = 5_000
const burnin       = 2_000
const thin         = 1
const warmup_iters = 500

# Semi-supervised: reveal 15% normals
const reveal_prop  = 0.15

# Output (bucketed per top-level script name, e.g., sim1, sim2, ...)
const timestamp    = Dates.format(now(), "yyyymmdd_HHMMSS")
const sim_name     = splitext(basename(PROGRAM_FILE))[1]
# Save under a folder that reflects the simulation script name
const out_root     = joinpath("simstudy_results_nofpca", string(sim_name, "_", timestamp))
const png_dir      = joinpath(out_root, "png")
const metrics_csv  = joinpath(out_root, "summary_metrics.csv")
const perrun_csv   = joinpath(out_root, "metrics_per_run.csv")

mkpath(png_dir)
@info "Writing outputs to: $(abspath(out_root))"

# -----------------------------
# Mean and kernel
# -----------------------------
mean_fun(t) = 0.6 .* sin.(4π .* t) .+ 0.25 .* cos.(10π .* t) .+ 0.1 .* t

"""
Quasi-periodic + long-scale SE kernel:
k(t,t') = sig1^2 * SE(ℓ1) ⊙ PER(p, ℓp) + sig2^2 * SE(ℓ2)
"""
function k_qp(t::AbstractVector, tp::AbstractVector;
              ell1=0.15, sig1_sq=1.0, p=0.30, ellp=0.30,
              ell2=0.60, sig2_sq=0.4)
    dt  = reshape(t, :, 1) .- reshape(tp, 1, :)
    se1 = @. exp(-(dt^2) / (2 * ell1^2))
    per = @. exp(- 2 * (sin(π * dt / p))^2 / (ellp^2))
    se2 = @. exp(-(dt^2) / (2 * ell2^2))
    sig1_sq .* (se1 .* per) .+ sig2_sq .* se2
end

"""
Draw GP curves (N functions) on grid t with mean mu_fun and covariance k_fun.
Returns an N×P matrix (rows are functions), with iid N(0, σ_eps^2) noise added.
"""
function gp_draw_matrix(N::Int, t::AbstractVector,
                        mu_fun::Function, k_fun::Function,
                        σ_eps::Real)
    P  = length(t)
    μ  = mu_fun(t)
    K  = k_fun(t, t)
    Kc = Symmetric(K + 1e-8I)
    L  = cholesky(Kc).L
    Z  = randn(P, N)
    G  = μ .+ L * Z
    X  = permutedims(G) .+ σ_eps .* randn(N, P)
    return X
end

# -----------------------------
# Anomaly perturbations
# -----------------------------
function add_isolated(xrow::AbstractVector, t::AbstractVector)
    P  = length(t)
    i0 = rand(3:(P-2))
    w  = rand(0.3*Δ:0.1*Δ:0.8*Δ)
    S  = rand((-1, +1))
    A  = rand(8.0:0.1:12.0)
    bump = @. S * A * exp(-(t - t[i0])^2 / (2w^2))
    return xrow .+ bump
end

function add_mag1(xrow::AbstractVector)
    S = rand((-1, +1))
    A = rand(12.0:0.1:15.0)
    return xrow .+ S*A
end

function add_mag2(xrow::AbstractVector, t::AbstractVector)
    P      = length(t)
    λ      = floor(Int, 0.10P)
    s      = rand(1:(P - λ + 1))
    S      = rand((-1, +1))
    A      = rand(10.0:0.1:15.0)
    rfun(u) = 0.5 * (1 - cos(π * u))
    bump   = zeros(Float64, P)
    idx    = s:(s + λ - 1)
    u      = (collect(0:(length(idx)-1))) ./ (λ - 1)
    bump[idx] .= S .* A .* rfun.(u)
    return xrow .+ bump
end

function add_shape(xrow::AbstractVector, t::AbstractVector)
    U = rand(0.2:0.01:2.0)
    return xrow .+ 3 .* sin.(2π .* U .* t)
end

# -----------------------------
# Dataset generators
# -----------------------------
function make_univariate_dataset(; N::Int=300, anomaly_type::Symbol=:isolated,
                                 t::AbstractVector=t_grid)
    X      = gp_draw_matrix(N, t, mean_fun, k_qp, σ_noise)
    y_true = zeros(Int, N)
    n_anom = max(1, round(Int, 0.15N))
    idx_anom = randperm(N)[1:n_anom]

    Xpert = copy(X)
    for i in idx_anom
        xi = @view X[i, :]
        if anomaly_type == :isolated
            Xpert[i, :] = add_isolated(xi, t)
        elseif anomaly_type == :mag1
            Xpert[i, :] = add_mag1(xi)
        elseif anomaly_type == :mag2
            Xpert[i, :] = add_mag2(xi, t)
        elseif anomaly_type == :shape
            Xpert[i, :] = add_shape(xi, t)
        else
            error("Unknown anomaly type: $anomaly_type")
        end
    end
    y_true[idx_anom] .= 1

    Y_list = [ reshape(@view(Xpert[i, :]), (length(t), 1)) for i in 1:N ]
    return (; Y_list, t, y_true)
end

function make_multivariate_dataset(; N::Int=300, regime::Symbol=:one,
                                   t::AbstractVector=t_grid)
    P = length(t)
    U1 = gp_draw_matrix(N, t, mean_fun, k_qp, 0.0)
    U2 = gp_draw_matrix(N, t, mean_fun, k_qp, 0.0)
    A  = [1.0  0.4;
          0.2  1.0;
          0.7 -0.3]

    Y = Vector{Matrix{Float64}}(undef, N)
    for i in 1:N
        Ui = hcat(@view(U1[i, :]), @view(U2[i, :]))
        Xi = Ui * A'
        Xi .+= σ_noise .* randn(P, 3)
        Y[i] = Xi
    end

    y_true   = zeros(Int, N)
    n_anom   = max(1, round(Int, 0.15N))
    idx_anom = randperm(N)[1:n_anom]
    y_true[idx_anom] .= 1

    types = (:isolated, :mag1, :mag2, :shape)

    apply_perturb(row::AbstractVector, type::Symbol) = begin
        if     type == :isolated; add_isolated(row, t)
        elseif type == :mag1;     add_mag1(row)
        elseif type == :mag2;     add_mag2(row, t)
        elseif type == :shape;    add_shape(row, t)
        else  row end
    end

    # Track which channels have anomalies for each observation
    channel_anomalies = zeros(Bool, N, 3)  # N observations, 3 channels
    
    for i in idx_anom
        Xi = copy(Y[i])
        if regime == :one
            ch   = 1  # Always use channel 1 for sim9 (one channel regime)
            type = rand(types)
            Xi[:, ch] = apply_perturb(@view(Xi[:, ch]), type)
            channel_anomalies[i, ch] = true  # Mark this channel as anomalous
        elseif regime == :two
            chs  = [1, 2]  # Always use channels 1 and 2 for sim10 (two channel regime)
            type = rand(types)
            for ch in chs
                Xi[:, ch] = apply_perturb(@view(Xi[:, ch]), type)
                channel_anomalies[i, ch] = true  # Mark these channels as anomalous
            end
        elseif regime == :three
            # Ensure each channel gets a different anomaly type
            types_vec = collect(types)  # Convert tuple to vector: [:isolated, :mag1, :mag2, :shape]
            shuffled_types = Random.shuffle(types_vec)  # Randomly shuffle the 4 types
            for (ch_idx, ch) in enumerate(1:3)
                type = shuffled_types[ch_idx]  # Assign different type to each channel
                Xi[:, ch] = apply_perturb(@view(Xi[:, ch]), type)
                channel_anomalies[i, ch] = true  # Mark all channels as anomalous
            end
        else
            error("Unknown regime: $regime")
        end
        Y[i] = Xi
    end

    return (; Y_list=Y, t, y_true, channel_anomalies=channel_anomalies)
end

# -----------------------------
# Channel anomalies expansion for derivatives
# -----------------------------
function expand_channel_anomalies_for_derivatives(ca::AbstractMatrix{Bool}, Y_list)
    Mplot = size(Y_list[1], 2)
    Mraw  = size(ca, 2)
    # If derivatives triple the channels, tile flags across each triplet
    if Mplot % Mraw == 0
        reps = Mplot ÷ Mraw
        return reduce(hcat, (repeat(ca[:, j:j], 1, reps) for j in 1:Mraw))
    else
        @warn "Cannot expand channel anomalies: Mplot=$(Mplot) not a multiple of Mraw=$(Mraw)"
        return nothing
    end
end

# -----------------------------
# Derivatives transform
# -----------------------------
function derivatives_transform(Y_list::Vector{<:AbstractMatrix}, t::AbstractVector)
    N = length(Y_list)
    P = length(t)
    function fd1(x::AbstractVector)
        d1 = similar(x)
        d1[1]   = (x[2] - x[1]) / (t[2] - t[1])
        d1[end] = (x[end] - x[end-1]) / (t[end] - t[end-1])
        @inbounds for i in 2:P-1
            d1[i] = (x[i+1] - x[i-1]) / (t[i+1] - t[i-1])
        end
        d1
    end
    function fd2(x::AbstractVector)
        d2 = similar(x)
        d2[1]   = (x[3] - 2x[2] + x[1]) / ((t[2]-t[1])^2)
        d2[end] = (x[end] - 2x[end-1] + x[end-2]) / ((t[end]-t[end-1])^2)
        @inbounds for i in 2:P-1
            dt1 = t[i] - t[i-1]; dt2 = t[i+1] - t[i]
            d2[i] = 2 * ( (x[i+1]-x[i]) / (dt2*(dt1+dt2)) - (x[i]-x[i-1]) / (dt1*(dt1+dt2)) )
        end
        d2
    end
    out = Vector{Matrix{Float64}}(undef, N)
    for i in 1:N
        Yi = Y_list[i]
        M  = size(Yi, 2)
        O  = zeros(Float64, P, 3M)
        for m in 1:M
            col = @view Yi[:, m]
            O[:, 3(m-1) + 1] = col
            O[:, 3(m-1) + 2] = fd1(col)
            O[:, 3(m-1) + 3] = fd2(col)
        end
        out[i] = O
    end
    return out
end

# -----------------------------
# Plot helpers
# -----------------------------
function plot_before!(Y_list, t, y_true; title::AbstractString, path::AbstractString, 
                       channel_anomalies::Union{Nothing,Matrix{Bool}}=nothing)
    P = length(t)
    M = size(Y_list[1], 2)
    plt = plot(layout=(M,1), size=(800, 200*M), legend=:topright, title=title)
    
    # Check if we have channel_anomalies and it matches dimensions exactly
    use_channel_specific = !isnothing(channel_anomalies) && 
                           size(channel_anomalies, 1) == length(Y_list) && 
                           size(channel_anomalies, 2) == M
    
    for m in 1:M
        # plot each series individually to avoid group-recipe issues
        for (i, Xi) in enumerate(Y_list)
            # If channel_anomalies is provided and valid (multivariate), use per-channel coloring
            # Otherwise, use observation-level coloring (univariate)
            is_anomaly = if use_channel_specific
                channel_anomalies[i, m]  # This specific channel has anomaly
            else
                y_true[i] == 1  # Entire observation is anomalous
            end
            
            if is_anomaly
                plot!(plt[m], t, @view(Xi[:, m]); color=:red,  alpha=0.6, label="")
            else
                plot!(plt[m], t, @view(Xi[:, m]); color=:blue, alpha=0.6, label="")
            end
        end
        plot!(plt[m]; title="Channel $m")
        # Single legend entries
        plot!(plt[m], [NaN], [NaN], color=:blue, label="Normal")
        plot!(plt[m], [NaN], [NaN], color=:red,  label="Anomaly")
    end
    savefig(plt, path)
end

function plot_after!(Y_list, t, pred_anom; title::AbstractString, path::AbstractString,
                      channel_anomalies::Union{Nothing,Matrix{Bool}}=nothing)
    P = length(t)
    M = size(Y_list[1], 2)
    plt = plot(layout=(M,1), size=(800, 200*M), legend=:topright, title=title)
    
    # Check if we have channel_anomalies and it matches dimensions exactly
    use_channel_specific = !isnothing(channel_anomalies) && 
                           size(channel_anomalies, 1) == length(Y_list) && 
                           size(channel_anomalies, 2) == M
    
    for m in 1:M
        # plot each series individually to avoid group-recipe issues
        for (i, Xi) in enumerate(Y_list)
            # For "after" plots, use channel_anomalies to show which channels were actually anomalous
            # (we want to see the ground truth channel-specific anomalies, not predictions)
            is_anomaly = if use_channel_specific
                channel_anomalies[i, m]  # Use channel_anomalies to show ground truth
            else
                pred_anom[i] == 1  # Fall back to observation-level prediction
            end
            
            if is_anomaly
                plot!(plt[m], t, @view(Xi[:, m]); color=:red,  alpha=0.6, label="")
            else
                plot!(plt[m], t, @view(Xi[:, m]); color=:blue, alpha=0.6, label="")
            end
        end
        plot!(plt[m]; title="Channel $m")
        # Single legend entries
        plot!(plt[m], [NaN], [NaN], color=:blue, label="Normal")
        plot!(plt[m], [NaN], [NaN], color=:red,  label="Anomaly")
    end
    savefig(plt, path)
end

# -----------------------------
# Metrics
# -----------------------------
function metrics_from_preds(y_true::Vector{Int}, pred_anom::Vector{Int})
    @assert length(y_true) == length(pred_anom)
    tn = sum((y_true .== 0) .& (pred_anom .== 0))
    fp = sum((y_true .== 0) .& (pred_anom .== 1))
    fn = sum((y_true .== 1) .& (pred_anom .== 0))
    tp = sum((y_true .== 1) .& (pred_anom .== 1))
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = (tp + fp) == 0 ? NaN : tp / (tp + fp)
    rec  = (tp + fn) == 0 ? NaN : tp / (tp + fn)
    f1   = (isnan(prec) || isnan(rec) || (prec + rec) == 0) ? NaN : 2 * prec * rec / (prec + rec)
    return (accuracy=acc, precision=prec, recall=rec, f1=f1)
end

# -----------------------------
# Confusion matrix helpers
# -----------------------------
function confusion_matrix(truth::Vector{Int}, pred::Vector{Int})
    (
        tn = sum((truth .== 0) .& (pred .== 0)),
        fp = sum((truth .== 0) .& (pred .== 1)),
        fn = sum((truth .== 1) .& (pred .== 0)),
        tp = sum((truth .== 1) .& (pred .== 1)),
    )
end

function clusters_to_binary(z::Vector{Int}, revealed_idx::Vector{Int})
    # Find normal label from revealed indices
    if !isempty(revealed_idx)
        labs = z[revealed_idx]
        counts = countmap(labs)
        normal_label = argmax(counts)
    else
        # If no revealed indices, use largest cluster
        counts = countmap(z)
        normal_label = argmax(counts)
    end
    [zi == normal_label ? 0 : 1 for zi in z]
end

function print_confusion_matrix(label::AbstractString, c::NamedTuple, dataset_label::AbstractString)
    println("\n[$dataset_label] Confusion Matrix - $label:")
    println("            Normal  Anomaly")
    @printf("True Normal  %6d  %6d\n", c.tn, c.fp)
    @printf("True Anomaly %6d  %6d\n", c.fn, c.tp)
end

# -----------------------------
# One dataset for one MC seed
# -----------------------------
function run_one(dataset_label::AbstractString, dataset_title::AbstractString,
                 representation::Symbol, make_data_fn::Function, mc_idx::Int)
    Random.seed!(UInt64(base_seed) + UInt64(mc_idx))
    dat    = make_data_fn()
    Y_raw  = dat.Y_list
    t      = dat.t
    y_true = dat.y_true
    N      = length(Y_raw)
    # Extract channel_anomalies if it exists (for multivariate datasets)
    channel_anomalies = try
        ca = dat.channel_anomalies
        Mraw = size(Y_raw[1], 2)  # number of channels in the raw representation
        if size(ca, 1) == N && size(ca, 2) == Mraw
            ca  # Valid channel_anomalies matrix
        else
            @warn "channel_anomalies has wrong dimensions $(size(ca)); expected (N=$(N), Mraw=$(Mraw)). Using observation-level coloring."
            nothing
        end
    catch
        nothing  # Field doesn't exist (univariate dataset)
    end

    Y_list = if representation == :raw
        Y_raw
    elseif representation == :derivatives
        derivatives_transform(Y_raw, t)
    else
        error("Unknown representation: $representation")
    end

    # Expand channel_anomalies if needed for derivatives representation
    if !isnothing(channel_anomalies)
        Mplot = size(Y_list[1], 2)
        if size(channel_anomalies, 2) != Mplot
            channel_anomalies = expand_channel_anomalies_for_derivatives(channel_anomalies, Y_list)
        end
    end

    for i in 1:length(Y_list)
        Xi = Y_list[i]
        @inbounds for j in eachindex(Xi)
            if !isfinite(Xi[j]); Xi[j] = 0.0; end
        end
        Y_list[i] = Xi
    end

    normals_idx = findall(==(0), y_true)
    n_reveal    = max(1, floor(Int, reveal_prop * length(normals_idx)))
    reveal_idx  = n_reveal == 0 ? Int[] : rand(normals_idx, n_reveal)

    res = nothing
    try
        res = WICMAD.wicmad(
            Y_list, t;
            n_iter=n_iter, burn=burnin, thin=thin, warmup_iters=warmup_iters,
            bootstrap_runs=0,
            revealed_idx=reveal_idx, unpin=false
        )
    catch e
        @warn "Error in WICMAD for dataset $dataset_label: $(e)"
        Zs = fill(1, n_iter - burnin, N)
        res = (; Z=Zs, K_occ=fill(1, size(Zs,1)), loglik=zeros(size(Zs,1)))
    end

    # Compute Dahl estimate
    dahl = WICMAD.dahl_from_res(res)
    pred_dahl = clusters_to_binary(dahl.z_hat, reveal_idx)
    c_dahl = confusion_matrix(y_true, pred_dahl)
    
    # Compute MAP estimate
    mapr = WICMAD.map_from_res(res)
    pred_map = clusters_to_binary(mapr.z_hat, reveal_idx)
    c_map = confusion_matrix(y_true, pred_map)
    
    # Print confusion matrices (only for first MC run to avoid clutter)
    if mc_idx == 1
        println("\n" * "="^60)
        println("Confusion Matrices for $dataset_label ($representation)")
        println("="^60)
        print_confusion_matrix("Dahl Estimate", c_dahl, dataset_label)
        print_confusion_matrix("MAP Estimate", c_map, dataset_label)
        println("="^60)
    end
    
    # Use Dahl estimate for metrics (existing behavior)
    pred_anom = pred_dahl

    if mc_idx == 1
        tag = string(dataset_label, "_", Symbol(representation))
        # Debug: check if channel_anomalies is available for multivariate datasets
        if startswith(dataset_label, "mv_")
            if isnothing(channel_anomalies)
                @warn "channel_anomalies is nothing for multivariate dataset $dataset_label - plots will show all channels"
            else
                # Count how many channels have anomalies per observation
                n_anom_obs = sum(y_true .== 1)
                if n_anom_obs > 0
                    channel_counts = sum(channel_anomalies[y_true .== 1, :], dims=1)
                    @info "Multivariate dataset $dataset_label: anomaly channels per observation - Channel 1: $(channel_counts[1]), Channel 2: $(channel_counts[2]), Channel 3: $(channel_counts[3])"
                end
            end
        end
        plot_before!(Y_list, t, y_true; title=dataset_title,
                     path=joinpath(png_dir, string(tag, "_before.png")),
                     channel_anomalies=channel_anomalies)
        plot_after!(Y_list, t, pred_anom; title=dataset_title,
                    path=joinpath(png_dir, string(tag, "_after.png")),
                    channel_anomalies=channel_anomalies)
    end

    met = metrics_from_preds(y_true, pred_anom)
    return (dataset=dataset_label,
            representation=String(representation),
            mc_run=mc_idx,
            accuracy=met.accuracy,
            precision=met.precision,
            recall=met.recall,
            f1=met.f1,
            map_tn=c_map.tn,
            map_fp=c_map.fp,
            map_fn=c_map.fn,
            map_tp=c_map.tp)
end

# -----------------------------
# Dataset registry
# -----------------------------
univariate_specs = [
    (id="uni_isolated",  title="Isolated",
     fn=() -> make_univariate_dataset(; anomaly_type=:isolated, t=t_grid)),
    (id="uni_mag1",      title="Magnitude I",
     fn=() -> make_univariate_dataset(; anomaly_type=:mag1, t=t_grid)),
    (id="uni_mag2",      title="Magnitude II",
     fn=() -> make_univariate_dataset(; anomaly_type=:mag2, t=t_grid)),
    (id="uni_shape",     title="Shape",
     fn=() -> make_univariate_dataset(; anomaly_type=:shape, t=t_grid))
]

multivariate_specs = [
    (id="mv_one_channel",    title="One Channel",
     fn=() -> make_multivariate_dataset(; regime=:one, t=t_grid)),
    (id="mv_two_channels",   title="Two Channels",
     fn=() -> make_multivariate_dataset(; regime=:two, t=t_grid)),
    (id="mv_three_channels", title="Three Channels",
     fn=() -> make_multivariate_dataset(; regime=:three, t=t_grid))
]

dataset_specs = NamedTuple[]
for spec in univariate_specs
    push!(dataset_specs, (id=string(spec.id, "_raw"),  title=spec.title, representation=:raw,         fn=spec.fn))
    push!(dataset_specs, (id=string(spec.id, "_deriv"), title=spec.title, representation=:derivatives, fn=spec.fn))
end
for spec in multivariate_specs
    push!(dataset_specs, (id=spec.id, title=spec.title, representation=:raw, fn=spec.fn))
end

# -----------------------------
# Runner function (optionally filter dataset ids)
# -----------------------------
function run_datasets(filter_ids::Union{Nothing,Vector{String}}=nothing)
    rows = Vector{NamedTuple}(undef, 0)
    selected = isnothing(filter_ids) ? dataset_specs : filter(s -> s.id in filter_ids, dataset_specs)
    for spec in selected
        @info "[Dataset: $(spec.id)] Running $(mc_runs) MC replications ($(spec.representation))..."
        rows_spec = Vector{NamedTuple}(undef, mc_runs)
        Threads.@threads for mc_idx in 1:mc_runs
            with_mc_logger(mc_idx) do
                @info "Starting run $(mc_idx)/$(mc_runs)"
                rows_spec[mc_idx] = run_one(spec.id, spec.title, spec.representation, spec.fn, mc_idx)
                @info "Finished run $(mc_idx)/$(mc_runs)"
            end
        end
        append!(rows, rows_spec)
    end
    df = DataFrame(rows)
    summary_df = combine(groupby(df, [:dataset, :representation])) do sdf
        (; accuracy=mean(skipmissing(sdf.accuracy)),
           precision=mean(skipmissing(sdf.precision)),
           recall=mean(skipmissing(sdf.recall)),
           f1=mean(skipmissing(sdf.f1)),
           mean_map_tn=mean(skipmissing(sdf.map_tn)),
           mean_map_fp=mean(skipmissing(sdf.map_fp)),
           mean_map_fn=mean(skipmissing(sdf.map_fn)),
           mean_map_tp=mean(skipmissing(sdf.map_tp)))
    end
    CSV.write(metrics_csv, summary_df)
    @info "Saved summary metrics CSV to: $(abspath(metrics_csv))"
    CSV.write(perrun_csv, df)
    @info "Saved per-run metrics CSV to: $(abspath(perrun_csv))"
    @info "PNG directory (first-run visualizations): $(abspath(png_dir))"
end


