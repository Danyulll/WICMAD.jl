#!/usr/bin/env julia

using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=dirname(@__DIR__))

using WICMAD
using WICMAD.Utils: load_ucr_dataset, prepare_anomaly_dataset, summarize_dataset, default_time_index
using WICMAD.PostProcessing: dahl_from_res, map_from_res
using WICMAD.Plotting: plot_clustering_summary
using Random
using StatsBase: sample, countmap
using Statistics: mean
using Printf
using Interpolations

Random.seed!(42)

data_dir = joinpath(dirname(@__DIR__), "data")

println("Loading Blink dataset (combining TRAIN + TEST)...")
ds_loaded = load_ucr_dataset("Blink/Blink", data_dir)

println("Preparing anomaly dataset with 15% anomalies (single anomaly group)...")
prepped = prepare_anomaly_dataset(ds_loaded; anomaly_ratio=0.15, rng=Random.default_rng())
println("$(summarize_dataset(prepped))")

# Interpolate each multivariate series to 32 time points (power-of-two)
function interpolate_to_length(series::Vector{Matrix{Float64}}; target_len::Int = 32)
    out = Vector{Matrix{Float64}}(undef, length(series))
    for (i, X) in pairs(series)
        P0, M0 = size(X)
        x_old = collect(1:P0)
        x_new = collect(range(1, P0, length=target_len))
        Xn = Array{Float64}(undef, target_len, M0)
        for m in 1:M0
            itp = linear_interpolation(x_old, @view X[:, m])
            Xn[:, m] = itp.(x_new)
        end
        out[i] = Xn
    end
    return out, target_len
end

Y, P = interpolate_to_length(prepped.series; target_len=32)
M = size(Y[1], 2)
t = collect(1:P)

# Reveal 15% of the normal group
normal_idx = findall(==(0), prepped.binary_labels)
n_reveal = max(1, round(Int, 0.15 * length(normal_idx)))
revealed_subset = isempty(normal_idx) ? Int[] : sample(normal_idx, min(n_reveal, length(normal_idx)); replace=false)
revealed_idx = sort(revealed_subset)

println("Samples: $(length(Y)), Length: $(P), Dims: $(M)")
println("Revealed normal count: $(length(revealed_idx)) (15% target)")

# Run WICMAD (multivariate; no derivatives added)
config = (
    n_iter=3500,
    burn=1500,
    thin=1,
    warmup_iters=400,
    alpha_prior=(12.0, 1.0),
    a_eta=2.5, b_eta=0.08,
    a_sig=2.5, b_sig=0.02,
    wf="sym8",
)

println("Running WICMAD...")
res = wicmad(Y, t;
    n_iter=config.n_iter,
    burn=config.burn,
    thin=config.thin,
    warmup_iters=config.warmup_iters,
    alpha_prior=config.alpha_prior,
    a_eta=config.a_eta, b_eta=config.b_eta,
    a_sig=config.a_sig, b_sig=config.b_sig,
    revealed_idx=revealed_idx,
    diagnostics=true,
    wf=config.wf,
    bootstrap_runs=0,
)

# Utility: compute confusion counts given binary truth and predictions
confusion(truth::Vector{Int}, pred::Vector{Int}) = (
    tn = sum((truth .== 0) .& (pred .== 0)),
    fp = sum((truth .== 0) .& (pred .== 1)),
    fn = sum((truth .== 1) .& (pred .== 0)),
    tp = sum((truth .== 1) .& (pred .== 1)),
)

# Convert cluster assignment to anomaly labels by taking the largest cluster as normal
function clusters_to_binary(z::Vector{Int})
    cc = countmap(z)
    biggest = argmax(cc)
    [zi == biggest ? 0 : 1 for zi in z]
end

# Best-F1 scan across MCMC samples
function best_f1_from_chain(Z::AbstractMatrix{<:Integer}, truth::Vector{Int})
    best = (-1.0, zeros(Int, length(truth)))
    for s in axes(Z, 1)
        pred = clusters_to_binary(vec(Z[s, :]))
        tp = sum((truth .== 1) .& (pred .== 1))
        fp = sum((truth .== 0) .& (pred .== 1))
        fn = sum((truth .== 1) .& (pred .== 0))
        prec = tp + fp > 0 ? tp / (tp + fp) : 0.0
        rec  = tp + fn > 0 ? tp / (tp + fn) : 0.0
        f1   = (prec + rec) > 0 ? 2 * prec * rec / (prec + rec) : 0.0
        if f1 > best[1]
            best = (f1, pred)
        end
    end
    best
end

y_true = prepped.binary_labels

# Dahl partition
dahl = dahl_from_res(res)
pred_dahl = clusters_to_binary(dahl.z_hat)
c_dahl = confusion(y_true, pred_dahl)

# MAP partition
mapr = map_from_res(res)
pred_map = clusters_to_binary(mapr.z_hat)
c_map = confusion(y_true, pred_map)

# Max F1 over chain
best_f1, pred_f1 = best_f1_from_chain(res.Z, y_true)
c_f1 = confusion(y_true, pred_f1)

println("\nConfusion Matrices (rows: True, cols: Predicted)")
println("Dahl:")
@printf("            Normal  Anomaly\n")
@printf("True Normal  %6d  %6d\n", c_dahl.tn, c_dahl.fp)
@printf("True Anomaly %6d  %6d\n", c_dahl.fn, c_dahl.tp)

println("\nMAP:")
@printf("            Normal  Anomaly\n")
@printf("True Normal  %6d  %6d\n", c_map.tn, c_map.fp)
@printf("True Anomaly %6d  %6d\n", c_map.fn, c_map.tp)

println("\nMax F1 over chain (F1=$(round(best_f1, digits=4))):")
@printf("            Normal  Anomaly\n")
@printf("True Normal  %6d  %6d\n", c_f1.tn, c_f1.fp)
@printf("True Anomaly %6d  %6d\n", c_f1.fn, c_f1.tp)

# Plots: before and after clustering, stacked per-channel
plots_dir = joinpath(dirname(@__DIR__), "plots")
mkpath(plots_dir)

save_dir = plots_dir
_ = plot_clustering_summary(Y, y_true, pred_dahl, t; save_dir=save_dir)

println("\nSaved plots to: $(save_dir)")
println("Done.")


