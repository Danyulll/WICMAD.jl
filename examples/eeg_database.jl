#!/usr/bin/env julia

using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=dirname(@__DIR__))

using WICMAD
using WICMAD.Utils: LoadedDataset, prepare_anomaly_dataset, summarize_dataset
using WICMAD.PostProcessing: dahl_from_res, map_from_res
using WICMAD.Plotting: plot_clustering_summary
using Random
using StatsBase: sample, countmap
using Printf
using Interpolations

Random.seed!(42)

data_root = joinpath(dirname(@__DIR__), "data", "eeg+database")
train_dir = joinpath(data_root, "SMNI_CMI_TRAIN")
test_dir  = joinpath(data_root, "SMNI_CMI_TEST")

isdir(train_dir) || error("Expected extracted directory: $(train_dir). Please extract SMNI_CMI_TRAIN.tar.gz here.")
isdir(test_dir)  || error("Expected extracted directory: $(test_dir). Please extract SMNI_CMI_TEST.tar.gz here.")

function parse_rd_file(path::AbstractString)
    # Returns: trials::Vector{Matrix{Float64}}, labels::Vector{String}
    # Each matrix is (samples × channels)
    lines = readlines(path)
    isempty(lines) && return Matrix{Float64}[], String[]
    # Determine label from first header line or filename
    subj_line = first(filter(l -> startswith(l, "#"), lines))
    subj_id = replace(strip(subj_line[2:end]), "," => "")
    # Alcoholic if contains "a_" or "co2a"; control if "c_" or "co2c"
    lab = occursin("co2a", lowercase(path)) || occursin("a_", lowercase(path)) ? "alcoholic" : "control"

    # Data rows start after header; format: trial SENSOR sample value
    data_rows = Tuple{Int,String,Int,Float64}[]
    for l in lines
        s = strip(l)
        isempty(s) && continue
        startswith(s, "#") && continue
        parts = split(s)
        length(parts) < 4 && continue
        try
            trl = parse(Int, parts[1])
            sens = parts[2]
            samp = parse(Int, parts[3])
            val = parse(Float64, parts[4])
            push!(data_rows, (trl, sens, samp, val))
        catch
            continue
        end
    end

    isempty(data_rows) && return Matrix{Float64}[], String[]

    # Collect sensors and samples
    sensors = unique(getindex.(data_rows, 2))
    sort!(sensors)
    trials_idx = unique(getindex.(data_rows, 1))
    sort!(trials_idx)
    nchan = length(sensors)
    nsamp = maximum(getindex.(data_rows, 3)) + 1  # samples 0..255 -> 256

    sensor_to_idx = Dict(s => i for (i, s) in enumerate(sensors))

    trials = Vector{Matrix{Float64}}()
    labels = String[]
    for tr in trials_idx
        X = zeros(Float64, nsamp, nchan)
        for (trl, sens, samp, val) in data_rows
            trl == tr || continue
            ci = sensor_to_idx[sens]
            si = samp + 1
            X[si, ci] = val
        end
        push!(trials, X)
        push!(labels, lab)
    end
    return trials, labels
end

function collect_trials_from_dir(dir::AbstractString)
    series = Matrix{Float64}[]
    labels = String[]
    for (root, _, files) in walkdir(dir)
        for f in files
            if endswith(lowercase(f), ".rd")
                path = joinpath(root, f)
                tr, lab = parse_rd_file(path)
                append!(series, tr)
                append!(labels, lab)
            end
        end
    end
    return series, labels
end

println("Loading EEG database (train+test) from extracted SMNI directories...")
train_series, train_labels = collect_trials_from_dir(train_dir)
test_series,  test_labels  = collect_trials_from_dir(test_dir)

all_series = vcat(train_series, test_series)
all_labels = vcat(train_labels, test_labels)

@assert length(all_series) == length(all_labels) && length(all_series) > 0

ds = LoadedDataset(all_series, all_labels)

println("Preparing anomaly dataset with 15% anomalies (alcoholic as anomaly)...")
prepped = prepare_anomaly_dataset(ds; anomaly_ratio=0.15, rng=Random.default_rng())
println("$(summarize_dataset(prepped))")

# Interpolate each sequence to 64 samples
function interpolate_to_length(series::Vector{Matrix{Float64}}; target_len::Int = 64)
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

Y, P = interpolate_to_length(prepped.series; target_len=64)
M = size(Y[1], 2)
t = collect(1:P)

normal_idx = findall(==(0), prepped.binary_labels)
n_reveal = max(1, round(Int, 0.15 * length(normal_idx)))
revealed_subset = isempty(normal_idx) ? Int[] : sample(normal_idx, min(n_reveal, length(normal_idx)); replace=false)
revealed_idx = sort(revealed_subset)

println("Samples: $(length(Y)), Length: $(P), Dims: $(M)")
println("Revealed normal count: $(length(revealed_idx)) (15% target)")

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

confusion(truth::Vector{Int}, pred::Vector{Int}) = (
    tn = sum((truth .== 0) .& (pred .== 0)),
    fp = sum((truth .== 0) .& (pred .== 1)),
    fn = sum((truth .== 1) .& (pred .== 0)),
    tp = sum((truth .== 1) .& (pred .== 1)),
)

function clusters_to_binary(z::Vector{Int})
    cc = countmap(z)
    biggest = argmax(cc)
    [zi == biggest ? 0 : 1 for zi in z]
end

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

dahl = dahl_from_res(res)
pred_dahl = clusters_to_binary(dahl.z_hat)
c_dahl = confusion(y_true, pred_dahl)

mapr = map_from_res(res)
pred_map = clusters_to_binary(mapr.z_hat)
c_map = confusion(y_true, pred_map)

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

plots_dir = joinpath(dirname(@__DIR__), "plots")
mkpath(plots_dir)
_ = plot_clustering_summary(Y, y_true, pred_dahl, t; save_dir=plots_dir)

println("\nSaved plots to: $(plots_dir)")
println("Done.")


