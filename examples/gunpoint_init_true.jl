#!/usr/bin/env julia

using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=dirname(@__DIR__))

using WICMAD
using Random
using Statistics
using StatsBase: countmap
using Interpolations
using Plots

Random.seed!(42)

function load_gunpoint_data(filepath)
    lines = readlines(filepath)
    data_rows = Vector{Vector{Float64}}()
    for line in lines
        s = strip(line)
        isempty(s) && continue
        parts = filter(!isempty, split(s, ' '))
        push!(data_rows, parse.(Float64, parts))
    end
    data = hcat(data_rows...)'
    labels = Int.(data[:, 1])
    X = data[:, 2:end]
    return X, labels
end

# Load train+test
train_path = joinpath(@__DIR__, "../data/GunPoint/GunPoint_TRAIN.txt")
test_path  = joinpath(@__DIR__, "../data/GunPoint/GunPoint_TEST.txt")
Xtr, ytr = load_gunpoint_data(train_path)
Xte, yte = load_gunpoint_data(test_path)
X = vcat(Xtr, Xte)
y = vcat(ytr, yte)

println("Loaded GunPoint: N=$(size(X,1)), P=$(size(X,2)), classes=$(sort(unique(y)))")
println("Class counts: ", countmap(y))

# Interpolate to power-of-two grid (32)
n_original = size(X, 2)
n_target = 32
Xin = zeros(size(X, 1), n_target)
for i in 1:size(X,1)
    itp = linear_interpolation(1:n_original, X[i, :])
    Xin[i, :] = itp.(collect(range(1, n_original, length=n_target)))
end
X = Xin

cm = countmap(y)
classes_sorted = sort(collect(keys(cm)); by = c -> -cm[c])
normal_class = classes_sorted[1]
anomaly_class = classes_sorted[2]

normal_indices_full = findall(==(normal_class), y)
anomaly_indices_full = findall(==(anomaly_class), y)

p_anom = 0.15
ratio = p_anom / (1 - p_anom)

avail_norm = length(normal_indices_full)
avail_anom = length(anomaly_indices_full)

n_anom_max_by_norm = floor(Int, ratio * avail_norm)
n_anomaly_used = min(avail_anom, n_anom_max_by_norm)
n_normal_used = min(avail_norm, round(Int, n_anomaly_used * (1 - p_anom) / p_anom))

using StatsBase: sample
used_normal_indices = sample(normal_indices_full, n_normal_used, replace=false)
used_anomaly_indices = sample(anomaly_indices_full, n_anomaly_used, replace=false)
all_used_indices = vcat(used_normal_indices, used_anomaly_indices)
shuffle!(all_used_indices)

X = X[all_used_indices, :]
y = y[all_used_indices]

println("Using N=$(size(X,1)) with ~15% anomalies. Counts: ", countmap(y))

Y = [X[i, :] for i in 1:size(X,1)]
t = collect(1:size(X,2))

plots_dir = joinpath(@__DIR__, "../plots/init_exp/gunpoint")
isdir(plots_dir) || mkpath(plots_dir)

normal_indices = findall(==(normal_class), y)
n_revealed = max(1, round(Int, 0.15 * length(normal_indices)))
revealed_subset = sample(normal_indices, n_revealed, replace=false)
revealed_idx = sort(revealed_subset)
println("Revealing $(length(revealed_idx)) of $(length(normal_indices)) normals for wavelet selection (15%)")
if !isempty(revealed_idx)
    println("Selecting wavelet using $(length(revealed_idx)) revealed normals...")
    Y_mats = [reshape(Float64.(Yi), :, 1) for Yi in Y]
    sel = WICMAD.KernelSelection.select_wavelet(Y_mats, t, revealed_idx;
        wf_candidates = nothing, J = nothing, boundary = "periodic",
        mcmc = (n_iter=3000, burnin=1000, thin=1))
    selected_wf = sel.selected_wf
    println("Selected wavelet: ", selected_wf)
else
    selected_wf = "sym8"
    println("No revealed indices available; defaulting to wavelet: ", selected_wf)
end

function relabel_consecutive(z::Vector{Int})
    labs = unique(z)
    m = Dict(lab => i for (i, lab) in enumerate(labs))
    return [m[v] for v in z]
end
z_true = relabel_consecutive(collect(y))

gt_binary = [yi == normal_class ? 0 : 1 for yi in y]

function largest_cluster(z::AbstractVector{Int})
    counts = countmap(z)
    keys_sorted = sort(collect(keys(counts)); by = k -> -counts[k])
    return keys_sorted[1]
end

function to_binary_labels(z::AbstractVector{Int})
    k_norm = largest_cluster(z)
    return [zi == k_norm ? 0 : 1 for zi in z]
end

function print_confusion(name::String, ytrue::Vector{Int}, ypred::Vector{Int})
    tp = sum((ytrue .== 1) .& (ypred .== 1))
    tn = sum((ytrue .== 0) .& (ypred .== 0))
    fp = sum((ytrue .== 0) .& (ypred .== 1))
    fn = sum((ytrue .== 1) .& (ypred .== 0))
    precision = tp + fp > 0 ? tp / (tp + fp) : 0.0
    recall = tp + fn > 0 ? tp / (tp + fn) : 0.0
    f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0
    println("$name Confusion Matrix:")
    println("                 Predicted")
    println("                 Normal  Anomaly")
    println("True Normal     $(lpad(tn, 6))  $(lpad(fp, 6))")
    println("True Anomaly    $(lpad(fn, 6))  $(lpad(tp, 6))")
    println("Precision=$(round(precision, digits=4)) Recall=$(round(recall, digits=4)) F1=$(round(f1, digits=4))")
end

println("\n=== ANALYSIS 1: RAW DATA ===")
res_raw = wicmad(Y, t;
    n_iter=1500,
    burn=500,
    thin=1,
    wf=selected_wf,
    diagnostics=true,
    bootstrap_runs=0,
    z_init=z_true,
)
z_final_raw = vec(res_raw.Z[end, :])
ari_raw = WICMAD.adj_rand_index(z_final_raw, z_true)
println("Final clusters (raw): ", length(unique(z_final_raw)))
println("ARI (raw): ", round(ari_raw, digits=4))

function plot_series_by_labels(Xm::Matrix{Float64}, labels::Vector{Int}; title_str::String, filepath::String)
    p = plot(title=title_str, xlabel="Time", ylabel="Value", legend=false, size=(1000, 600))
    for i in 1:size(Xm,1)
        color = labels[i] == 0 ? :blue : :red
        plot!(p, Xm[i, :], color=color, alpha=0.5, linewidth=1)
    end
    savefig(p, filepath)
    println("Saved: ", filepath)
end

plot_series_by_labels(X, gt_binary; title_str="RAW - Ground Truth (blue=normal, red=anomaly)", filepath=joinpath(plots_dir, "raw_ground_truth.png"))
pred_bin_raw_map = to_binary_labels(z_final_raw)
plot_series_by_labels(X, pred_bin_raw_map; title_str="RAW - MAP (last sample) prediction", filepath=joinpath(plots_dir, "raw_after_map.png"))

function dahl_consensus_from_Z(Z::Matrix{Int})
    N = size(Z, 2)
    B = size(Z, 1)
    A = zeros(Float64, N, N)
    for b in 1:B
        zb = view(Z, b, :)
        for i in 1:N
            @inbounds for j in i:N
                v = (zb[i] == zb[j]) ? 1.0 : 0.0
                A[i, j] += v
                if j != i
                    A[j, i] += v
                end
            end
        end
    end
    A ./= B
    best_b, best_loss = 1, Inf
    for b in 1:B
        zb = view(Z, b, :)
        loss = 0.0
        for i in 1:N
            @inbounds for j in 1:N
                s = (zb[i] == zb[j]) ? 1.0 : 0.0
                d = s - A[i, j]
                loss += d * d
            end
        end
        if loss < best_loss
            best_loss = loss
            best_b = b
        end
    end
    return vec(view(Z, best_b, :))
end

function find_best_f1_partition(Z::Matrix{Int}, ytrue_bin::Vector{Int})
    best_f1 = -Inf
    best_z = vec(view(Z, 1, :))
    for b in 1:size(Z, 1)
        zb = vec(view(Z, b, :))
        ypred = to_binary_labels(zb)
        tp = sum((ytrue_bin .== 1) .& (ypred .== 1))
        fp = sum((ytrue_bin .== 0) .& (ypred .== 1))
        fn = sum((ytrue_bin .== 1) .& (ypred .== 0))
        precision = tp + fp > 0 ? tp / (tp + fp) : 0.0
        recall = tp + fn > 0 ? tp / (tp + fn) : 0.0
        f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0
        if f1 > best_f1
            best_f1 = f1
            best_z = zb
        end
    end
    return best_z, best_f1
end

z_dahl_raw = dahl_consensus_from_Z(res_raw.Z)
z_f1_raw, best_f1_raw = find_best_f1_partition(res_raw.Z, gt_binary)
pred_bin_raw_dahl = to_binary_labels(z_dahl_raw)
pred_bin_raw_f1 = to_binary_labels(z_f1_raw)
plot_series_by_labels(X, pred_bin_raw_dahl; title_str="RAW - Dahl consensus prediction", filepath=joinpath(plots_dir, "raw_after_dahl.png"))
plot_series_by_labels(X, pred_bin_raw_f1; title_str="RAW - F1-max prediction", filepath=joinpath(plots_dir, "raw_after_f1.png"))

print_confusion("RAW (MAP)", gt_binary, pred_bin_raw_map)
print_confusion("RAW (Dahl)", gt_binary, pred_bin_raw_dahl)
print_confusion("RAW (F1-max)", gt_binary, pred_bin_raw_f1)

function finite_diff_pad(v::AbstractVector{<:Real})
    d = diff(v)
    out = similar(v, length(v))
    out[1:end-1] = d
    out[end] = d[end]
    return out
end

function build_derivatives_matrix(Xm::AbstractMatrix{<:Real}; orders::Vector{Int})
    N, P = size(Xm)
    Ymulti = Vector{Matrix{Float64}}(undef, N)
    for i in 1:N
        raw = Float64.(Xm[i, :])
        cols = Vector{Vector{Float64}}()
        push!(cols, raw)
        dprev = raw
        for ord in orders
            dprev = finite_diff_pad(dprev)
            push!(cols, dprev)
        end
        Yi = hcat(cols...)
        Ymulti[i] = Yi
    end
    return Ymulti
end

println("\n=== ANALYSIS 2: RAW + FIRST DERIVATIVE ===")
Y_d1 = build_derivatives_matrix(X; orders=[1])
Y_d1_mats = Y_d1
sel_d1 = WICMAD.KernelSelection.select_wavelet(Y_d1_mats, t, revealed_idx;
    wf_candidates = nothing, J = nothing, boundary = "periodic",
    mcmc = (n_iter=3000, burnin=1000, thin=1))
wf_d1 = sel_d1.selected_wf
println("Selected wavelet (raw+d1): ", wf_d1)
res_d1 = wicmad(Y_d1, t;
    n_iter=1500,
    burn=500,
    thin=1,
    wf=wf_d1,
    diagnostics=true,
    bootstrap_runs=0,
    z_init=z_true,
)
z_final_d1 = vec(res_d1.Z[end, :])
ari_d1 = WICMAD.adj_rand_index(z_final_d1, z_true)
println("Final clusters (raw+d1): ", length(unique(z_final_d1)))
println("ARI (raw+d1): ", round(ari_d1, digits=4))

function plot_multi_channel_stacked(Ymulti::Vector{Matrix{Float64}}, labels::Vector{Int}; title_str::String, filepath::String)
    N = length(Ymulti)
    M = size(Ymulti[1], 2)
    plt = plot(layout=(M,1), size=(900, 350*M))
    titles = ["Raw"; ["Derivative $(k)" for k in 1:(M-1)]...]
    for c in 1:M
        for i in 1:N
            color = labels[i] == 0 ? :blue : :red
            plot!(plt[c], view(Ymulti[i], :, c), color=color, alpha=0.5, linewidth=1, label=false)
        end
        plot!(plt[c], title=(title_str * " - " * titles[c]), xlabel="Time", ylabel="Value")
    end
    savefig(plt, filepath)
    println("Saved: ", filepath)
end

plot_multi_channel_stacked(Y_d1, gt_binary; title_str="RAW+D1 - Ground Truth", filepath=joinpath(plots_dir, "raw_d1_ground_truth.png"))
pred_bin_d1_map = to_binary_labels(z_final_d1)
plot_multi_channel_stacked(Y_d1, pred_bin_d1_map; title_str="RAW+D1 - MAP (last sample) prediction", filepath=joinpath(plots_dir, "raw_d1_after_map.png"))
z_dahl_d1 = dahl_consensus_from_Z(res_d1.Z)
z_f1_d1, best_f1_d1 = find_best_f1_partition(res_d1.Z, gt_binary)
pred_bin_d1_dahl = to_binary_labels(z_dahl_d1)
pred_bin_d1_f1 = to_binary_labels(z_f1_d1)
plot_multi_channel_stacked(Y_d1, pred_bin_d1_dahl; title_str="RAW+D1 - Dahl consensus prediction", filepath=joinpath(plots_dir, "raw_d1_after_dahl.png"))
plot_multi_channel_stacked(Y_d1, pred_bin_d1_f1; title_str="RAW+D1 - F1-max prediction", filepath=joinpath(plots_dir, "raw_d1_after_f1.png"))
print_confusion("RAW+D1 (MAP)", gt_binary, pred_bin_d1_map)
print_confusion("RAW+D1 (Dahl)", gt_binary, pred_bin_d1_dahl)
print_confusion("RAW+D1 (F1-max)", gt_binary, pred_bin_d1_f1)

println("\n=== ANALYSIS 3: RAW + FIRST + SECOND DERIVATIVES ===")
Y_d12 = build_derivatives_matrix(X; orders=[1, 2])
sel_d12 = WICMAD.KernelSelection.select_wavelet(Y_d12, t, revealed_idx;
    wf_candidates = nothing, J = nothing, boundary = "periodic",
    mcmc = (n_iter=3000, burnin=1000, thin=1))
wf_d12 = sel_d12.selected_wf
println("Selected wavelet (raw+d1+d2): ", wf_d12)
res_d12 = wicmad(Y_d12, t;
    n_iter=1500,
    burn=500,
    thin=1,
    wf=wf_d12,
    diagnostics=true,
    bootstrap_runs=0,
    z_init=z_true,
)
z_final_d12 = vec(res_d12.Z[end, :])
ari_d12 = WICMAD.adj_rand_index(z_final_d12, z_true)
println("Final clusters (raw+d1+d2): ", length(unique(z_final_d12)))
println("ARI (raw+d1+d2): ", round(ari_d12, digits=4))
plot_multi_channel_stacked(Y_d12, gt_binary; title_str="RAW+D1+D2 - Ground Truth", filepath=joinpath(plots_dir, "raw_d12_ground_truth.png"))
pred_bin_d12_map = to_binary_labels(z_final_d12)
plot_multi_channel_stacked(Y_d12, pred_bin_d12_map; title_str="RAW+D1+D2 - MAP (last sample) prediction", filepath=joinpath(plots_dir, "raw_d12_after_map.png"))
z_dahl_d12 = dahl_consensus_from_Z(res_d12.Z)
z_f1_d12, best_f1_d12 = find_best_f1_partition(res_d12.Z, gt_binary)
pred_bin_d12_dahl = to_binary_labels(z_dahl_d12)
pred_bin_d12_f1 = to_binary_labels(z_f1_d12)
plot_multi_channel_stacked(Y_d12, pred_bin_d12_dahl; title_str="RAW+D1+D2 - Dahl consensus prediction", filepath=joinpath(plots_dir, "raw_d12_after_dahl.png"))
plot_multi_channel_stacked(Y_d12, pred_bin_d12_f1; title_str="RAW+D1+D2 - F1-max prediction", filepath=joinpath(plots_dir, "raw_d12_after_f1.png"))
print_confusion("RAW+D1+D2 (MAP)", gt_binary, pred_bin_d12_map)
print_confusion("RAW+D1+D2 (Dahl)", gt_binary, pred_bin_d12_dahl)
print_confusion("RAW+D1+D2 (F1-max)", gt_binary, pred_bin_d12_f1)



