#!/usr/bin/env julia

using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=dirname(@__DIR__))

using WICMAD
using WICMAD.PostProcessing: map_from_res
using Random
using Statistics
using StatsBase: countmap, sample
using Interpolations
using Plots

Random.seed!(42)

data_dir = joinpath(dirname(@__DIR__), "data")

println("Loading AsphaltRegularity dataset (combining TRAIN + TEST)...")
ds_loaded = WICMAD.Utils.load_ucr_dataset("AsphaltRegularity/AsphaltRegularity", data_dir)

println("Original dataset: $(length(ds_loaded.series)) samples")
println("Class distribution: ", countmap(ds_loaded.labels))

# Manually select one class as normal and one as anomaly
# Choose largest class as normal, second largest as anomaly
cm = countmap(ds_loaded.labels)
classes_sorted = sort(collect(keys(cm)); by = c -> -cm[c])
normal_class = classes_sorted[1]
anomaly_class = classes_sorted[2]

println("Selected normal class: $(normal_class) ($(cm[normal_class]) cases)")
println("Selected anomaly class: $(anomaly_class) ($(cm[anomaly_class]) cases)")

normal_indices_full = findall(==(normal_class), ds_loaded.labels)
anomaly_indices_full = findall(==(anomaly_class), ds_loaded.labels)

# Target anomaly proportion
p_anom = 0.15
ratio = p_anom / (1 - p_anom) # anomalies per normal

avail_norm = length(normal_indices_full)
avail_anom = length(anomaly_indices_full)

# Max anomalies permitted by available normals to keep ~15% share
n_anom_max_by_norm = floor(Int, ratio * avail_norm)
n_anomaly_used = min(avail_anom, n_anom_max_by_norm)

# Adjust normals to match 15% exactly where possible
n_normal_used = min(avail_norm, round(Int, n_anomaly_used * (1 - p_anom) / p_anom))

used_normal_indices = sample(normal_indices_full, n_normal_used, replace=false)
used_anomaly_indices = sample(anomaly_indices_full, n_anomaly_used, replace=false)
all_used_indices = vcat(used_normal_indices, used_anomaly_indices)
shuffle!(all_used_indices)

# Subset data
Y_full = ds_loaded.series[all_used_indices]
y_full = [ds_loaded.labels[i] for i in all_used_indices]

println("Using N=$(length(Y_full)) with ~15% anomalies. Counts: ", countmap(y_full))

# Interpolate to power-of-two grid (32)
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

Y_interp, P = interpolate_to_length(Y_full; target_len=32)

# Convert to matrix format for easier manipulation (N × P)
X = zeros(length(Y_interp), P)
for i in 1:length(Y_interp)
    X[i, :] = vec(Y_interp[i])  # Extract univariate series
end

t = collect(1:P)

# Binary ground-truth labels for anomaly evaluation (normal = majority class)
gt_binary = [yi == normal_class ? 0 : 1 for yi in y_full]

# Initialize cluster labels from true labels, relabeled to 1..K
function relabel_consecutive(z::Vector)
    labs = unique(z)
    m = Dict(lab => i for (i, lab) in enumerate(labs))
    return [m[v] for v in z]
end
z_true = relabel_consecutive(y_full)

# Helpers for partitions and evaluation
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

function dahl_consensus_from_Z(Z::Matrix{Int})
    N = size(Z, 2)
    B = size(Z, 1)
    # Average coassignment matrix
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
    # Choose sample minimizing ||S_b - A||_F^2
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

function plot_series_by_labels(Xm::Matrix{Float64}, labels::Vector{Int}; title_str::String, filepath::String)
    p = plot(title=title_str, xlabel="Time", ylabel="Value", legend=false, size=(1000, 600))
    for i in 1:size(Xm,1)
        color = labels[i] == 0 ? :blue : :red
        plot!(p, Xm[i, :], color=color, alpha=0.5, linewidth=1)
    end
    savefig(p, filepath)
    println("Saved: ", filepath)
end

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

# Ensure plots directory exists
plots_dir = joinpath(dirname(@__DIR__), "plots")
isdir(plots_dir) || mkpath(plots_dir)

# Wavelet selection using 15% of normals as revealed indices
normal_indices = findall(==(normal_class), y_full)
n_revealed = max(1, round(Int, 0.15 * length(normal_indices)))
revealed_subset = sample(normal_indices, n_revealed, replace=false)
revealed_idx = sort(revealed_subset)
println("Revealing $(length(revealed_idx)) of $(length(normal_indices)) normals for wavelet selection (15%)")

# Convert to vector format for wavelet selection
Y_vec = [X[i, :] for i in 1:size(X, 1)]
Y_mats = [reshape(Float64.(Yi), :, 1) for Yi in Y_vec]

if !isempty(revealed_idx)
    println("Selecting wavelet using $(length(revealed_idx)) revealed normals...")
    sel = WICMAD.KernelSelection.select_wavelet(Y_mats, t, revealed_idx;
        wf_candidates = nothing, J = nothing, boundary = "periodic",
        mcmc = (n_iter=3000, burnin=1000, thin=1))
    selected_wf = sel.selected_wf
    println("Selected wavelet: ", selected_wf)
else
    selected_wf = "sym8"
    println("No revealed indices available; defaulting to wavelet: ", selected_wf)
end

# ANALYSIS 1: RAW DATA
println("\n=== ANALYSIS 1: RAW DATA ===")
Y_raw = [reshape(X[i, :], :, 1) for i in 1:size(X, 1)]
res_raw = wicmad(Y_raw, t;
    n_iter=5000,
    burn=2000,
    thin=1,
    wf=selected_wf,
    diagnostics=true,
    bootstrap_runs=0,
    revealed_idx=revealed_idx,
)
# Compute MAP estimate (most frequent partition across all MCMC samples)
mapr_raw = map_from_res(res_raw)
z_final_raw = mapr_raw.z_hat
ari_raw = WICMAD.adj_rand_index(z_final_raw, z_true)
println("Final clusters (raw): ", length(unique(z_final_raw)))
println("ARI (raw): ", round(ari_raw, digits=4))
println("MAP partition frequency (raw): $(mapr_raw.freq) out of $(size(res_raw.Z, 1)) samples")

# Plots before and after (raw)
plot_series_by_labels(X, gt_binary; title_str="RAW - Ground Truth (blue=normal, red=anomaly)", filepath=joinpath(plots_dir, "asphaltregularity_raw_ground_truth.png"))
pred_bin_raw_map = to_binary_labels(z_final_raw)
plot_series_by_labels(X, pred_bin_raw_map; title_str="RAW - MAP prediction", filepath=joinpath(plots_dir, "asphaltregularity_raw_after_map.png"))

# Dahl and F1-max partitions
z_dahl_raw = dahl_consensus_from_Z(res_raw.Z)
z_f1_raw, best_f1_raw = find_best_f1_partition(res_raw.Z, gt_binary)
pred_bin_raw_dahl = to_binary_labels(z_dahl_raw)
pred_bin_raw_f1 = to_binary_labels(z_f1_raw)
plot_series_by_labels(X, pred_bin_raw_dahl; title_str="RAW - Dahl consensus prediction", filepath=joinpath(plots_dir, "asphaltregularity_raw_after_dahl.png"))
plot_series_by_labels(X, pred_bin_raw_f1; title_str="RAW - F1-max prediction", filepath=joinpath(plots_dir, "asphaltregularity_raw_after_f1.png"))

# Confusion matrices
print_confusion("RAW (MAP)", gt_binary, pred_bin_raw_map)
print_confusion("RAW (Dahl)", gt_binary, pred_bin_raw_dahl)
print_confusion("RAW (F1-max)", gt_binary, pred_bin_raw_f1)

# Derivative utilities
function finite_diff_pad(v::AbstractVector{<:Real})
    d = diff(v)
    out = similar(v, length(v))
    out[1:end-1] = d
    out[end] = d[end]
    return out
end

function build_derivatives_matrix(Xm::AbstractMatrix{<:Real}; orders::Vector{Int})
    N, P = size(Xm)
    M = 1 + length(orders)
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

# ANALYSIS 2: raw + first derivative
println("\n=== ANALYSIS 2: RAW + FIRST DERIVATIVE ===")
Y_d1 = build_derivatives_matrix(X; orders=[1])
# Wavelet selection for multivariate case
sel_d1 = WICMAD.KernelSelection.select_wavelet(Y_d1, t, revealed_idx;
    wf_candidates = nothing, J = nothing, boundary = "periodic",
    mcmc = (n_iter=3000, burnin=1000, thin=1))
wf_d1 = sel_d1.selected_wf
println("Selected wavelet (raw+d1): ", wf_d1)
res_d1 = wicmad(Y_d1, t;
    n_iter=5000,
    burn=2000,
    thin=1,
    wf=wf_d1,
    diagnostics=true,
    bootstrap_runs=0,
    revealed_idx=revealed_idx,
)
# Compute MAP estimate (most frequent partition across all MCMC samples)
mapr_d1 = map_from_res(res_d1)
z_final_d1 = mapr_d1.z_hat
ari_d1 = WICMAD.adj_rand_index(z_final_d1, z_true)
println("Final clusters (raw+d1): ", length(unique(z_final_d1)))
println("ARI (raw+d1): ", round(ari_d1, digits=4))
println("MAP partition frequency (raw+d1): $(mapr_d1.freq) out of $(size(res_d1.Z, 1)) samples")

# Plots for multivariate: show raw vs derivative channels
plot_multi_channel_stacked(Y_d1, gt_binary; title_str="RAW+D1 - Ground Truth", filepath=joinpath(plots_dir, "asphaltregularity_raw_d1_ground_truth.png"))
pred_bin_d1_map = to_binary_labels(z_final_d1)
plot_multi_channel_stacked(Y_d1, pred_bin_d1_map; title_str="RAW+D1 - MAP prediction", filepath=joinpath(plots_dir, "asphaltregularity_raw_d1_after_map.png"))
z_dahl_d1 = dahl_consensus_from_Z(res_d1.Z)
z_f1_d1, best_f1_d1 = find_best_f1_partition(res_d1.Z, gt_binary)
pred_bin_d1_dahl = to_binary_labels(z_dahl_d1)
pred_bin_d1_f1 = to_binary_labels(z_f1_d1)
plot_multi_channel_stacked(Y_d1, pred_bin_d1_dahl; title_str="RAW+D1 - Dahl consensus prediction", filepath=joinpath(plots_dir, "asphaltregularity_raw_d1_after_dahl.png"))
plot_multi_channel_stacked(Y_d1, pred_bin_d1_f1; title_str="RAW+D1 - F1-max prediction", filepath=joinpath(plots_dir, "asphaltregularity_raw_d1_after_f1.png"))
print_confusion("RAW+D1 (MAP)", gt_binary, pred_bin_d1_map)
print_confusion("RAW+D1 (Dahl)", gt_binary, pred_bin_d1_dahl)
print_confusion("RAW+D1 (F1-max)", gt_binary, pred_bin_d1_f1)

# ANALYSIS 3: raw + first + second derivatives
println("\n=== ANALYSIS 3: RAW + FIRST + SECOND DERIVATIVES ===")
Y_d12 = build_derivatives_matrix(X; orders=[1, 2])
sel_d12 = WICMAD.KernelSelection.select_wavelet(Y_d12, t, revealed_idx;
    wf_candidates = nothing, J = nothing, boundary = "periodic",
    mcmc = (n_iter=3000, burnin=1000, thin=1))
wf_d12 = sel_d12.selected_wf
println("Selected wavelet (raw+d1+d2): ", wf_d12)
res_d12 = wicmad(Y_d12, t;
    n_iter=5000,
    burn=2000,
    thin=1,
    wf=wf_d12,
    diagnostics=true,
    bootstrap_runs=0,
    revealed_idx=revealed_idx,
)
# Compute MAP estimate (most frequent partition across all MCMC samples)
mapr_d12 = map_from_res(res_d12)
z_final_d12 = mapr_d12.z_hat
ari_d12 = WICMAD.adj_rand_index(z_final_d12, z_true)
println("Final clusters (raw+d1+d2): ", length(unique(z_final_d12)))
println("ARI (raw+d1+d2): ", round(ari_d12, digits=4))
println("MAP partition frequency (raw+d1+d2): $(mapr_d12.freq) out of $(size(res_d12.Z, 1)) samples")

plot_multi_channel_stacked(Y_d12, gt_binary; title_str="RAW+D1+D2 - Ground Truth", filepath=joinpath(plots_dir, "asphaltregularity_raw_d12_ground_truth.png"))
pred_bin_d12_map = to_binary_labels(z_final_d12)
plot_multi_channel_stacked(Y_d12, pred_bin_d12_map; title_str="RAW+D1+D2 - MAP prediction", filepath=joinpath(plots_dir, "asphaltregularity_raw_d12_after_map.png"))
z_dahl_d12 = dahl_consensus_from_Z(res_d12.Z)
z_f1_d12, best_f1_d12 = find_best_f1_partition(res_d12.Z, gt_binary)
pred_bin_d12_dahl = to_binary_labels(z_dahl_d12)
pred_bin_d12_f1 = to_binary_labels(z_f1_d12)
plot_multi_channel_stacked(Y_d12, pred_bin_d12_dahl; title_str="RAW+D1+D2 - Dahl consensus prediction", filepath=joinpath(plots_dir, "asphaltregularity_raw_d12_after_dahl.png"))
plot_multi_channel_stacked(Y_d12, pred_bin_d12_f1; title_str="RAW+D1+D2 - F1-max prediction", filepath=joinpath(plots_dir, "asphaltregularity_raw_d12_after_f1.png"))
print_confusion("RAW+D1+D2 (MAP)", gt_binary, pred_bin_d12_map)
print_confusion("RAW+D1+D2 (Dahl)", gt_binary, pred_bin_d12_dahl)
print_confusion("RAW+D1+D2 (F1-max)", gt_binary, pred_bin_d12_f1)

println("\nDone.")

