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

println("Loading CharacterTrajectories dataset (combining TRAIN + TEST)...")
ds_loaded = WICMAD.Utils.load_ucr_dataset("CharacterTrajectories/CharacterTrajectories", data_dir)

println("Original dataset: $(length(ds_loaded.series)) samples")
println("Class distribution: ", countmap(ds_loaded.labels))

# Manually select normal and anomaly classes
# Labels are numeric: "1" = 'a', "2" = 'b', etc. (20 characters total)
# Normal class is "1" (which corresponds to 'a'), anomaly class is another letter
normal_class = "1"  # Corresponds to 'a'
cm = countmap(ds_loaded.labels)
classes_sorted = sort(collect(keys(cm)); by = c -> -cm[c])
# Find normal class and remove it, then pick the largest remaining class as anomaly
other_classes = filter(c -> c != normal_class, classes_sorted)
anomaly_class = isempty(other_classes) ? error("No other classes found") : other_classes[1]

println("Selected normal class: $(normal_class) (corresponds to 'a', $(cm[normal_class]) cases)")
println("Selected anomaly class: $(anomaly_class) ($(cm[anomaly_class]) cases)")

normal_indices_full = findall(==(normal_class), ds_loaded.labels)
anomaly_indices_full = findall(==(anomaly_class), ds_loaded.labels)

# Target anomaly proportion: 15% anomalies, 85% normal
p_anom = 0.15
p_normal = 0.85

avail_norm = length(normal_indices_full)
avail_anom = length(anomaly_indices_full)

# Calculate how many samples we can use
# For 15% anomalies: n_anom / (n_norm + n_anom) = 0.15
# This means: n_anom = 0.15/0.85 * n_norm = (3/17) * n_norm

# Start with available normals and calculate how many anomalies we can use
n_normal_used = avail_norm  # Use all available normals
n_anomaly_needed = round(Int, n_normal_used * p_anom / p_normal)  # How many anomalies needed for 15%
n_anomaly_used = min(avail_anom, n_anomaly_needed)  # Use what's available, up to what's needed

# If we have more anomalies than needed, we might need to adjust normals
if n_anomaly_used < n_anomaly_needed
    # If we don't have enough anomalies, adjust normals to match available anomalies
    n_normal_used = min(avail_norm, round(Int, n_anomaly_used * p_normal / p_anom))
end

used_normal_indices = sample(normal_indices_full, n_normal_used, replace=false)
used_anomaly_indices = sample(anomaly_indices_full, n_anomaly_used, replace=false)
all_used_indices = vcat(used_normal_indices, used_anomaly_indices)
shuffle!(all_used_indices)

# Subset data
Y_full = ds_loaded.series[all_used_indices]
y_full = [ds_loaded.labels[i] for i in all_used_indices]

println("Using N=$(length(Y_full)) with ~15% anomalies. Counts: ", countmap(y_full))
actual_anom_pct = round(100 * count(==(anomaly_class), y_full) / length(y_full), digits=1)
println("Actual anomaly percentage: $(actual_anom_pct)%")

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

t = collect(1:P)

# Binary ground-truth labels for anomaly evaluation (normal = 0, anomaly = 1)
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

# Ensure plots directory exists
plots_dir = joinpath(dirname(@__DIR__), "plots")
isdir(plots_dir) || mkpath(plots_dir)

# Wavelet selection using 15% of normals as revealed indices
normal_indices = findall(==(normal_class), y_full)
n_revealed = max(1, round(Int, 0.15 * length(normal_indices)))
revealed_subset = sample(normal_indices, n_revealed, replace=false)
revealed_idx = sort(revealed_subset)
println("Revealing $(length(revealed_idx)) of $(length(normal_indices)) normals for wavelet selection (15%)")

if !isempty(revealed_idx)
    println("Selecting wavelet using $(length(revealed_idx)) revealed normals...")
    sel = WICMAD.KernelSelection.select_wavelet(Y_interp, t, revealed_idx;
        wf_candidates = nothing, J = nothing, boundary = "periodic",
        mcmc = (n_iter=3000, burnin=1000, thin=1))
    selected_wf = sel.selected_wf
    println("Selected wavelet: ", selected_wf)
else
    selected_wf = "sym8"
    println("No revealed indices available; defaulting to wavelet: ", selected_wf)
end

# Run WICMAD
println("\n=== Running WICMAD ===")
res = wicmad(Y_interp, t;
    n_iter=5000,
    burn=2000,
    thin=1,
    wf=selected_wf,
    diagnostics=true,
    bootstrap_runs=0,
    revealed_idx=revealed_idx,
)

# Compute MAP estimate (most frequent partition across all MCMC samples)
mapr = map_from_res(res)
z_final = mapr.z_hat
ari = WICMAD.adj_rand_index(z_final, z_true)
println("Final clusters: ", length(unique(z_final)))
println("ARI: ", round(ari, digits=4))
println("MAP partition frequency: $(mapr.freq) out of $(size(res.Z, 1)) samples")

# Plots before and after
WICMAD.Plotting.plot_dataset_before_clustering(Y_interp, gt_binary, t;
    title="CharacterTrajectories - Ground Truth (blue=normal class '$(normal_class)'='a', red=anomaly class '$(anomaly_class)')",
    save_path=joinpath(plots_dir, "charactertrajectories_ground_truth.png"))

pred_bin_map = to_binary_labels(z_final)
WICMAD.Plotting.plot_dataset_after_clustering(Y_interp, pred_bin_map, t;
    title="CharacterTrajectories - MAP prediction",
    save_path=joinpath(plots_dir, "charactertrajectories_after_map.png"))

# Dahl and F1-max partitions
z_dahl = dahl_consensus_from_Z(res.Z)
z_f1, best_f1 = find_best_f1_partition(res.Z, gt_binary)
pred_bin_dahl = to_binary_labels(z_dahl)
pred_bin_f1 = to_binary_labels(z_f1)

WICMAD.Plotting.plot_dataset_after_clustering(Y_interp, pred_bin_dahl, t;
    title="CharacterTrajectories - Dahl consensus prediction",
    save_path=joinpath(plots_dir, "charactertrajectories_after_dahl.png"))

WICMAD.Plotting.plot_dataset_after_clustering(Y_interp, pred_bin_f1, t;
    title="CharacterTrajectories - F1-max prediction",
    save_path=joinpath(plots_dir, "charactertrajectories_after_f1.png"))

# Confusion matrices
print_confusion("MAP", gt_binary, pred_bin_map)
print_confusion("Dahl", gt_binary, pred_bin_dahl)
print_confusion("F1-max", gt_binary, pred_bin_f1)

# ============================================================================
# ANALYSIS 2: Anomalies sampled uniformly from all other classes
# ============================================================================
println("\n" * "="^70)
println("ANALYSIS 2: Anomalies sampled uniformly from all other classes")
println("="^70)

# Get all anomaly classes (all classes except normal)
all_anomaly_classes = filter(c -> c != normal_class, classes_sorted)
println("Anomaly classes (uniform sampling): ", all_anomaly_classes)

# Collect all indices from all anomaly classes
all_anomaly_indices_full = Int[]
for cls in all_anomaly_classes
    append!(all_anomaly_indices_full, findall(==(cls), ds_loaded.labels))
end

println("Total anomaly samples available: $(length(all_anomaly_indices_full))")

# Calculate how many samples we can use for uniform sampling
n_normal_used_2 = avail_norm  # Use all available normals
n_anomaly_needed_2 = round(Int, n_normal_used_2 * p_anom / p_normal)
n_anomaly_used_2 = min(length(all_anomaly_indices_full), n_anomaly_needed_2)

# If we don't have enough anomalies, adjust normals
if n_anomaly_used_2 < n_anomaly_needed_2
    n_normal_used_2 = min(avail_norm, round(Int, n_anomaly_used_2 * p_normal / p_anom))
end

# Sample normals
used_normal_indices_2 = sample(normal_indices_full, n_normal_used_2, replace=false)

# Sample anomalies uniformly from all anomaly classes
# To ensure uniform sampling, we'll sample from each class proportionally
used_anomaly_indices_2 = Int[]
if n_anomaly_used_2 > 0
    # Count how many samples per anomaly class
    n_anomaly_classes = length(all_anomaly_classes)
    samples_per_class = div(n_anomaly_used_2, n_anomaly_classes)
    remainder = n_anomaly_used_2 - samples_per_class * n_anomaly_classes
    
    # Sample from each class
    for (idx, cls) in enumerate(all_anomaly_classes)
        cls_indices = findall(==(cls), ds_loaded.labels)
        n_to_sample = samples_per_class + (idx <= remainder ? 1 : 0)
        if n_to_sample > 0 && length(cls_indices) > 0
            sampled = sample(cls_indices, min(n_to_sample, length(cls_indices)), replace=false)
            append!(used_anomaly_indices_2, sampled)
        end
    end
    
    # If we still need more samples, randomly sample from remaining
    if length(used_anomaly_indices_2) < n_anomaly_used_2
        remaining_needed = n_anomaly_used_2 - length(used_anomaly_indices_2)
        remaining_indices = setdiff(all_anomaly_indices_full, used_anomaly_indices_2)
        if length(remaining_indices) > 0
            additional = sample(remaining_indices, min(remaining_needed, length(remaining_indices)), replace=false)
            append!(used_anomaly_indices_2, additional)
        end
    end
end

all_used_indices_2 = vcat(used_normal_indices_2, used_anomaly_indices_2)
shuffle!(all_used_indices_2)

# Subset data for analysis 2
Y_full_2 = ds_loaded.series[all_used_indices_2]
y_full_2 = [ds_loaded.labels[i] for i in all_used_indices_2]

println("Using N=$(length(Y_full_2)) with ~15% anomalies (uniform from all classes). Counts: ", countmap(y_full_2))
actual_anom_pct_2 = round(100 * count(!=(normal_class), y_full_2) / length(y_full_2), digits=1)
println("Actual anomaly percentage: $(actual_anom_pct_2)%")

# Interpolate
Y_interp_2, P_2 = interpolate_to_length(Y_full_2; target_len=32)
t_2 = collect(1:P_2)

# Binary ground-truth labels for anomaly evaluation (normal = 0, anomaly = 1)
gt_binary_2 = [yi == normal_class ? 0 : 1 for yi in y_full_2]

# Initialize cluster labels
z_true_2 = relabel_consecutive(y_full_2)

# Wavelet selection using 15% of normals as revealed indices
normal_indices_2 = findall(==(normal_class), y_full_2)
n_revealed_2 = max(1, round(Int, 0.15 * length(normal_indices_2)))
revealed_subset_2 = sample(normal_indices_2, n_revealed_2, replace=false)
revealed_idx_2 = sort(revealed_subset_2)
println("Revealing $(length(revealed_idx_2)) of $(length(normal_indices_2)) normals for wavelet selection (15%)")

if !isempty(revealed_idx_2)
    println("Selecting wavelet using $(length(revealed_idx_2)) revealed normals...")
    sel_2 = WICMAD.KernelSelection.select_wavelet(Y_interp_2, t_2, revealed_idx_2;
        wf_candidates = nothing, J = nothing, boundary = "periodic",
        mcmc = (n_iter=3000, burnin=1000, thin=1))
    selected_wf_2 = sel_2.selected_wf
    println("Selected wavelet: ", selected_wf_2)
else
    selected_wf_2 = "sym8"
    println("No revealed indices available; defaulting to wavelet: ", selected_wf_2)
end

# Run WICMAD
println("\n=== Running WICMAD (Analysis 2: Uniform Anomalies) ===")
res_2 = wicmad(Y_interp_2, t_2;
    n_iter=5000,
    burn=2000,
    thin=1,
    wf=selected_wf_2,
    diagnostics=true,
    bootstrap_runs=0,
    revealed_idx=revealed_idx_2,
)

# Compute MAP estimate (most frequent partition across all MCMC samples)
mapr_2 = map_from_res(res_2)
z_final_2 = mapr_2.z_hat
ari_2 = WICMAD.adj_rand_index(z_final_2, z_true_2)
println("Final clusters: ", length(unique(z_final_2)))
println("ARI: ", round(ari_2, digits=4))
println("MAP partition frequency: $(mapr_2.freq) out of $(size(res_2.Z, 1)) samples")

# Plots before and after
WICMAD.Plotting.plot_dataset_before_clustering(Y_interp_2, gt_binary_2, t_2;
    title="CharacterTrajectories (Uniform Anomalies) - Ground Truth (blue=normal class '$(normal_class)'='a', red=anomalies from all other classes)",
    save_path=joinpath(plots_dir, "charactertrajectories_uniform_ground_truth.png"))

pred_bin_map_2 = to_binary_labels(z_final_2)
WICMAD.Plotting.plot_dataset_after_clustering(Y_interp_2, pred_bin_map_2, t_2;
    title="CharacterTrajectories (Uniform Anomalies) - MAP prediction",
    save_path=joinpath(plots_dir, "charactertrajectories_uniform_after_map.png"))

# Dahl and F1-max partitions
z_dahl_2 = dahl_consensus_from_Z(res_2.Z)
z_f1_2, best_f1_2 = find_best_f1_partition(res_2.Z, gt_binary_2)
pred_bin_dahl_2 = to_binary_labels(z_dahl_2)
pred_bin_f1_2 = to_binary_labels(z_f1_2)

WICMAD.Plotting.plot_dataset_after_clustering(Y_interp_2, pred_bin_dahl_2, t_2;
    title="CharacterTrajectories (Uniform Anomalies) - Dahl consensus prediction",
    save_path=joinpath(plots_dir, "charactertrajectories_uniform_after_dahl.png"))

WICMAD.Plotting.plot_dataset_after_clustering(Y_interp_2, pred_bin_f1_2, t_2;
    title="CharacterTrajectories (Uniform Anomalies) - F1-max prediction",
    save_path=joinpath(plots_dir, "charactertrajectories_uniform_after_f1.png"))

# Confusion matrices
print_confusion("Uniform Anomalies (MAP)", gt_binary_2, pred_bin_map_2)
print_confusion("Uniform Anomalies (Dahl)", gt_binary_2, pred_bin_dahl_2)
print_confusion("Uniform Anomalies (F1-max)", gt_binary_2, pred_bin_f1_2)

println("\nDone.")

