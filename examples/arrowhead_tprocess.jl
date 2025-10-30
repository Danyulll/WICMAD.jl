#!/usr/bin/env julia

# ------------------------------------------------------------
# WICMAD ArrowHead Dataset - T-Process on Original Data
#
# Runs a single WICMAD analysis on the combined ArrowHead dataset
# using the original (raw) data only, with a Student-t process
# residual model. 15% of the normal samples are revealed.
# ------------------------------------------------------------

using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=dirname(@__DIR__))

using WICMAD
using StatsBase: sample, countmap
using Random
using Statistics: mean
using Interpolations

Random.seed!(42)

function load_arrowhead_data(filepath)
    lines = readlines(filepath)
    data_rows = []
    for (i, line) in enumerate(lines)
        s = strip(line)
        isempty(s) && continue
        parts = filter(!isempty, split(s, ' '))
        if !isempty(parts)
            try
                row = parse.(Float64, parts)
                push!(data_rows, row)
            catch
                @warn "Failed to parse line" i line
            end
        end
    end
    isempty(data_rows) && error("No data parsed from $(filepath)")
    data = hcat(data_rows...)'
    labels = Int.(data[:, 1])
    X = data[:, 2:end]
    return X, labels
end

# Paths relative to repository root
train_path = joinpath(@__DIR__, "..", "data", "ArrowHead", "ArrowHead_TRAIN.txt")
test_path  = joinpath(@__DIR__, "..", "data", "ArrowHead", "ArrowHead_TEST.txt")

println("Loading ArrowHead train/test ...")
X_train, y_train = load_arrowhead_data(train_path)
X_test,  y_test  = load_arrowhead_data(test_path)

X = vcat(X_train, X_test)
y = vcat(y_train, y_test)
println("Combined shape: ", size(X))
println("Class distribution: ", countmap(y))

# Interpolate to nearest dyadic length (use 32 to match other scripts)
P_target = 32
X_interp = zeros(size(X, 1), P_target)
for i in 1:size(X, 1)
    itp = linear_interpolation(collect(1:size(X, 2)), X[i, :])
    new_t = collect(range(1, size(X, 2), length=P_target))
    X_interp[i, :] = itp.(new_t)
end
X = X_interp
println("Interpolated to: ", size(X))

# Subset to ensure ~15% anomalies in the dataset (normal = class 0)
normal_class = 0
normal_idx_full = findall(==(normal_class), y)
anomaly_idx_full = findall(!=(normal_class), y)

N_full = length(y)
n_anom_target = round(Int, 0.15 * N_full)
n_anom_used = min(length(anomaly_idx_full), n_anom_target)
n_norm_used = min(length(normal_idx_full), N_full - n_anom_used)

used_normals = sample(normal_idx_full, n_norm_used, replace=false)
used_anoms  = sample(anomaly_idx_full, n_anom_used, replace=false)
keep_idx = vcat(used_normals, used_anoms)
shuffle!(keep_idx)

X = X[keep_idx, :]
y = y[keep_idx]
println("After subsetting: N=$(size(X,1)) with ~15% anomalies. Counts: ", countmap(y))

# Choose normal class and reveal 15% of normals (after subsetting)
normal_indices = findall(==(normal_class), y)
anomaly_indices = findall(!=(normal_class), y)
println("Normals: $(length(normal_indices)) | Anomalies: $(length(anomaly_indices))")

num_reveal = max(1, round(Int, 0.15 * length(normal_indices)))
revealed_normal_subset = sample(normal_indices, min(num_reveal, length(normal_indices)), replace=false)

# Build inputs for WICMAD (raw data only)
Y = [X[i, :] for i in 1:size(X, 1)]
t = collect(1:size(X, 2))

# Helpers and ground-truth binary labels (normal=0, anomaly=1)
gt_binary = [yi == normal_class ? 0 : 1 for yi in y]

if !isdefined(Main, :largest_cluster)
    largest_cluster(z) = begin
        counts = countmap(z)
        keys_sorted = sort(collect(keys(counts)); by = k -> -counts[k])
        keys_sorted[1]
    end
end

if !isdefined(Main, :to_binary_labels)
    to_binary_labels(z) = begin
        k_norm = largest_cluster(z)
        [zi == k_norm ? 0 : 1 for zi in z]
    end
end

if !isdefined(Main, :print_confusion)
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
end

if !isdefined(Main, :dahl_consensus_from_Z)
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
end

if !isdefined(Main, :find_best_f1_partition)
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
end

println("Running WICMAD with t-process residuals (15% revealed normals)...")
fit = WICMAD.wicmad(Y, t;
    n_iter=3000,
    burn=1000,
    thin=1,
    alpha_prior=(10.0, 1.0),
    revealed_idx=revealed_normal_subset,
    warmup_iters=500,
    diagnostics=true,
    wf="sym8",
    # t-process options
    process=:tprocess,
    nu_df=5.0,
    learn_nu=false,
    # no bootstrap for single run
    bootstrap_runs=0,
)

# Summary
final_z = vec(fit.Z[end, :])
K_final = length(unique(final_z))
println("\nMCMC completed.")
println("Final clusters: ", K_final)
println("ν (t-process df): ", fit.process.nu_df)

# Confusion matrices (MAP, Dahl, F1-max)
pred_bin_map = to_binary_labels(final_z)
z_dahl = dahl_consensus_from_Z(fit.Z)
pred_bin_dahl = to_binary_labels(z_dahl)
z_f1, best_f1 = find_best_f1_partition(fit.Z, gt_binary)
pred_bin_f1 = to_binary_labels(z_f1)
print_confusion("RAW (MAP)", gt_binary, pred_bin_map)
print_confusion("RAW (Dahl)", gt_binary, pred_bin_dahl)
print_confusion("RAW (F1-max)", gt_binary, pred_bin_f1)

# Simple ARI proxy against revealed normals as a sanity check (largest cluster = normal)
cluster_counts = countmap(final_z)
majority_cluster = argmax(cluster_counts)
predicted_labels = [z == majority_cluster ? 0 : 1 for z in final_z]
revealed_mask = zeros(Bool, length(y))
revealed_mask[revealed_normal_subset] .= true
revealed_truth = [revealed_mask[i] ? 0 : -1 for i in 1:length(y)]  # -1 for unknown
revealed_pred = [revealed_mask[i] ? predicted_labels[i] : -1 for i in 1:length(y)]
acc = sum((revealed_truth .== 0) .& (revealed_pred .== 0)) / max(1, sum(revealed_truth .== 0))
println("Revealed-normal accuracy (sanity check): ", round(acc, digits=4))

# ------------------------------------------------------------
# Derivative-augmented (multivariate) dataset run
# ------------------------------------------------------------

println("\nPreparing derivative-augmented dataset (raw + derivative)...")

function compute_derivatives(Xin)
    n_samples, n_points = size(Xin)
    deriv = zeros(n_samples, n_points - 1)
    for i in 1:n_samples
        deriv[i, :] = diff(Xin[i, :])
    end
    deriv_padded = zeros(n_samples, n_points)
    deriv_padded[:, 1:end-1] = deriv
    deriv_padded[:, end] = deriv[:, end]
    Xmv = zeros(n_samples, n_points, 2)
    Xmv[:, :, 1] = Xin
    Xmv[:, :, 2] = deriv_padded
    return Xmv
end

X_mv = compute_derivatives(X)
println("Multivariate shape: ", size(X_mv))

# Convert to vector-of-matrices expected by WICMAD
Y_mv = [X_mv[i, :, :] for i in 1:size(X_mv, 1)]
t_mv = collect(1:size(X_mv, 2))

println("Running WICMAD (t-process) on multivariate data (raw + derivative)...")
fit_mv = WICMAD.wicmad(Y_mv, t_mv;
    n_iter=3000,
    burn=1000,
    thin=1,
    alpha_prior=(10.0, 1.0),
    revealed_idx=revealed_normal_subset,
    warmup_iters=500,
    diagnostics=true,
    wf="sym8",
    process=:tprocess,
    nu_df=5.0,
    learn_nu=false,
    bootstrap_runs=0,
)

final_z_mv = vec(fit_mv.Z[end, :])
K_final_mv = length(unique(final_z_mv))
println("Final clusters (multivariate): ", K_final_mv)
println("ν (t-process df, multivariate): ", fit_mv.process.nu_df)

# Confusion matrices for multivariate run
pred_bin_mv_map = to_binary_labels(final_z_mv)
z_dahl_mv = dahl_consensus_from_Z(fit_mv.Z)
pred_bin_mv_dahl = to_binary_labels(z_dahl_mv)
z_f1_mv, best_f1_mv = find_best_f1_partition(fit_mv.Z, gt_binary)
pred_bin_mv_f1 = to_binary_labels(z_f1_mv)
print_confusion("RAW+DERIV (MAP)", gt_binary, pred_bin_mv_map)
print_confusion("RAW+DERIV (Dahl)", gt_binary, pred_bin_mv_dahl)
print_confusion("RAW+DERIV (F1-max)", gt_binary, pred_bin_mv_f1)


