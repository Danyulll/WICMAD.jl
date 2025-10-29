#!/usr/bin/env julia

# ------------------------------------------------------------
# WICMAD ArrowHead Dataset - Dual Analysis (Raw vs Raw+Derivative)
# 
# Performs two separate WICMAD analyses on combined ArrowHead data:
# 1. Raw data only analysis
# 2. Raw + derivative data analysis
# 
# Dataset: Combined training + test datasets
# Anomaly Setup: Class 0 (normal) vs Class 1 (anomalies) at 15% anomaly rate
# Uses Conservative-Balanced initialization with comprehensive plotting 
# and MCMC diagnostics for both analyses
# ------------------------------------------------------------

using Pkg
Pkg.activate(@__DIR__)

# Add the WICMAD package to the environment
Pkg.develop(path=dirname(@__DIR__))

using WICMAD
using Plots
using StatsBase: sample, countmap, autocor
using Random
using Statistics: mean, std, median
using DataFrames
using CSV
using Interpolations

Random.seed!(42)

# Function to compute derivatives of time series data
function compute_derivatives(X)
    """
    Compute derivatives of time series data using finite differences.
    Returns multivariate functional dataset with 2 channels: [raw_data, derivatives].
    Each observation is now a 2D functional curve.
    """
    n_samples, n_points = size(X)
    
    # Compute derivatives using finite differences
    derivatives = zeros(n_samples, n_points - 1)
    for i in 1:n_samples
        derivatives[i, :] = diff(X[i, :])
    end
    
    # Pad derivatives to match original length by repeating last value
    derivatives_padded = zeros(n_samples, n_points)
    derivatives_padded[:, 1:end-1] = derivatives
    derivatives_padded[:, end] = derivatives[:, end]  # Repeat last derivative value
    
    # Create multivariate functional dataset: each sample is now a 2D curve
    # Shape: (n_samples, n_points, 2) where 2 is the number of channels
    X_multivariate = zeros(n_samples, n_points, 2)
    X_multivariate[:, :, 1] = X  # First channel: raw data
    X_multivariate[:, :, 2] = derivatives_padded  # Second channel: derivatives
    
    return X_multivariate
end

# Function to find partition with maximum F1 score
function find_best_f1_partition(Z_matrix, revealed_labels)
    max_f1 = 0.0
    best_partition = nothing
    best_f1_partition = nothing
    
    for i in 1:size(Z_matrix, 1)
        # Get cluster assignments for this sample
        sample_assignments = Z_matrix[i, :]
        
        # Convert to binary labels (anomaly detection)
        # Use the largest cluster as normal, others as anomalies
        cluster_counts = countmap(sample_assignments)
        if !isempty(cluster_counts)
            largest_cluster = argmax(cluster_counts)
            predicted_labels_sample = [assign == largest_cluster ? 0 : 1 for assign in sample_assignments]
            
            # Calculate F1 score
            tp = sum((revealed_labels .== 1) .& (predicted_labels_sample .== 1))
            fp = sum((revealed_labels .== 0) .& (predicted_labels_sample .== 1))
            fn = sum((revealed_labels .== 1) .& (predicted_labels_sample .== 0))
            
            precision = tp > 0 ? tp / (tp + fp) : 0.0
            recall = tp > 0 ? tp / (tp + fn) : 0.0
            f1_score = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0.0
            
            if f1_score > max_f1
                max_f1 = f1_score
                best_partition = sample_assignments
                best_f1_partition = predicted_labels_sample
            end
        end
    end
    
    return max_f1, best_partition, best_f1_partition
end

# Function to compute MAP estimate
function compute_map_estimate(Z_matrix, loglik_values)
    map_loglik = -Inf
    map_partition = nothing
    
    for i in 1:size(Z_matrix, 1)
        # Get cluster assignments for this sample
        z_sample = Z_matrix[i, :]
        
        # Use log-likelihood as proxy for log posterior probability
        # P(z | Y) ∝ P(Y | z) × P(z)
        # We'll use just the likelihood for MAP estimation
        log_posterior = loglik_values[i]
        
        if log_posterior > map_loglik
            map_loglik = log_posterior
            map_partition = z_sample
        end
    end
    
    return map_partition, map_loglik
end

# Load ArrowHead dataset from text file
function load_arrowhead_data(filepath)
    println("Loading ArrowHead dataset from: $filepath")
    
    # Read the file line by line to handle any formatting issues
    lines = readlines(filepath)
    println("Number of lines: $(length(lines))")
    
    # Parse each line manually
    data_rows = []
    for (i, line) in enumerate(lines)
        if isempty(strip(line))
            continue
        end
        
        # Split by spaces and filter out empty strings
        parts = filter(!isempty, split(strip(line), ' '))
        
        if length(parts) > 0
            try
                # Convert to Float64
                row = parse.(Float64, parts)
                push!(data_rows, row)
            catch e
                println("Error parsing line $i: $e")
                println("Line content: $line")
                continue
            end
        end
    end
    
    println("Successfully parsed $(length(data_rows)) rows")
    
    if length(data_rows) == 0
        error("No data was successfully parsed!")
    end
    
    # Convert to matrix
    data = hcat(data_rows...)'
    println("Data shape: $(size(data))")
    
    # Extract labels and time series
    labels = Int.(data[:, 1])
    time_series = data[:, 2:end]
    
    println("Class distribution:")
    for class in 0:2
        count_class = count(==(class), labels)
        println("Class $class: $count_class")
    end
    
    println("Time series shape: $(size(time_series))")
    println("Time series range: [$(minimum(time_series)), $(maximum(time_series))]")
    
    return time_series, labels
end

# Load and combine training and test datasets
train_path = "../data/ArrowHead/ArrowHead_TRAIN.txt"
test_path = "../data/ArrowHead/ArrowHead_TEST.txt"

println("Loading training dataset...")
X_train, labels_train = load_arrowhead_data(train_path)

println("Loading test dataset...")
X_test, labels_test = load_arrowhead_data(test_path)

# Combine datasets
X = vcat(X_train, X_test)
true_labels = vcat(labels_train, labels_test)

println("\nCombined dataset:")
println("Training samples: $(size(X_train, 1))")
println("Test samples: $(size(X_test, 1))")
println("Total samples: $(size(X, 1))")
println("Combined class distribution: $(countmap(true_labels))")

println("\n" * "="^60)
println("ARROWHEAD DATASET LOADED SUCCESSFULLY")
println("="^60)
println("Dataset shape: $(size(X))")
println("Number of classes: $(length(unique(true_labels)))")
println("Class distribution: $(countmap(true_labels))")

# Create plots directory structure
plots_dir = "plots/arrowhead"
raw_plots_dir = "plots/arrowhead/raw_analysis"
multivariate_plots_dir = "plots/arrowhead/multivariate_analysis"

for dir in [plots_dir, raw_plots_dir, multivariate_plots_dir]
    if !isdir(dir)
        mkpath(dir)
    end
end

# Interpolate data down to 32 time points (power of 2)
println("\n" * "="^50)
println("INTERPOLATING DATA TO 32 TIME POINTS")
println("="^50)

# Store original data before interpolation
X_original = copy(X)

n_original = size(X, 2)
n_target = 32

println("Original time points: $n_original")
println("Target time points: $n_target")

# Create interpolation for each time series
X_interpolated = zeros(size(X, 1), n_target)

for i in 1:size(X, 1)
    # Create interpolation function for this time series
    itp = linear_interpolation(collect(1:n_original), X[i, :])
    
    # Interpolate to new time points
    new_time_points = collect(range(1, n_original, length=n_target))
    X_interpolated[i, :] = itp.(new_time_points)
end

# Replace original data with interpolated data
X = X_interpolated

println("Data interpolated successfully!")
println("New dataset shape: $(size(X))")
println("Time series range: [$(minimum(X)), $(maximum(X))]")

# Conservative-Balanced configuration with higher alpha prior for more clusters
conservative_balanced_config = Dict(
    :name => "Conservative-Balanced-HighAlpha",
    :alpha_prior => (15.0, 1.0),
    :a_eta => 2.8,
    :b_eta => 0.06,
    :a_sig => 2.8,
    :b_sig => 0.015
)

println("\n" * "="^60)
println("CONSERVATIVE-BALANCED CONFIGURATION")
println("="^60)
println("Configuration: $(conservative_balanced_config[:name])")
println("Alpha prior: $(conservative_balanced_config[:alpha_prior])")
println("Eta prior: (a=$(conservative_balanced_config[:a_eta]), b=$(conservative_balanced_config[:b_eta]))")
println("Sigma prior: (a=$(conservative_balanced_config[:a_sig]), b=$(conservative_balanced_config[:b_sig]))")

# Set up anomaly detection scenario - simplified approach
println("\n" * "="^50)
println("SETTING UP ANOMALY DETECTION SCENARIO")
println("="^50)

# Use class 0 as normal, class 2 as anomalies
normal_class = 0
anomaly_class = 2

normal_indices = findall(==(normal_class), true_labels)
anomaly_indices = findall(==(anomaly_class), true_labels)

println("Normal samples (class $normal_class): $(length(normal_indices))")
println("Anomaly samples (class $anomaly_class): $(length(anomaly_indices))")

# Calculate how many anomalies we need for exactly 15% of total data
total_samples = length(true_labels)
n_anomaly_target = round(Int, 0.15 * total_samples)
n_normal_target = total_samples - n_anomaly_target

println("Target: $(n_anomaly_target) anomalies (15% of $total_samples total samples)")
println("Target: $(n_normal_target) normal samples (85% of $total_samples total samples)")

# Ensure we have exactly 15% anomalies by adjusting the total dataset size if needed
if length(anomaly_indices) < n_anomaly_target
    println("Warning: Only $(length(anomaly_indices)) anomaly samples available, but need $(n_anomaly_target)")
    println("Adjusting target to use all available anomaly samples")
    n_anomaly_used = length(anomaly_indices)
    n_normal_used = min(length(normal_indices), round(Int, n_anomaly_used / 0.15 * 0.85))
elseif length(normal_indices) < n_normal_target
    println("Warning: Only $(length(normal_indices)) normal samples available, but need $(n_normal_target)")
    println("Adjusting target to use all available normal samples")
    n_normal_used = length(normal_indices)
    n_anomaly_used = min(length(anomaly_indices), round(Int, n_normal_used / 0.85 * 0.15))
else
    n_anomaly_used = n_anomaly_target
    n_normal_used = n_normal_target
end

println("Using $(n_normal_used) normal samples (available: $(length(normal_indices)))")
println("Using $(n_anomaly_used) anomaly samples (available: $(length(anomaly_indices)))")

# Sample indices randomly
used_normal_indices = sample(normal_indices, n_normal_used, replace=false)
used_anomaly_indices = sample(anomaly_indices, n_anomaly_used, replace=false)

# Combine all used indices
all_used_indices = vcat(used_normal_indices, used_anomaly_indices)
shuffle!(all_used_indices)

# Create revealed labels (1 for anomalies, 0 for normal)
revealed_labels = zeros(Int, length(all_used_indices))
for (i, idx) in enumerate(all_used_indices)
    if idx in used_anomaly_indices
        revealed_labels[i] = 1  # Anomaly
    else
        revealed_labels[i] = 0  # Normal
    end
end

println("Total samples used: $(length(all_used_indices))")
actual_anomaly_percentage = round(100 * n_anomaly_used / length(all_used_indices), digits=1)
println("Actual anomaly percentage: $(actual_anomaly_percentage)%")
println("Revealed labels distribution: $(countmap(revealed_labels))")

# Verify we achieved the target percentage
if abs(actual_anomaly_percentage - 15.0) < 0.1
    println("✓ Successfully achieved ~15% anomaly rate")
else
    println("⚠ Warning: Anomaly rate is $(actual_anomaly_percentage)%, not exactly 15%")
end

# For WICMAD, we need to specify which indices are revealed (normal samples)
# We'll reveal 25% of the normal samples to make the task more challenging
n_revealed_normal = max(1, round(Int, 0.25 * n_normal_used))  # At least 1, but 25% of normal samples
revealed_normal_subset = sample(used_normal_indices, n_revealed_normal, replace=false)
revealed_normal_indices = findall(idx -> idx in revealed_normal_subset, all_used_indices)

println("Revealing $(n_revealed_normal) out of $(n_normal_used) normal samples ($(round(100 * n_revealed_normal / n_normal_used, digits=1))%)")
println("Revealed normal indices: $(length(revealed_normal_indices))")

println("\n" * "="^50)
println("VERIFYING DATA WITH EARLY PLOTS")
println("="^50)

println("Creating comprehensive data verification plots...")

# Plot 1: All classes with different colors
p1 = plot(title="ArrowHead Dataset - Combined Training+Test (All Classes)", 
          xlabel="Time", ylabel="Value", legend=:topright, size=(800, 600))

for class in 0:2
    class_indices = findall(==(class), true_labels)
    if length(class_indices) > 0
        for idx in class_indices
            color = class == 0 ? :blue : (class == 1 ? :green : :red)
            plot!(p1, X[idx, :], color=color, alpha=0.7, 
                  label=class == 0 ? "Class 0" : (class == 1 ? "Class 1" : "Class 2"))
        end
    end
end

savefig(p1, joinpath(plots_dir, "data_verification_all_classes_combined.png"))
println("Saved: $(joinpath(plots_dir, "data_verification_all_classes_combined.png"))")

# Plot 2: Class means
p2 = plot(title="ArrowHead Dataset - Combined Training+Test (Class Means)", 
          xlabel="Time", ylabel="Value", legend=:topright, size=(800, 600))

for class in 0:2
    class_indices = findall(==(class), true_labels)
    if length(class_indices) > 0
        class_mean = mean(X[class_indices, :], dims=1)[:]
        color = class == 0 ? :blue : (class == 1 ? :green : :red)
        plot!(p2, class_mean, color=color, linewidth=3, 
              label="Class $class Mean")
    end
end

savefig(p2, joinpath(plots_dir, "data_verification_class_means_combined.png"))
println("Saved: $(joinpath(plots_dir, "data_verification_class_means_combined.png"))")

# Plot 3: Individual samples from each class
p3 = plot(layout=(1, 3), size=(1200, 400))

for (plot_idx, class) in enumerate(0:2)
    class_indices = findall(==(class), true_labels)
    if length(class_indices) > 0
        # Plot first 4 samples from this class
        for i in 1:min(4, length(class_indices))
            idx = class_indices[i]
            color = i == 1 ? :blue : (i == 2 ? :red : (i == 3 ? :green : :orange))
            plot!(p3[plot_idx], X[idx, :], color=color, alpha=0.8,
                  label=i == 1 ? "Sample $i" : "")
        end
        plot!(p3[plot_idx], title="Class $class Samples", xlabel="Time", ylabel="Value")
    end
end

savefig(p3, joinpath(plots_dir, "data_verification_class_samples_combined.png"))
println("Saved: $(joinpath(plots_dir, "data_verification_class_samples_combined.png"))")

# Plot 4: Data statistics
p4 = plot(layout=(2, 2), size=(800, 600))

# Plot mean and std for each class
for class in 0:2
    class_indices = findall(==(class), true_labels)
    if length(class_indices) > 0
        class_data = X[class_indices, :]
        class_mean = mean(class_data, dims=1)[:]
        class_std = std(class_data, dims=1)[:]
        
        color = class == 0 ? :blue : (class == 1 ? :green : :red)
        plot!(p4[1], class_mean, color=color, linewidth=2, 
              label="Class $class Mean")
        plot!(p4[2], class_std, color=color, linewidth=2, 
              label="Class $class Std")
    end
end

plot!(p4[1], title="Class Means", xlabel="Time", ylabel="Value")
plot!(p4[2], title="Class Standard Deviations", xlabel="Time", ylabel="Std Dev")

# Plot data range
plot!(p4[3], [minimum(X, dims=1)[:], maximum(X, dims=1)[:]], 
      title="Data Range", xlabel="Time", ylabel="Value", 
      label=["Min" "Max"], linewidth=2)

# Plot sample count per class
class_counts = [count(==(class), true_labels) for class in 0:2]
bar!(p4[4], 0:2, class_counts, title="Samples per Class", 
     xlabel="Class", ylabel="Count", label="Count")

savefig(p4, joinpath(plots_dir, "data_verification_statistics_combined.png"))
println("Saved: $(joinpath(plots_dir, "data_verification_statistics_combined.png"))")

# Plot 5: Anomaly detection setup visualization
p5 = plot(layout=(1, 2), size=(1000, 400))

# Plot normal class samples
normal_sample_indices = sample(normal_indices, min(10, length(normal_indices)), replace=false)
for idx in normal_sample_indices
    plot!(p5[1], X[idx, :], color=:blue, alpha=0.7, linewidth=1)
end
plot!(p5[1], title="Normal Class Samples (Class $normal_class)", xlabel="Time", ylabel="Value")

# Plot anomaly class samples
anomaly_sample_indices = sample(anomaly_indices, min(10, length(anomaly_indices)), replace=false)
for idx in anomaly_sample_indices
    plot!(p5[2], X[idx, :], color=:red, alpha=0.7, linewidth=1)
end
plot!(p5[2], title="Anomaly Class Samples (Class $anomaly_class)", xlabel="Time", ylabel="Value")

savefig(p5, joinpath(plots_dir, "anomaly_detection_setup.png"))
println("Saved: $(joinpath(plots_dir, "anomaly_detection_setup.png"))")

# Create comparison plot showing original vs interpolated
p_interp = plot(layout=(1, 2), size=(1000, 400))

# Plot original data (first sample from combined dataset before interpolation)
# We need to reload the original data for this comparison
X_original = vcat(X_train, X_test)  # Reload original combined data
original_sample = X_original[1, :]
plot!(p_interp[1], original_sample, title="Original Combined Data (251 points)", 
      xlabel="Time", ylabel="Value", color=:blue, linewidth=2)

# Plot interpolated data (first sample)
plot!(p_interp[2], X[1, :], title="Interpolated Combined Data (32 points)", 
      xlabel="Time", ylabel="Value", color=:red, linewidth=2)

savefig(p_interp, joinpath(plots_dir, "data_interpolation_comparison_combined.png"))
println("Saved: $(joinpath(plots_dir, "data_interpolation_comparison_combined.png"))")

println("\nData verification plots created! Check the plots directory to verify the data looks correct.")
println("Continuing with analysis...")

println("\n" * "="^50)
println("GENERATING PLOTS")
println("="^50)

# Extract the data subset and convert to vector format
X_subset = X[all_used_indices, :]

# No padding needed since we interpolated to 32 points (power of 2)
println("Using interpolated data with $(size(X_subset, 2)) time points (power of 2)")

X_subset_vec = [X_subset[i, :] for i in 1:size(X_subset, 1)]

# Create time vector
t = collect(1:size(X_subset, 2))

# println("\n" * "="^50)
# println("ANALYSIS 1: RAW DATA ONLY")
# println("="^50)

# # Run WICMAD with Conservative-Balanced configuration on raw data only
# println("Running WICMAD with Conservative-Balanced configuration on raw data...")
# println("Data shape: $(size(X_subset))")
# println("Revealed indices: $(length(revealed_normal_indices))")

# results_raw = wicmad(X_subset_vec, t;
#     n_iter=5000,
#     burn=2000,
#     thin=1,
#     alpha_prior=conservative_balanced_config[:alpha_prior],
#     a_eta=conservative_balanced_config[:a_eta],
#     b_eta=conservative_balanced_config[:b_eta],
#     a_sig=conservative_balanced_config[:a_sig],
#     b_sig=conservative_balanced_config[:b_sig],
#     revealed_idx=revealed_normal_indices,
#     warmup_iters=500,
#     diagnostics=true,
#     wf="sym8"
# )

# println("WICMAD Raw Data Analysis completed!")

# # Extract clustering results for Raw Data Analysis
# cluster_assignments_raw = vec(results_raw.Z[end, :])  # Get final MCMC sample as vector
# n_clusters_raw = length(unique(cluster_assignments_raw))
# cluster_distribution_raw = countmap(cluster_assignments_raw)

# println("\n" * "="^50)
# println("CLUSTERING RESULTS - RAW DATA")
# println("="^50)
# println("Number of clusters: $n_clusters_raw")
# println("Cluster distribution: $cluster_distribution_raw")

# # Calculate performance metrics for Raw Data Analysis
# predicted_labels_raw = zeros(Int, length(revealed_labels))
# for i in 1:length(revealed_labels)
#     if cluster_assignments_raw[i] == cluster_assignments_raw[1]  # Assume first cluster is normal
#         predicted_labels_raw[i] = 0
#     else
#         predicted_labels_raw[i] = 1
#     end
# end

# # Calculate metrics
# ari_raw = adj_rand_index(revealed_labels, predicted_labels_raw)
# precision_raw = sum((revealed_labels .== 1) .& (predicted_labels_raw .== 1)) / max(1, sum(predicted_labels_raw .== 1))
# recall_raw = sum((revealed_labels .== 1) .& (predicted_labels_raw .== 1)) / max(1, sum(revealed_labels .== 1))
# f1_raw = 2 * precision_raw * recall_raw / max(1e-10, precision_raw + recall_raw)

# println("Performance Metrics (Raw Data):")
# println("ARI: $(round(ari_raw, digits=4))")
# println("Precision: $(round(precision_raw, digits=4))")
# println("Recall: $(round(recall_raw, digits=4))")
# println("F1 Score: $(round(f1_raw, digits=4))")

# # Confusion Matrix for Raw Data Analysis
# println("\nConfusion Matrix (Raw Data):")
# println("                 Predicted")
# println("                 Normal  Anomaly")
# println("True Normal     $(lpad(sum((revealed_labels .== 0) .& (predicted_labels_raw .== 0)), 6))  $(lpad(sum((revealed_labels .== 0) .& (predicted_labels_raw .== 1)), 6))")
# println("True Anomaly    $(lpad(sum((revealed_labels .== 1) .& (predicted_labels_raw .== 0)), 6))  $(lpad(sum((revealed_labels .== 1) .& (predicted_labels_raw .== 1)), 6))")

# # Find partition with maximum F1 score for Raw Data Analysis
# println("\nFinding partition with maximum F1 score (Raw Data)...")
# max_f1_raw, best_partition_raw, best_f1_partition_raw = find_best_f1_partition(results_raw.Z, revealed_labels)
# println("Maximum F1 score found: $(round(max_f1_raw, digits=4))")

# # Confusion Matrix for F1-maximizing partition (Raw Data)
# println("\nF1-Maximizing Confusion Matrix (Raw Data):")
# println("                 Predicted")
# println("                 Normal  Anomaly")
# println("True Normal     $(lpad(sum((revealed_labels .== 0) .& (best_f1_partition_raw .== 0)), 6))  $(lpad(sum((revealed_labels .== 0) .& (best_f1_partition_raw .== 1)), 6))")
# println("True Anomaly    $(lpad(sum((revealed_labels .== 1) .& (best_f1_partition_raw .== 0)), 6))  $(lpad(sum((revealed_labels .== 1) .& (best_f1_partition_raw .== 1)), 6))")

# # Find MAP estimate for Raw Data Analysis
# println("\nFinding MAP estimate (Raw Data)...")
# map_partition_raw, map_loglik_raw = compute_map_estimate(results_raw.Z, results_raw.loglik)
# println("MAP log-likelihood: $(round(map_loglik_raw, digits=2))")

# # Convert MAP partition to binary labels
# cluster_counts_map_raw = countmap(map_partition_raw)
# largest_cluster_map_raw = argmax(cluster_counts_map_raw)
# predicted_labels_map_raw = [assign == largest_cluster_map_raw ? 0 : 1 for assign in map_partition_raw]

# # MAP Confusion Matrix (Raw Data)
# println("\nMAP Confusion Matrix (Raw Data):")
# println("                 Predicted")
# println("                 Normal  Anomaly")
# println("True Normal     $(lpad(sum((revealed_labels .== 0) .& (predicted_labels_map_raw .== 0)), 6))  $(lpad(sum((revealed_labels .== 0) .& (predicted_labels_map_raw .== 1)), 6))")
# println("True Anomaly    $(lpad(sum((revealed_labels .== 1) .& (predicted_labels_map_raw .== 0)), 6))  $(lpad(sum((revealed_labels .== 1) .& (predicted_labels_map_raw .== 1)), 6))")

# # MH Acceptance Rates for Raw Data Analysis
# println("\nMH Acceptance Rates (Raw Data):")
# println("="^50)
# for (k, param) in enumerate(results_raw.params)
#     # Only print if this cluster has any samples assigned to it
#     cluster_samples = count(==(k), cluster_assignments_raw)
#     if cluster_samples > 0
#         println("Cluster $k (samples: $cluster_samples):")
#         println("  Kernel parameters:")
#         for (param_name, acc_count) in param.acc.kernel
#             rate = acc_count.n > 0 ? round(acc_count.a / acc_count.n, digits=4) : 0.0
#             println("    $param_name: $(acc_count.a)/$(acc_count.n) ($(rate*100)%)")
#         end
#         println("  L matrix: $(param.acc.L.a)/$(param.acc.L.n) ($(param.acc.L.n > 0 ? round(param.acc.L.a / param.acc.L.n, digits=4) * 100 : 0)%)")
#         println("  Eta parameters:")
#         for (m, eta_acc) in enumerate(param.acc.eta)
#             rate = eta_acc.n > 0 ? round(eta_acc.a / eta_acc.n, digits=4) : 0.0
#             println("    eta[$m]: $(eta_acc.a)/$(eta_acc.n) ($(rate*100)%)")
#         end
#         println("  TauB: $(param.acc.tauB.a)/$(param.acc.tauB.n) ($(param.acc.tauB.n > 0 ? round(param.acc.tauB.a / param.acc.tauB.n, digits=4) * 100 : 0)%)")
#     end
# end

println("\n" * "="^50)
println("ANALYSIS 2: RAW + DERIVATIVE DATA")
println("="^50)

# Compute derivatives and create multivariate functional dataset
println("Computing derivatives and creating multivariate functional dataset...")
X_multivariate = compute_derivatives(X_subset)
println("Multivariate data shape: $(size(X_multivariate))")

# Convert to vector format for WICMAD (each element is a 2D matrix representing a multivariate functional observation)
X_multivariate_vec = [X_multivariate[i, :, :] for i in 1:size(X_multivariate, 1)]

# Create time vector for multivariate data
t_multivariate = collect(1:size(X_multivariate, 2))

# Run WICMAD with Conservative-Balanced configuration on multivariate functional data
println("Running WICMAD with Conservative-Balanced configuration on multivariate functional data...")
println("Multivariate data shape: $(size(X_multivariate))")
println("Revealed indices: $(length(revealed_normal_indices))")

# Comprehensive analysis: F1 scores vs revealed normal percentage
println("\n" * "="^60)
println("COMPREHENSIVE ANALYSIS: F1 SCORES VS REVEALED PERCENTAGE")
println("="^60)

# Define revealed percentages to test (0% to 100% in 5% increments)
revealed_percentages = collect(0:5:100)
n_percentages = length(revealed_percentages)

# Storage for results
dahl_f1_scores = Vector{Float64}(undef, n_percentages)
map_f1_scores = Vector{Float64}(undef, n_percentages)
dahl_cluster_counts = Vector{Int}(undef, n_percentages)
map_cluster_counts = Vector{Int}(undef, n_percentages)

println("Testing $(n_percentages) different revealed percentages: $(revealed_percentages)%")
println("Each run uses: 100 bootstrap runs, 3000 MCMC iterations (1000 burn-in)")
println()

# Define bootstrap parameters for the analysis
bootstrap_runs = 100
n_iter = 3000
burn = 1000
thin = 1
warmup_iters = 500

# Run analysis for each revealed percentage
for (i, pct) in enumerate(revealed_percentages)
println("="^50)
    println("ANALYSIS $(i)/$(n_percentages): $(pct)% REVEALED NORMAL")
println("="^50)

    # Calculate number of normal samples to reveal
    local n_revealed_normal = max(0, round(Int, pct/100 * n_normal_used))
    local revealed_normal_subset = sample(used_normal_indices, min(n_revealed_normal, length(used_normal_indices)), replace=false)
    revealed_normal_indices_current = findall(idx -> idx in revealed_normal_subset, all_used_indices)
    
    println("Revealing $(length(revealed_normal_indices_current)) out of $(n_normal_used) normal samples ($(round(100 * length(revealed_normal_indices_current) / n_normal_used, digits=1))%)")
    
    # Run bootstrap WICMAD with current revealed indices
    println("Running bootstrap WICMAD...")
    results_current = wicmad(X_multivariate_vec, t_multivariate;
        n_iter=n_iter,
        burn=burn,
        thin=thin,
        alpha_prior=conservative_balanced_config[:alpha_prior],
        a_eta=conservative_balanced_config[:a_eta],
        b_eta=conservative_balanced_config[:b_eta],
        a_sig=conservative_balanced_config[:a_sig],
        b_sig=conservative_balanced_config[:b_sig],
        revealed_idx=revealed_normal_indices_current,
        warmup_iters=warmup_iters,
        diagnostics=true,
        wf="sym8",
        # Bootstrap parameters
        bootstrap_runs=bootstrap_runs,
        bootstrap_parallel=:threads,
        bootstrap_seed=42
    )
    
    # Extract clustering results
    if hasfield(typeof(results_current), :z_consensus)
        # Bootstrap results
        cluster_assignments_dahl = results_current.z_consensus
        cluster_assignments_map = results_current.z_map_consensus
        
        println("Bootstrap completed! Dahl: $(length(unique(cluster_assignments_dahl))) clusters, MAP: $(length(unique(cluster_assignments_map))) clusters")
    else
        # Regular MCMC results (fallback)
        cluster_assignments_dahl = vec(results_current.Z[end, :])
        cluster_assignments_map = cluster_assignments_dahl
        println("MCMC completed! $(length(unique(cluster_assignments_dahl))) clusters")
    end
    
    # Calculate F1 scores for both Dahl and MAP consensus
    for (method_name, assignments) in [("Dahl", cluster_assignments_dahl), ("MAP", cluster_assignments_map)]
        # Convert to binary labels (anomaly detection)
        cluster_counts = countmap(assignments)
        largest_cluster = argmax(cluster_counts)
        predicted_labels = [assign == largest_cluster ? 0 : 1 for assign in assignments]
        
        # Calculate F1 score
        tp = sum((revealed_labels .== 1) .& (predicted_labels .== 1))
        fp = sum((revealed_labels .== 0) .& (predicted_labels .== 1))
        fn = sum((revealed_labels .== 1) .& (predicted_labels .== 0))
        
        precision = tp > 0 ? tp / (tp + fp) : 0.0
        recall = tp > 0 ? tp / (tp + fn) : 0.0
        f1_score = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0.0
        
        if method_name == "Dahl"
            dahl_f1_scores[i] = f1_score
            dahl_cluster_counts[i] = length(unique(assignments))
        else
            map_f1_scores[i] = f1_score
            map_cluster_counts[i] = length(unique(assignments))
        end
        
        println("$(method_name) F1 Score: $(round(f1_score, digits=4)) (Precision: $(round(precision, digits=4)), Recall: $(round(recall, digits=4)))")
    end
    
    println("Completed $(pct)% revealed normal analysis")
    println()
end

# Print summary table
println("="^80)
println("SUMMARY RESULTS")
println("="^80)
println("Revealed% | Dahl F1  | MAP F1   | Dahl Clusters | MAP Clusters")
println("-"^80)
for i in 1:n_percentages
    pct = revealed_percentages[i]
    println("$(lpad(pct, 8))% | $(lpad(round(dahl_f1_scores[i], digits=4), 8)) | $(lpad(round(map_f1_scores[i], digits=4), 8)) | $(lpad(dahl_cluster_counts[i], 13)) | $(lpad(map_cluster_counts[i], 12))")
end
println("="^80)

# Create F1 score vs revealed percentage plots
println("\nCreating F1 score vs revealed percentage plots...")

# Main F1 score comparison plot
p_f1_comparison = plot(size=(1000, 600), 
                      title="F1 Score vs Revealed Normal Percentage",
                      xlabel="Revealed Normal Percentage (%)",
                      ylabel="F1 Score",
                      legend=:topright)

plot!(p_f1_comparison, revealed_percentages, dahl_f1_scores, 
      label="Dahl Consensus", color=:blue, linewidth=3, marker=:circle, markersize=6)
plot!(p_f1_comparison, revealed_percentages, map_f1_scores, 
      label="MAP Consensus", color=:red, linewidth=3, marker=:square, markersize=6)

# Add horizontal line at F1 = 0.5 for reference
hline!(p_f1_comparison, [0.5], color=:gray, linestyle=:dash, alpha=0.7, label="F1 = 0.5")

savefig(p_f1_comparison, joinpath(plots_dir, "f1_scores_vs_revealed_percentage.png"))
println("Saved: $(joinpath(plots_dir, "f1_scores_vs_revealed_percentage.png"))")

# Cluster count comparison plot
p_cluster_comparison = plot(size=(1000, 600),
                           title="Number of Clusters vs Revealed Normal Percentage",
                           xlabel="Revealed Normal Percentage (%)",
                           ylabel="Number of Clusters",
                           legend=:topright)

plot!(p_cluster_comparison, revealed_percentages, dahl_cluster_counts,
      label="Dahl Consensus", color=:blue, linewidth=3, marker=:circle, markersize=6)
plot!(p_cluster_comparison, revealed_percentages, map_cluster_counts,
      label="MAP Consensus", color=:red, linewidth=3, marker=:square, markersize=6)

savefig(p_cluster_comparison, joinpath(plots_dir, "cluster_counts_vs_revealed_percentage.png"))
println("Saved: $(joinpath(plots_dir, "cluster_counts_vs_revealed_percentage.png"))")

# Combined plot with both F1 scores and cluster counts
p_combined = plot(layout=(2, 1), size=(1000, 800))

# F1 scores subplot
plot!(p_combined[1], revealed_percentages, dahl_f1_scores, 
      label="Dahl Consensus", color=:blue, linewidth=3, marker=:circle, markersize=6)
plot!(p_combined[1], revealed_percentages, map_f1_scores, 
      label="MAP Consensus", color=:red, linewidth=3, marker=:square, markersize=6)
hline!(p_combined[1], [0.5], color=:gray, linestyle=:dash, alpha=0.7, label="F1 = 0.5")
plot!(p_combined[1], title="F1 Score vs Revealed Normal Percentage",
      xlabel="Revealed Normal Percentage (%)", ylabel="F1 Score", legend=:topright)

# Cluster counts subplot
plot!(p_combined[2], revealed_percentages, dahl_cluster_counts,
      label="Dahl Consensus", color=:blue, linewidth=3, marker=:circle, markersize=6)
plot!(p_combined[2], revealed_percentages, map_cluster_counts,
      label="MAP Consensus", color=:red, linewidth=3, marker=:square, markersize=6)
plot!(p_combined[2], title="Number of Clusters vs Revealed Normal Percentage",
      xlabel="Revealed Normal Percentage (%)", ylabel="Number of Clusters", legend=:topright)

savefig(p_combined, joinpath(plots_dir, "comprehensive_analysis_results.png"))
println("Saved: $(joinpath(plots_dir, "comprehensive_analysis_results.png"))")

# Find and report optimal revealed percentage
best_dahl_idx = argmax(dahl_f1_scores)
best_map_idx = argmax(map_f1_scores)

println("\n" * "="^60)
println("OPTIMAL RESULTS")
println("="^60)
println("Best Dahl Consensus F1 Score: $(round(dahl_f1_scores[best_dahl_idx], digits=4)) at $(revealed_percentages[best_dahl_idx])% revealed")
println("Best MAP Consensus F1 Score: $(round(map_f1_scores[best_map_idx], digits=4)) at $(revealed_percentages[best_map_idx])% revealed")
println("="^60)

println("WICMAD Comprehensive Analysis completed!")

println("\n" * "="^60)
println("ANALYSIS COMPLETED SUCCESSFULLY")
println("="^60)
println("Comprehensive F1 score analysis completed for all revealed percentages (0-100% in 5% increments)")
println("Results saved to plots directory:")
println("- f1_scores_vs_revealed_percentage.png")
println("- cluster_counts_vs_revealed_percentage.png") 
println("- comprehensive_analysis_results.png")
println("="^60)