#!/usr/bin/env julia

# ------------------------------------------------------------
# WICMAD ArrowHead Dataset - 100 Run Analysis for MAP Confusion Matrix
# 
# Runs the Raw + Derivative analysis 100 times and computes average MAP confusion matrix
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

# Interpolate data down to 32 time points (power of 2)
println("\n" * "="^50)
println("INTERPOLATING DATA TO 32 TIME POINTS")
println("="^50)

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

# Use all normal samples and enough anomalies to reach 15% of total
n_normal_used = min(length(normal_indices), n_normal_target)
n_anomaly_used = min(length(anomaly_indices), n_anomaly_target)

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
println("Actual anomaly percentage: $(round(100 * n_anomaly_used / length(all_used_indices), digits=1))%")
println("Revealed labels distribution: $(countmap(revealed_labels))")

# For WICMAD, we need to specify which indices are revealed (normal samples)
# We'll reveal all normal samples since they're the "known normal" group
revealed_normal_indices = findall(idx -> idx in used_normal_indices, all_used_indices)

# Extract the data subset and convert to vector format
X_subset = X[all_used_indices, :]
X_subset_vec = [X_subset[i, :] for i in 1:size(X_subset, 1)]

# Create time vector
t = collect(1:size(X_subset, 2))

# Compute derivatives and create multivariate functional dataset
println("Computing derivatives and creating multivariate functional dataset...")
X_multivariate = compute_derivatives(X_subset)
println("Multivariate data shape: $(size(X_multivariate))")

# Convert to vector format for WICMAD (each element is a 2D matrix representing a multivariate functional observation)
X_multivariate_vec = [X_multivariate[i, :, :] for i in 1:size(X_multivariate, 1)]

# Create time vector for multivariate data
t_multivariate = collect(1:size(X_multivariate, 2))

println("\n" * "="^60)
println("RUNNING 100 WICMAD ANALYSES")
println("="^60)

# Storage for confusion matrices
confusion_matrices = []
ari_scores = []
f1_scores = []
n_clusters_list = []

# Run 100 analyses
for run in 1:100
    println("Run $run/100...")
    
    # Set different random seed for each run
    Random.seed!(42 + run)
    
    # Run WICMAD with Conservative-Balanced configuration on multivariate functional data
    results_combined = wicmad(X_multivariate_vec, t_multivariate;
        n_iter=5000,
        burn=2000,
        thin=1,
        alpha_prior=conservative_balanced_config[:alpha_prior],
        a_eta=conservative_balanced_config[:a_eta],
        b_eta=conservative_balanced_config[:b_eta],
        a_sig=conservative_balanced_config[:a_sig],
        b_sig=conservative_balanced_config[:b_sig],
        revealed_idx=revealed_normal_indices,
        warmup_iters=500,
        diagnostics=true,
        wf="sym8"
    )
    
    # Extract clustering results
    cluster_assignments_combined = vec(results_combined.Z[end, :])
    n_clusters_combined = length(unique(cluster_assignments_combined))
    
    # Find MAP estimate
    map_partition_combined, map_loglik_combined = compute_map_estimate(results_combined.Z, results_combined.loglik)
    
    # Convert MAP partition to binary labels
    cluster_counts_map_combined = countmap(map_partition_combined)
    largest_cluster_map_combined = argmax(cluster_counts_map_combined)
    predicted_labels_map_combined = [assign == largest_cluster_map_combined ? 0 : 1 for assign in map_partition_combined]
    
    # Calculate confusion matrix
    tp = sum((revealed_labels .== 1) .& (predicted_labels_map_combined .== 1))
    fp = sum((revealed_labels .== 0) .& (predicted_labels_map_combined .== 1))
    fn = sum((revealed_labels .== 1) .& (predicted_labels_map_combined .== 0))
    tn = sum((revealed_labels .== 0) .& (predicted_labels_map_combined .== 0))
    
    confusion_matrix = [tn fp; fn tp]
    push!(confusion_matrices, confusion_matrix)
    
    # Calculate ARI
    ari = adj_rand_index(revealed_labels, predicted_labels_map_combined)
    push!(ari_scores, ari)
    
    # Calculate F1 score
    precision = tp > 0 ? tp / (tp + fp) : 0.0
    recall = tp > 0 ? tp / (tp + fn) : 0.0
    f1_score = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0.0
    push!(f1_scores, f1_score)
    
    push!(n_clusters_list, n_clusters_combined)
    
    # Print progress every 10 runs
    if run % 10 == 0
        println("Completed $run runs. Current average ARI: $(round(mean(ari_scores), digits=4)), F1: $(round(mean(f1_scores), digits=4))")
    end
end

println("\n" * "="^60)
println("RESULTS SUMMARY")
println("="^60)

# Compute average confusion matrix
avg_confusion_matrix = mean(confusion_matrices)
std_confusion_matrix = std(confusion_matrices)

println("Average MAP Confusion Matrix (Raw + Derivative Data):")
println("                 Predicted")
println("                 Normal  Anomaly")
println("True Normal     $(lpad(round(Int, avg_confusion_matrix[1,1]), 6))  $(lpad(round(Int, avg_confusion_matrix[1,2]), 6))")
println("True Anomaly    $(lpad(round(Int, avg_confusion_matrix[2,1]), 6))  $(lpad(round(Int, avg_confusion_matrix[2,2]), 6))")

println("\nStandard Deviation of Confusion Matrix:")
println("                 Predicted")
println("                 Normal  Anomaly")
println("True Normal     $(lpad(round(std_confusion_matrix[1,1], digits=2), 6))  $(lpad(round(std_confusion_matrix[1,2], digits=2), 6))")
println("True Anomaly    $(lpad(round(std_confusion_matrix[2,1], digits=2), 6))  $(lpad(round(std_confusion_matrix[2,2], digits=2), 6))")

# Calculate performance metrics from average confusion matrix
avg_tp = avg_confusion_matrix[2,2]
avg_fp = avg_confusion_matrix[1,2]
avg_fn = avg_confusion_matrix[2,1]
avg_tn = avg_confusion_matrix[1,1]

avg_precision = avg_tp / (avg_tp + avg_fp)
avg_recall = avg_tp / (avg_tp + avg_fn)
avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)

println("\nAverage Performance Metrics:")
println("ARI: $(round(mean(ari_scores), digits=4)) ± $(round(std(ari_scores), digits=4))")
println("Precision: $(round(avg_precision, digits=4))")
println("Recall: $(round(avg_recall, digits=4))")
println("F1 Score: $(round(avg_f1, digits=4))")
println("F1 Score (from individual runs): $(round(mean(f1_scores), digits=4)) ± $(round(std(f1_scores), digits=4))")

println("\nNumber of Clusters:")
println("Average: $(round(mean(n_clusters_list), digits=2)) ± $(round(std(n_clusters_list), digits=2))")
println("Range: $(minimum(n_clusters_list)) - $(maximum(n_clusters_list))")

# Create plots directory
plots_dir = joinpath(dirname(@__DIR__), "plots")
if !isdir(plots_dir)
    mkpath(plots_dir)
end

# Plot distribution of confusion matrix elements
p_confusion_dist = plot(layout=(2, 2), size=(800, 600))

# Extract individual confusion matrix elements
tn_values = [cm[1,1] for cm in confusion_matrices]
fp_values = [cm[1,2] for cm in confusion_matrices]
fn_values = [cm[2,1] for cm in confusion_matrices]
tp_values = [cm[2,2] for cm in confusion_matrices]

histogram!(p_confusion_dist[1], tn_values, title="True Negatives Distribution", 
          xlabel="Count", ylabel="Frequency", bins=20, alpha=0.7)
histogram!(p_confusion_dist[2], fp_values, title="False Positives Distribution", 
          xlabel="Count", ylabel="Frequency", bins=20, alpha=0.7)
histogram!(p_confusion_dist[3], fn_values, title="False Negatives Distribution", 
          xlabel="Count", ylabel="Frequency", bins=20, alpha=0.7)
histogram!(p_confusion_dist[4], tp_values, title="True Positives Distribution", 
          xlabel="Count", ylabel="Frequency", bins=20, alpha=0.7)

savefig(p_confusion_dist, joinpath(plots_dir, "confusion_matrix_distributions.png"))
println("Saved: $(joinpath(plots_dir, "confusion_matrix_distributions.png"))")

# Plot ARI and F1 score distributions
p_performance = plot(layout=(1, 2), size=(800, 400))

histogram!(p_performance[1], ari_scores, title="ARI Score Distribution", 
          xlabel="ARI", ylabel="Frequency", bins=20, alpha=0.7)
histogram!(p_performance[2], f1_scores, title="F1 Score Distribution", 
          xlabel="F1 Score", ylabel="Frequency", bins=20, alpha=0.7)

savefig(p_performance, joinpath(plots_dir, "performance_distributions.png"))
println("Saved: $(joinpath(plots_dir, "performance_distributions.png"))")

# Plot number of clusters distribution
p_clusters = histogram(n_clusters_list, title="Number of Clusters Distribution", 
                      xlabel="Number of Clusters", ylabel="Frequency", bins=20, alpha=0.7)

savefig(p_clusters, joinpath(plots_dir, "clusters_distribution.png"))
println("Saved: $(joinpath(plots_dir, "clusters_distribution.png"))")

# Create summary table
summary_df = DataFrame(
    Run = 1:100,
    ARI = ari_scores,
    F1_Score = f1_scores,
    N_Clusters = n_clusters_list,
    True_Negatives = tn_values,
    False_Positives = fp_values,
    False_Negatives = fn_values,
    True_Positives = tp_values
)

CSV.write(joinpath(plots_dir, "summary_results.csv"), summary_df)
println("Saved: $(joinpath(plots_dir, "summary_results.csv"))")

println("\n" * "="^60)
println("ANALYSIS COMPLETED SUCCESSFULLY")
println("="^60)
println("Results saved to: $plots_dir")
println("Configuration used: $(conservative_balanced_config[:name])")
println("Average MAP Confusion Matrix:")
println("                 Predicted")
println("                 Normal  Anomaly")
println("True Normal     $(lpad(round(Int, avg_confusion_matrix[1,1]), 6))  $(lpad(round(Int, avg_confusion_matrix[1,2]), 6))")
println("True Anomaly    $(lpad(round(Int, avg_confusion_matrix[2,1]), 6))  $(lpad(round(Int, avg_confusion_matrix[2,2]), 6))")
println("="^60)

