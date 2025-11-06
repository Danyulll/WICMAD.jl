#!/usr/bin/env julia

# ------------------------------------------------------------
# WICMAD Coffee Dataset - Smart Parallel Prior Search
# 
# Intelligent parallel search with automatic cluster detection
# and adaptive parameter selection
# ------------------------------------------------------------

using Pkg
Pkg.activate(@__DIR__)

# Add the WICMAD package to the environment
Pkg.develop(path=dirname(@__DIR__))

using WICMAD
using Plots
using StatsBase: sample, countmap
using Random
using Statistics: mean, std, median
using DataFrames
using CSV
using Distributed

# Setup parallel workers
# if nworkers() == 1
#     addprocs(min(4, Sys.CPU_THREADS))
# end
addprocs(10)
println("Coffee Dataset Smart Parallel Prior Search")
println("="^50)
println("Workers: $(nworkers())")

# Load packages on all workers
@everywhere begin
    using WICMAD
    using StatsBase: sample, countmap
    using Random
    using Statistics: mean, std, median
end

Random.seed!(42)

# Load Coffee dataset (same as before)
function read_arff_data(filepath)
    data = []
    labels = Int64[]
    
    in_data_section = false
    
    open(filepath, "r") do file
        for line in eachline(file)
            line = strip(line)
            
            if line == "@data"
                in_data_section = true
                continue
            end
            
            if in_data_section && !isempty(line)
                parts = split(line, ",")
                if length(parts) >= 2
                    label = parse(Int64, parts[end])
                    series_data = [parse(Float64, x) for x in parts[1:end-1]]
                    push!(data, series_data)
                    push!(labels, label)
                end
            end
        end
    end
    
    return data, labels
end

println("\nLoading Coffee dataset...")
data_dir = joinpath(dirname(@__DIR__), "data", "Coffee")
train_data, train_labels = read_arff_data(joinpath(data_dir, "Coffee_TRAIN.arff"))
test_data, test_labels = read_arff_data(joinpath(data_dir, "Coffee_TEST.arff"))

all_data = vcat(train_data, test_data)
all_labels = vcat(train_labels, test_labels)

# Data preparation
P = 64
interpolate_to_length(data, target_length) = collect(range(minimum(data), maximum(data), length=target_length))

normal_indices = findall(==(0), all_labels)
anomaly_indices = findall(==(1), all_labels)
# Make dataset 15% anomalies
# Calculate how many anomalies we can use (15% of what we have available)
n_anomaly_available = length(anomaly_indices)
# For 15% anomaly ratio: n_anomaly / (n_anomaly + n_normal) = 0.15
# This means: n_anomaly = 0.15 * (n_anomaly + n_normal) => n_normal = n_anomaly * (1/0.15 - 1) = n_anomaly * (85/15)
n_anomaly_used = min(n_anomaly_available, length(normal_indices) * 15 ÷ 85)
n_anomaly_used = max(1, n_anomaly_used)  # At least 1 anomaly
selected_anomaly_indices = sample(anomaly_indices, n_anomaly_used, replace=false)
# Select corresponding normal samples to maintain 15% anomaly ratio
# n_normal = n_anomaly * (85/15), but limit by available normals
n_normal_needed = round(Int, n_anomaly_used * 85 / 15)
n_normal_used = min(n_normal_needed, length(normal_indices))
selected_normal_indices = sample(normal_indices, n_normal_used, replace=false)
selected_indices = vcat(selected_normal_indices, selected_anomaly_indices)

t_grid = collect(range(0, 1, length=P))

Y_raw = []
for i in selected_indices
    interpolated_data = interpolate_to_length(all_data[i], P)
    push!(Y_raw, reshape(interpolated_data, P, 1))
end

# Map original indices to new indices in the subset
original_to_new = Dict(zip(selected_indices, 1:length(selected_indices)))

# Reveal 15% of the normal group
n_normal_in_subset = length(selected_normal_indices)
n_revealed = max(1, round(Int, 0.15 * n_normal_in_subset))
original_revealed_idx = sample(selected_normal_indices, n_revealed, replace=false)

# Map to new indices in the subset
revealed_idx = [original_to_new[idx] for idx in original_revealed_idx if idx in keys(original_to_new)]

# Create normal_indices for the subset (all normal indices, not including revealed ones)
normal_indices_subset_new = [original_to_new[idx] for idx in selected_normal_indices if idx in keys(original_to_new)]
normal_indices_subset = setdiff(normal_indices_subset_new, revealed_idx)

println("Data prepared: $(length(Y_raw)) samples, $(length(revealed_idx)) revealed")

# Smart configuration generation
println("\n" * "="^50)
println("SMART CONFIGURATION GENERATION")
println("="^50)

# Define smart configurations based on anomaly detection principles
smart_configs = [
    # Conservative configurations (fewer clusters, higher precision)
    Dict(
        :name => "Conservative-HighPrecision",
        :alpha_prior => (5.0, 1.0),
        :a_eta => 3.0,
        :b_eta => 0.05,
        :a_sig => 3.0,
        :b_sig => 0.01
    ),
    Dict(
        :name => "Conservative-Balanced",
        :alpha_prior => (8.0, 1.0),
        :a_eta => 2.5,
        :b_eta => 0.08,
        :a_sig => 2.5,
        :b_sig => 0.02
    ),
    
    # Default configurations
    Dict(
        :name => "Default-Standard",
        :alpha_prior => (10.0, 1.0),
        :a_eta => 2.0,
        :b_eta => 0.1,
        :a_sig => 2.5,
        :b_sig => 0.02
    ),
    Dict(
        :name => "Default-HighNoise",
        :alpha_prior => (10.0, 1.0),
        :a_eta => 1.5,
        :b_eta => 0.15,
        :a_sig => 2.0,
        :b_sig => 0.03
    ),
    
    # Liberal configurations (more clusters, higher recall)
    Dict(
        :name => "Liberal-HighRecall",
        :alpha_prior => (20.0, 1.0),
        :a_eta => 2.0,
        :b_eta => 0.1,
        :a_sig => 2.5,
        :b_sig => 0.02
    ),
    Dict(
        :name => "Liberal-Balanced",
        :alpha_prior => (15.0, 1.0),
        :a_eta => 2.5,
        :b_eta => 0.08,
        :a_sig => 2.5,
        :b_sig => 0.02
    ),
    
    # Adaptive configurations (auto-tuning)
    Dict(
        :name => "Adaptive-LowNoise",
        :alpha_prior => (12.0, 1.0),
        :a_eta => 3.0,
        :b_eta => 0.06,
        :a_sig => 3.0,
        :b_sig => 0.015
    ),
    Dict(
        :name => "Adaptive-HighNoise",
        :alpha_prior => (12.0, 1.0),
        :a_eta => 1.5,
        :b_eta => 0.12,
        :a_sig => 2.0,
        :b_sig => 0.04
    ),
    
    # Extreme configurations for boundary testing
    Dict(
        :name => "Extreme-Conservative",
        :alpha_prior => (3.0, 1.0),
        :a_eta => 4.0,
        :b_eta => 0.03,
        :a_sig => 3.5,
        :b_sig => 0.008
    ),
    Dict(
        :name => "Extreme-Liberal",
        :alpha_prior => (30.0, 1.0),
        :a_eta => 1.0,
        :b_eta => 0.2,
        :a_sig => 2.0,
        :b_sig => 0.05
    )
]

# MCMC parameters
n_iter = 5000
burnin = 2000
thin = 1
warmup_iters = 500

println("Testing $(length(smart_configs)) smart configurations...")
println("MCMC: $(n_iter) iter, $(burnin) burn, thin=$(thin)")

# Run wavelet selection once on the main process to avoid parallel conflicts
println("\n" * "="^50)
println("WAVELET SELECTION")
println("="^50)

# Use wavelet selection function directly to avoid parallel conflicts
global selected_wavelet = "db8"  # Default fallback
try
    println("Running wavelet selection...")
    wavelet_selection_result = KernelSelection.select_wavelet(Y_raw, t_grid, revealed_idx; 
                                                             wf_candidates=["haar", "db2", "db4", "db6", "db8"],
                                                             J=nothing, 
                                                             boundary="periodic",
                                                             mcmc=(n_iter=10, burnin=1, thin=1))
    global selected_wavelet = wavelet_selection_result.selected_wf
    println("Successfully selected wavelet: $selected_wavelet")
catch e
    println("Warning: Wavelet selection failed with: $e")
    println("Using default wavelet: $selected_wavelet")
end

println("Selected wavelet: $selected_wavelet")
println("="^50)

# Parallel evaluation function
@everywhere function evaluate_smart_config(config_data)
    config, Y_data, t_data, revealed_data, normal_indices_data, n_iter, burnin, thin, warmup_iters, selected_wf = config_data
    
    try
        results = wicmad(Y_data, t_data,
                        n_iter=n_iter,
                        burn=burnin,
                        thin=thin,
                        alpha_prior=config[:alpha_prior],
                        a_eta=config[:a_eta],
                        b_eta=config[:b_eta],
                        a_sig=config[:a_sig],
                        b_sig=config[:b_sig],
                        revealed_idx=revealed_data,
                        warmup_iters=warmup_iters,
                        diagnostics=false,
                        wf=selected_wf,
                        bootstrap_runs=0)
        
        # Calculate comprehensive metrics
        # Check if we have valid results
        if isempty(results.Z) || size(results.Z, 1) == 0
            error("No valid clustering results found")
        end
        
        # Use explicit indexing instead of end
        last_row_idx = size(results.Z, 1)
        final_clusters = vec(results.Z[last_row_idx, :])
        n_clusters = length(unique(final_clusters))
        
        # ARI calculation
        true_labels = [i in normal_indices_data ? 0 : 1 for i in 1:length(final_clusters)]
        predicted_labels = [i in revealed_data ? 0 : (final_clusters[i] == final_clusters[revealed_data[1]] ? 0 : 1) for i in eachindex(final_clusters)]
        ari_score = adj_rand_index(true_labels, predicted_labels)
        
        # Anomaly detection metrics
        true_positives = sum((true_labels .== 1) .& (predicted_labels .== 1))
        false_positives = sum((true_labels .== 0) .& (predicted_labels .== 1))
        false_negatives = sum((true_labels .== 1) .& (predicted_labels .== 0))
        
        precision = true_positives / (true_positives + false_positives + eps())
        recall = true_positives / (true_positives + false_negatives + eps())
        f1_score = 2 * precision * recall / (precision + recall + eps())
        
        # Cluster quality metrics
        alpha_mean = mean(results.alpha)
        loglik_mean = mean(results.loglik)
        
        # Stability analysis
        n_samples = size(results.Z, 1)
        if n_samples > 1
            stability_scores = [adj_rand_index(vec(results.Z[i-1, :]), vec(results.Z[i, :])) for i in 2:n_samples]
            avg_stability = mean(stability_scores)
            stability_std = std(stability_scores)
        else
            avg_stability = 1.0
            stability_std = 0.0
        end
        
        # Kernel analysis
        kernel_counts = countmap(results.kern)
        most_used_kernel = argmax(kernel_counts)
        kernel_usage = maximum(values(kernel_counts)) / length(results.kern)
        
        # Auto-clustering score (weighted combination)
        auto_score = 0.35 * ari_score + 0.25 * f1_score + 0.25 * avg_stability + 0.15 * precision
        
        # Cluster efficiency (penalize too many or too few clusters)
        optimal_clusters = 2  # For anomaly detection
        cluster_efficiency = exp(-abs(n_clusters - optimal_clusters) / optimal_clusters)
        
        # Final composite score
        composite_score = 0.7 * auto_score + 0.3 * cluster_efficiency
        
        return Dict(
            :name => config[:name],
            :alpha_prior => config[:alpha_prior],
            :a_eta => config[:a_eta],
            :b_eta => config[:b_eta],
            :a_sig => config[:a_sig],
            :b_sig => config[:b_sig],
            :n_clusters => n_clusters,
            :ari => ari_score,
            :f1_score => f1_score,
            :precision => precision,
            :recall => recall,
            :alpha_mean => alpha_mean,
            :loglik_mean => loglik_mean,
            :most_used_kernel => most_used_kernel,
            :kernel_usage => kernel_usage,
            :stability => avg_stability,
            :stability_std => stability_std,
            :auto_score => auto_score,
            :cluster_efficiency => cluster_efficiency,
            :composite_score => composite_score,
            :success => true
        )
        
    catch e
        println("Error in config $(config[:name]): $e")
        return Dict(
            :name => config[:name],
            :success => false,
            :error => string(e)
        )
    end
end

# Prepare configurations for parallel execution
config_data_list = []
for config in smart_configs
    config_data = (config, Y_raw, t_grid, revealed_idx, normal_indices_subset, n_iter, burnin, thin, warmup_iters, selected_wavelet)
    push!(config_data_list, config_data)
end

# Run parallel evaluation
println("\n" * "="^50)
println("RUNNING SMART PARALLEL SEARCH")
println("="^50)

start_time = time()
results_list = pmap(evaluate_smart_config, config_data_list)
end_time = time()

println("Parallel execution completed in $(round(end_time - start_time, digits=2)) seconds")

# Analyze results
println("\n" * "="^50)
println("SMART SEARCH RESULTS")
println("="^50)

successful_results = filter(r -> r[:success], results_list)
failed_results = filter(r -> !r[:success], results_list)

println("Successful: $(length(successful_results)), Failed: $(length(failed_results))")

if length(successful_results) > 0
    df = DataFrame(successful_results)
    sort!(df, :composite_score, rev=true)
    
    println("\nResults ranked by Composite Score:")
    println("="^90)
    println("Rank | Name                | Composite | Auto-Score | ARI    | F1     | Precision | Clusters | Stability")
    println("="^90)
    
    for i in 1:nrow(df)
        row = df[i, :]
        println("$(lpad(i, 4)) | $(lpad(row[:name], 20)) | $(lpad(round(row[:composite_score], digits=4), 9)) | $(lpad(round(row[:auto_score], digits=4), 10)) | $(lpad(round(row[:ari], digits=4), 6)) | $(lpad(round(row[:f1_score], digits=4), 6)) | $(lpad(round(row[:precision], digits=4), 9)) | $(lpad(row[:n_clusters], 8)) | $(lpad(round(row[:stability], digits=4), 9))")
    end
    
    # Best configuration
    best = df[1, :]
    println("\n" * "="^60)
    println("BEST CONFIGURATION: $(best[:name])")
    println("="^60)
    println("Composite Score: $(round(best[:composite_score], digits=4))")
    println("Auto-Clustering Score: $(round(best[:auto_score], digits=4))")
    println("ARI Score: $(round(best[:ari], digits=4))")
    println("F1 Score: $(round(best[:f1_score], digits=4))")
    println("Precision: $(round(best[:precision], digits=4))")
    println("Recall: $(round(best[:recall], digits=4))")
    println("Stability: $(round(best[:stability], digits=4)) ± $(round(best[:stability_std], digits=4))")
    println("Final Clusters: $(best[:n_clusters])")
    println("Cluster Efficiency: $(round(best[:cluster_efficiency], digits=4))")
    println("Alpha Prior: $(best[:alpha_prior])")
    println("Eta Prior: ($(best[:a_eta]), $(best[:b_eta]))")
    println("Sigma Prior: ($(best[:a_sig]), $(best[:b_sig]))")
    println("Most Used Kernel: $(best[:most_used_kernel]) ($(round(best[:kernel_usage]*100, digits=1))%)")
    
    # Create analysis plots
    p1 = bar(df[!, :name], df[!, :composite_score], 
             title="Composite Score by Configuration",
             xlabel="Configuration",
             ylabel="Composite Score",
             legend=false,
             xrotation=45)
    
    p2 = scatter(df[!, :ari], df[!, :f1_score], 
                 title="ARI vs F1 Score",
                 xlabel="ARI Score",
                 ylabel="F1 Score",
                 legend=false)
    
    p3 = scatter(df[!, :n_clusters], df[!, :composite_score], 
                 title="Composite Score vs Cluster Number",
                 xlabel="Number of Clusters",
                 ylabel="Composite Score",
                 legend=false)
    
    p4 = scatter(df[!, :stability], df[!, :composite_score], 
                 title="Stability vs Composite Score",
                 xlabel="Stability",
                 ylabel="Composite Score",
                 legend=false)
    
    plot_smart = plot(p1, p2, p3, p4, layout=(2,2), size=(800, 600))
    savefig(plot_smart, "coffee_smart_prior_search.png")
    println("\nSaved analysis plot to: coffee_smart_prior_search.png")
    
    # Save results
    CSV.write("coffee_smart_prior_search.csv", df)
    println("Saved results to: coffee_smart_prior_search.csv")
    
    # Recommendations
    println("\n" * "="^50)
    println("RECOMMENDATIONS")
    println("="^50)
    
    # Find best configuration for each criterion
    best_ari = df[argmax(df.ari), :]
    best_f1 = df[argmax(df.f1_score), :]
    best_stability = df[argmax(df.stability), :]
    
    println("Best for ARI: $(best_ari[:name]) (ARI: $(round(best_ari[:ari], digits=4)))")
    println("Best for F1: $(best_f1[:name]) (F1: $(round(best_f1[:f1_score], digits=4)))")
    println("Best for Stability: $(best_stability[:name]) (Stability: $(round(best_stability[:stability], digits=4)))")
    println("Best Overall: $(best[:name]) (Composite: $(round(best[:composite_score], digits=4)))")
    
else
    println("No successful configurations!")
end

# Clean up
rmprocs(workers())

println("\nSmart parallel search completed!")
