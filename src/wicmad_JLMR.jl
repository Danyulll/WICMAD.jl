#!/usr/bin/env julia

# WICMAD: Comprehensive Dataset Analysis
# This script runs WICMAD on all simulated and real datasets.
# - Simulated: Median metrics over 100 MC trials
# - Real: Single run metrics
# All plots are saved as PNG files.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using WICMAD, Random, Statistics, Printf
using StatsBase: countmap, sample, median, shuffle
using Plots, Interpolations
using WICMAD.PostProcessing: map_from_res
using ProgressMeter
using RCall

gr()
include(joinpath(@__DIR__, "sim_core.jl"))
project_root = dirname(@__DIR__)
mkpath(joinpath(project_root, "plots", "notebook_output"))

# Helper functions
function largest_cluster(z)
    counts = countmap(z)
    return sort(collect(keys(counts)); by = k -> -counts[k])[1]
end

to_binary_labels(z) = [zi == largest_cluster(z) ? 0 : 1 for zi in z]

function compute_metrics(z_pred, y_true)
    pred_bin = to_binary_labels(z_pred)
    tp = sum((y_true .== 1) .& (pred_bin .== 1))
    tn = sum((y_true .== 0) .& (pred_bin .== 0))
    fp = sum((y_true .== 0) .& (pred_bin .== 1))
    fn = sum((y_true .== 1) .& (pred_bin .== 0))
    precision = tp + fp > 0 ? tp / (tp + fp) : 0.0
    recall = tp + fn > 0 ? tp / (tp + fn) : 0.0
    f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0
    accuracy = (tp + tn) / length(y_true)
    return (; precision, recall, f1, accuracy, tp, tn, fp, fn)
end

function plot_dataset(Y_list, y_true, t, title_str; save_path=nothing)
    M = size(Y_list[1], 2)
    plt = plot(layout=(M,1), size=(1000, 300*M))
    for m in 1:M
        for i in 1:length(Y_list)
            color, alpha_val = y_true[i] == 0 ? (:blue, 0.4) : (:red, 0.6)
            plot!(plt[m], t, Y_list[i][:, m], color=color, alpha=alpha_val, linewidth=1, label="")
        end
        plot!(plt[m], title="Channel $m", xlabel="Time", ylabel="Value")
        m == 1 && (plot!(plt[m], [NaN], [NaN], color=:blue, linewidth=2, label="Normal"); plot!(plt[m], [NaN], [NaN], color=:red, linewidth=2, label="Anomaly"))
    end
    plot!(plt, plot_title=title_str)
    if !isnothing(save_path)
        savefig(plt, save_path)
        println("Saved plot to: $save_path")
    else
        display(plt)
    end
end

plot_clustered(Y_list, z_pred, t, title_str; save_path=nothing) = 
    plot_dataset(Y_list, to_binary_labels(z_pred), t, title_str; save_path=save_path)

function interpolate_to_length(series; target_len=32)
    out = similar(series)
    for (i, X) in pairs(series)
        P0, M0 = size(X)
        Xn = zeros(Float64, target_len, M0)
        for m in 1:M0
            itp = linear_interpolation(collect(1:P0), X[:, m])
            Xn[:, m] = itp.(range(1, P0, length=target_len))
        end
        out[i] = Xn
    end
    return out, target_len
end

function print_metrics(name, m)
    println("\n" * "="^70, "\n$name\n", "="^70)
    @printf("Precision: %.4f  Recall: %.4f  F1: %.4f  Accuracy: %.4f\n", m.precision, m.recall, m.f1, m.accuracy)
    println("-"^70)
    @printf("Confusion Matrix:     Predicted Normal | Predicted Anomaly\n")
    @printf("True Normal           %6d (TN)      | %6d (FP)\n", m.tn, m.fp)
    @printf("True Anomaly          %6d (FN)      | %6d (TP)\n", m.fn, m.tp)
    println("="^70)
end

println("Helper functions loaded")

# ============================================================================
# R Setup and Functional Data Depth Functions
# ============================================================================

# Initialize R and install/load required packages
println("Setting up R environment...")
R"""
if (!require("fda.usc", quietly = TRUE)) {
    install.packages("fda.usc", repos = "https://cloud.r-project.org")
    library(fda.usc)
}
"""

# Function to find threshold that maximizes F1 score
function find_best_f1_threshold(scores, y_true)
    unique_scores = sort(unique(scores), rev=true)
    best_f1 = -Inf
    best_threshold = unique_scores[1]
    best_metrics = nothing
    
    for threshold in unique_scores
        pred = Int.(scores .>= threshold)
        m = compute_metrics_from_binary(pred, y_true)
        if m.f1 > best_f1
            best_f1 = m.f1
            best_threshold = threshold
            best_metrics = m
        end
    end
    
    return (; threshold=best_threshold, metrics=best_metrics)
end

# Helper to compute metrics from binary predictions
function compute_metrics_from_binary(pred, y_true)
    tp = sum((y_true .== 1) .& (pred .== 1))
    tn = sum((y_true .== 0) .& (pred .== 0))
    fp = sum((y_true .== 0) .& (pred .== 1))
    fn = sum((y_true .== 1) .& (pred .== 0))
    precision = tp + fp > 0 ? tp / (tp + fp) : 0.0
    recall = tp + fn > 0 ? tp / (tp + fn) : 0.0
    f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0
    accuracy = (tp + tn) / length(y_true)
    return (; precision, recall, f1, accuracy, tp, tn, fp, fn)
end

# Convert Julia data to R format for functional depth computation
function prepare_r_data(Y_list, t)
    N = length(Y_list)
    M = size(Y_list[1], 2)
    P = length(t)
    
    # Create matrix N x P (use first channel for univariate or when M=1)
    data_matrix = zeros(N, P)
    for i in 1:N
        data_matrix[i, :] = Y_list[i][:, 1]
    end
    return data_matrix
end

# Convert multivariate Julia data to R mfdata format
function prepare_r_mfdata(Y_list, t)
    N = length(Y_list)
    M = size(Y_list[1], 2)  # number of channels/components
    P = length(t)
    
    # Create array N x P x M (observations x time x channels)
    data_array = zeros(N, P, M)
    for i in 1:N
        for m in 1:M
            data_array[i, :, m] = Y_list[i][:, m]
        end
    end
    return data_array
end

# Compute functional depth using fda.usc methods
function compute_fdepth_fm(Y_list, t, y_true)
    try
        data_matrix = prepare_r_data(Y_list, t)
        @rput data_matrix t
        R"""
        library(fda.usc)
        fdata_obj <- fdata(data_matrix, argvals = t, rangeval = range(t))
        depth_result <- depth.FM(fdata_obj, trim = 0.1, dfunc = "FM1", par.dfunc = list(scale = TRUE), draw = FALSE)
        depth_vals <- depth_result$dep
        """
        depth_vals = @rget depth_vals
        anomaly_scores = 1.0 .- depth_vals
        return find_best_f1_threshold(anomaly_scores, y_true)
    catch e
        @warn "depth.FM failed: $e"
        return nothing
    end
end

function compute_fdepth_rt(Y_list, t, y_true)
    try
        data_matrix = prepare_r_data(Y_list, t)
        @rput data_matrix t
        R"""
        library(fda.usc)
        fdata_obj <- fdata(data_matrix, argvals = t, rangeval = range(t))
        depth_result <- depth.RT(fdata_obj, trim = 0.1, nproj = 50, proj = "vexponential", draw = FALSE)
        depth_vals <- depth_result$dep
        """
        depth_vals = @rget depth_vals
        anomaly_scores = 1.0 .- depth_vals
        return find_best_f1_threshold(anomaly_scores, y_true)
    catch e
        @warn "depth.RT failed: $e"
        return nothing
    end
end

function compute_fdepth_rpd(Y_list, t, y_true)
    try
        data_matrix = prepare_r_data(Y_list, t)
        @rput data_matrix t
        R"""
        library(fda.usc)
        fdata_obj <- fdata(data_matrix, argvals = t, rangeval = range(t))
        depth_result <- depth.RPD(fdata_obj, deriv = c(0, 1), dfunc2 = mdepth.LD, trim = 0.1, draw = FALSE)
        depth_vals <- depth_result$dep
        """
        depth_vals = @rget depth_vals
        anomaly_scores = 1.0 .- depth_vals
        return find_best_f1_threshold(anomaly_scores, y_true)
    catch e
        @warn "depth.RPD failed: $e"
        return nothing
    end
end

function compute_fdepth_kfsd(Y_list, t, y_true)
    try
        data_matrix = prepare_r_data(Y_list, t)
        @rput data_matrix t
        R"""
        library(fda.usc)
        fdata_obj <- fdata(data_matrix, argvals = t, rangeval = range(t))
        depth_result <- depth.KFSD(fdata_obj, trim = 0.1, h = NULL, scale = FALSE, draw = FALSE)
        depth_vals <- depth_result$dep
        """
        depth_vals = @rget depth_vals
        anomaly_scores = 1.0 .- depth_vals
        return find_best_f1_threshold(anomaly_scores, y_true)
    catch e
        @warn "depth.KFSD failed: $e"
        return nothing
    end
end

function compute_fdepth_mode(Y_list, t, y_true)
    try
        data_matrix = prepare_r_data(Y_list, t)
        @rput data_matrix t
        R"""
        library(fda.usc)
        fdata_obj <- fdata(data_matrix, argvals = t, rangeval = range(t))
        depth_result <- depth.mode(fdata_obj, trim = 0.1, h = NULL, metric = metric.lp, draw = FALSE)
        depth_vals <- depth_result$dep
        """
        depth_vals = @rget depth_vals
        anomaly_scores = 1.0 .- depth_vals
        return find_best_f1_threshold(anomaly_scores, y_true)
    catch e
        @warn "depth.mode failed: $e"
        return nothing
    end
end

function compute_fdepth_rp(Y_list, t, y_true)
    try
        data_matrix = prepare_r_data(Y_list, t)
        @rput data_matrix t
        R"""
        library(fda.usc)
        fdata_obj <- fdata(data_matrix, argvals = t, rangeval = range(t))
        depth_result <- depth.RP(fdata_obj, trim = 0.1, nproj = 50, proj = "vexponential", draw = FALSE)
        depth_vals <- depth_result$dep
        """
        depth_vals = @rget depth_vals
        anomaly_scores = 1.0 .- depth_vals
        return find_best_f1_threshold(anomaly_scores, y_true)
    catch e
        @warn "depth.RP failed: $e"
        return nothing
    end
end

function compute_fdepth_fsd(Y_list, t, y_true)
    try
        data_matrix = prepare_r_data(Y_list, t)
        @rput data_matrix t
        R"""
        library(fda.usc)
        fdata_obj <- fdata(data_matrix, argvals = t, rangeval = range(t))
        depth_result <- depth.FSD(fdata_obj, trim = 0.1, scale = FALSE, draw = FALSE)
        depth_vals <- depth_result$dep
        """
        depth_vals = @rget depth_vals
        anomaly_scores = 1.0 .- depth_vals
        return find_best_f1_threshold(anomaly_scores, y_true)
    catch e
        @warn "depth.FSD failed: $e"
        return nothing
    end
end

# Multivariate functional depth methods (for M > 1)
function compute_fdepth_fmp(Y_list, t, y_true)
    try
        data_array = prepare_r_mfdata(Y_list, t)
        @rput data_array t
        R"""
        library(fda.usc)
        n <- dim(data_array)[1]
        m <- dim(data_array)[2]
        p <- dim(data_array)[3]
        
        X_list <- vector("list", p)
        tgrid <- $t
        rangeval <- range(tgrid)
        
        for (j in seq_len(p)) {
            X_list[[j]] <- fdata(
                data = data_array[,,j],
                argvals = tgrid,
                rangeval = rangeval
            )
        }
        
        mf <- do.call(mfdata, X_list)
        depth_result <- depth.FMp(mf)
        depth_vals <- depth_result$dep
        """
        depth_vals = @rget depth_vals
        anomaly_scores = 1.0 .- depth_vals
        return find_best_f1_threshold(anomaly_scores, y_true)
    catch e
        @warn "depth.FMp failed: $e"
        return nothing
    end
end

function compute_fdepth_modep(Y_list, t, y_true)
    try
        data_array = prepare_r_mfdata(Y_list, t)
        @rput data_array t
        R"""
        library(fda.usc)
        n <- dim(data_array)[1]
        m <- dim(data_array)[2]
        p <- dim(data_array)[3]
        
        X_list <- vector("list", p)
        tgrid <- $t
        rangeval <- range(tgrid)
        
        for (j in seq_len(p)) {
            X_list[[j]] <- fdata(
                data = data_array[,,j],
                argvals = tgrid,
                rangeval = rangeval
            )
        }
        
        mf <- do.call(mfdata, X_list)
        depth_result <- depth.modep(mf)
        depth_vals <- depth_result$dep
        """
        depth_vals = @rget depth_vals
        anomaly_scores = 1.0 .- depth_vals
        return find_best_f1_threshold(anomaly_scores, y_true)
    catch e
        @warn "depth.modep failed: $e"
        return nothing
    end
end

function compute_fdepth_rpp(Y_list, t, y_true)
    try
        data_array = prepare_r_mfdata(Y_list, t)
        @rput data_array t
        R"""
        library(fda.usc)
        n <- dim(data_array)[1]
        m <- dim(data_array)[2]
        p <- dim(data_array)[3]
        
        X_list <- vector("list", p)
        tgrid <- $t
        rangeval <- range(tgrid)
        
        for (j in seq_len(p)) {
            X_list[[j]] <- fdata(
                data = data_array[,,j],
                argvals = tgrid,
                rangeval = rangeval
            )
        }
        
        mf <- do.call(mfdata, X_list)
        depth_result <- depth.RPp(mf)
        depth_vals <- depth_result$dep
        """
        depth_vals = @rget depth_vals
        anomaly_scores = 1.0 .- depth_vals
        return find_best_f1_threshold(anomaly_scores, y_true)
    catch e
        @warn "depth.RPp failed: $e"
        return nothing
    end
end


# Run all depth methods on a dataset
function run_all_depth_methods(Y_list, t, y_true; is_multivariate=false)
    methods = Dict{String, Union{Nothing, NamedTuple}}()
    
    println("  Computing functional depths...")
    
    if is_multivariate
        # Use multivariate functional depth methods
        methods["depth.FMp"] = compute_fdepth_fmp(Y_list, t, y_true)
        methods["depth.modep"] = compute_fdepth_modep(Y_list, t, y_true)
        methods["depth.RPp"] = compute_fdepth_rpp(Y_list, t, y_true)
    else
        # Use univariate functional depth methods from fda.usc
        methods["depth.FM"] = compute_fdepth_fm(Y_list, t, y_true)
        methods["depth.mode"] = compute_fdepth_mode(Y_list, t, y_true)
        methods["depth.RT"] = compute_fdepth_rt(Y_list, t, y_true)
        methods["depth.RP"] = compute_fdepth_rp(Y_list, t, y_true)
        methods["depth.RPD"] = compute_fdepth_rpd(Y_list, t, y_true)
        methods["depth.FSD"] = compute_fdepth_fsd(Y_list, t, y_true)
        methods["depth.KFSD"] = compute_fdepth_kfsd(Y_list, t, y_true)
    end
    
    return methods
end

println("R environment and depth functions loaded")

# ============================================================================
# Part 1: Simulated Datasets
# For each simulated dataset, we run 100 Monte Carlo trials and report median metrics.
# ============================================================================

const mc_trials = 1
const base_seed_val = 0x000000000135CFF1

# Function to run MC trials for a simulated dataset
function run_simulated_experiment(anomaly_type, dataset_name; is_multivariate=false, regime=nothing)
    println("\n" * "="^70)
    println("Running experiment: $dataset_name")
    println("="^70)
    
    # Plot ground truth data
    Random.seed!(base_seed_val)
    if is_multivariate
        data = make_multivariate_dataset(; regime=regime, t=t_grid)
    else
        data = make_univariate_dataset(; anomaly_type=anomaly_type, t=t_grid)
    end
    plot_path = joinpath(project_root, "plots", "notebook_output", "$(dataset_name)_ground_truth.png")
    plot_dataset(data.Y_list, data.y_true, data.t, "$dataset_name - Ground Truth"; save_path=plot_path)
    
    # Run WICMAD and depth methods on MC trials
    all_metrics_wicmad = []
    all_metrics_depths = Dict{String, Vector{NamedTuple}}()
    
    # Determine if multivariate based on first data generation
    Random.seed!(base_seed_val)
    if is_multivariate
        test_data = make_multivariate_dataset(; regime=regime, t=t_grid)
    else
        test_data = make_univariate_dataset(; anomaly_type=anomaly_type, t=t_grid)
    end
    M = size(test_data.Y_list[1], 2)
    is_multivariate_data = M > 1
    
    # Initialize depth method storage (different for univariate vs multivariate)
    if is_multivariate_data
        depth_method_names = ["depth.FMp", "depth.modep", "depth.RPp"]
    else
        depth_method_names = ["depth.FM", "depth.mode", "depth.RT", "depth.RP", "depth.RPD",
                              "depth.FSD", "depth.KFSD"]
    end
    for method in depth_method_names
        all_metrics_depths[method] = []
    end
    
    for mc_idx in 1:mc_trials
        println("MC run $mc_idx")
        Random.seed!(base_seed_val + mc_idx - 1)
        if is_multivariate
            data_mc = make_multivariate_dataset(; regime=regime, t=t_grid)
        else
            data_mc = make_univariate_dataset(; anomaly_type=anomaly_type, t=t_grid)
        end
        
        # Run all depth methods first (use appropriate methods for univariate vs multivariate)
        M = size(data_mc.Y_list[1], 2)
        is_multivariate_data = M > 1
        depth_results = run_all_depth_methods(data_mc.Y_list, data_mc.t, data_mc.y_true; is_multivariate=is_multivariate_data)
        for (method_name, result) in depth_results
            if !isnothing(result) && !isnothing(result.metrics)
                push!(all_metrics_depths[method_name], result.metrics)
            end
        end
        
        # Run WICMAD after depth methods
        normal_indices = findall(==(0), data_mc.y_true)
        revealed_idx = sort(sample(normal_indices, max(1, round(Int, 0.15 * length(normal_indices))), replace=false))
        res = wicmad(data_mc.Y_list, data_mc.t; n_iter=5000, burn=2000, thin=1,
                     wf="sym8", revealed_idx=revealed_idx, verbose=false)
        mapr = map_from_res(res)
        push!(all_metrics_wicmad, compute_metrics(mapr.z_hat, data_mc.y_true))
    end
    
    # Display median metrics for WICMAD
    if !isempty(all_metrics_wicmad)
        median_metrics_wicmad = (
            precision = median([m.precision for m in all_metrics_wicmad]),
            recall = median([m.recall for m in all_metrics_wicmad]),
            f1 = median([m.f1 for m in all_metrics_wicmad]),
            accuracy = median([m.accuracy for m in all_metrics_wicmad]),
            tp = round(Int, median([m.tp for m in all_metrics_wicmad])),
            tn = round(Int, median([m.tn for m in all_metrics_wicmad])),
            fp = round(Int, median([m.fp for m in all_metrics_wicmad])),
            fn = round(Int, median([m.fn for m in all_metrics_wicmad]))
        )
        print_metrics("WICMAD - $dataset_name [Median over $mc_trials MC]", median_metrics_wicmad)
    end
    
    # Display median metrics for each depth method
    println("\n" * "="^70)
    println("Functional Depth Methods - $dataset_name [Median over $mc_trials MC]")
    println("="^70)
    for method_name in depth_method_names
        if !isempty(all_metrics_depths[method_name])
            metrics_list = all_metrics_depths[method_name]
            median_m = (
                precision = median([m.precision for m in metrics_list]),
                recall = median([m.recall for m in metrics_list]),
                f1 = median([m.f1 for m in metrics_list]),
                accuracy = median([m.accuracy for m in metrics_list]),
                tp = round(Int, median([m.tp for m in metrics_list])),
                tn = round(Int, median([m.tn for m in metrics_list])),
                fp = round(Int, median([m.fp for m in metrics_list])),
                fn = round(Int, median([m.fn for m in metrics_list]))
            )
            print_metrics(method_name, median_m)
        end
    end
    
    # Plot clustering result
    Random.seed!(base_seed_val)
    if is_multivariate
        data_plot = make_multivariate_dataset(; regime=regime, t=t_grid)
    else
        data_plot = make_univariate_dataset(; anomaly_type=anomaly_type, t=t_grid)
    end
    normal_indices = findall(==(0), data_plot.y_true)
    revealed_idx = sort(sample(normal_indices, max(1, round(Int, 0.15 * length(normal_indices))), replace=false))
    res_plot = wicmad(data_plot.Y_list, data_plot.t; n_iter=5000, burn=2000, thin=1,
                      wf="sym8", revealed_idx=revealed_idx, verbose=false)
    mapr_plot = map_from_res(res_plot)
    plot_path = joinpath(project_root, "plots", "notebook_output", "$(dataset_name)_clustering.png")
    plot_clustered(data_plot.Y_list, mapr_plot.z_hat, data_plot.t, "$dataset_name - WICMAD Clustering"; save_path=plot_path)
end

# 1. Univariate: Isolated Spike
run_simulated_experiment(:isolated, "Univariate_Isolated_Spike")

# 2. Univariate: Magnitude I
run_simulated_experiment(:mag1, "Univariate_Magnitude_I")

# 3. Univariate: Magnitude II
run_simulated_experiment(:mag2, "Univariate_Magnitude_II")

# 4. Univariate: Shape
run_simulated_experiment(:shape, "Univariate_Shape")

# 5. Multivariate: One Anomalous Channel
run_simulated_experiment(nothing, "Multivariate_One_Anomalous_Channel"; is_multivariate=true, regime=:one)

# 6. Multivariate: Two Anomalous Channels
run_simulated_experiment(nothing, "Multivariate_Two_Anomalous_Channels"; is_multivariate=true, regime=:two)

# 7. Multivariate: Three Anomalous Channels
run_simulated_experiment(nothing, "Multivariate_Three_Anomalous_Channels"; is_multivariate=true, regime=:three)

# ============================================================================
# Part 2: Real Datasets
# For real datasets, we run WICMAD once and report the metrics.
# ============================================================================

const p_anom = 0.15
const data_dir = joinpath(project_root, "data")

# Function to run experiment on a real dataset
function run_real_experiment(dataset_name, dataset_path)
    println("\n" * "="^70)
    println("Running experiment: $dataset_name")
    println("="^70)
    
    # Load and prepare data
    Random.seed!(42)
    ds = WICMAD.Utils.load_ucr_dataset(dataset_path, data_dir)
    cm = countmap(ds.labels)
    classes = sort(collect(keys(cm)); by = c -> -cm[c])
    
    # Handle different dataset structures
    if dataset_name == "CharacterTrajectories"
        normal_class = "1"  # letter 'a'
        other_classes = filter(c -> c != normal_class, sort(collect(keys(cm)); by = c -> -cm[c]))
        anomaly_class = other_classes[1]
    else
        normal_class, anomaly_class = classes[1], classes[2]
    end
    
    # Create balanced dataset (~15% anomalies)
    norm_idx = findall(==(normal_class), ds.labels)
    anom_idx = findall(==(anomaly_class), ds.labels)
    n_anom = min(length(anom_idx), floor(Int, length(norm_idx) * p_anom / (1 - p_anom)))
    n_norm = min(length(norm_idx), round(Int, n_anom * (1 - p_anom) / p_anom))
    
    used_idx = shuffle(vcat(sample(norm_idx, n_norm, replace=false), sample(anom_idx, n_anom, replace=false)))
    Y_full = ds.series[used_idx]
    y_full = [ds.labels[i] for i in used_idx]
    Y_interp, P = interpolate_to_length(Y_full; target_len=32)
    t = collect(1:P)
    gt_binary = [yi == normal_class ? 0 : 1 for yi in y_full]
    
    println("$dataset_name: Using $(length(Y_full)) curves, $(countmap(y_full))")
    plot_path = joinpath(project_root, "plots", "notebook_output", "$(dataset_name)_ground_truth.png")
    plot_dataset(Y_interp, gt_binary, t, "$dataset_name - Ground Truth"; save_path=plot_path)
    
    # Run WICMAD with progress bar
    normal_indices = findall(==(normal_class), y_full)
    revealed_idx = sort(sample(normal_indices, max(1, round(Int, 0.15 * length(normal_indices))), replace=false))
    println("Running WICMAD with $(length(revealed_idx)) revealed indices...")
    
    # Create progress bar for MCMC iterations
    n_iter_mcmc = 5000
    p_mcmc = Progress(n_iter_mcmc, desc="MCMC Progress: ", color=:green, showspeed=true, output=stdout)
    
    # Run MCMC with progress tracking
    # Since wicmad prints progress every 5% (250 iterations for 5000 total),
    # we'll update the progress bar based on time estimates
    res_channel = Channel(1)
    
    task = @async begin
        res = wicmad(Y_interp, t; n_iter=n_iter_mcmc, burn=2000, thin=1, wf="sym8", 
                     revealed_idx=revealed_idx, verbose=true)
        put!(res_channel, res)
    end
    
    # Update progress bar periodically while MCMC runs
    # Use time-based estimation with periodic updates
    start_time = time()
    update_interval = 0.05  # Update every 50ms for smoother progress
    last_update = 0
    
    while !istaskdone(task)
        elapsed = time() - start_time
        # Estimate progress based on elapsed time
        # Assume roughly linear progress (this is a heuristic)
        # Typical MCMC takes ~30-120 seconds for 5000 iterations depending on data size
        # We'll use adaptive estimation: start with a conservative estimate and adjust
        estimated_time_per_iter = 0.015  # Rough estimate: 15ms per iteration
        estimated_iter = min(n_iter_mcmc, round(Int, elapsed / estimated_time_per_iter))
        
        # Update progress bar if we've made progress
        if estimated_iter > last_update && estimated_iter <= n_iter_mcmc
            update!(p_mcmc, estimated_iter)
            last_update = estimated_iter
        end
        
        sleep(update_interval)
    end
    
    # Get the result
    res = take!(res_channel)
    
    # Complete the progress bar
    update!(p_mcmc, n_iter_mcmc)
    finish!(p_mcmc)
    println("✓ WICMAD completed")
    
    # Display WICMAD metrics
    mapr = map_from_res(res)
    metrics_wicmad = compute_metrics(mapr.z_hat, gt_binary)
    print_metrics("WICMAD - $dataset_name", metrics_wicmad)
    
    # Determine if multivariate and run appropriate depth methods
    M = size(Y_interp[1], 2)
    is_multivariate_data = M > 1
    
    # Run all depth methods (use appropriate methods for univariate vs multivariate)
    depth_results = run_all_depth_methods(Y_interp, t, gt_binary; is_multivariate=is_multivariate_data)
    
    # Display metrics for each depth method
    println("\n" * "="^70)
    println("Functional Depth Methods - $dataset_name")
    println("="^70)
    for (method_name, result) in depth_results
        if !isnothing(result) && !isnothing(result.metrics)
            print_metrics(method_name, result.metrics)
        end
    end
    
    # Plot clustering result
    plot_path = joinpath(project_root, "plots", "notebook_output", "$(dataset_name)_clustering.png")
    plot_clustered(Y_interp, mapr.z_hat, t, "$dataset_name - WICMAD Clustering"; save_path=plot_path)
end

# 1. AsphaltRegularity
run_real_experiment("AsphaltRegularity", "AsphaltRegularity/AsphaltRegularity")

# 2. CharacterTrajectories
run_real_experiment("CharacterTrajectories", "CharacterTrajectories/CharacterTrajectories")

# 3. Chinatown
run_real_experiment("Chinatown", "Chinatown/Chinatown")

# ============================================================================
# Summary
# ============================================================================

println("\n" * "="^70)
println("Summary")
println("="^70)
println("""
This script demonstrated WICMAD and functional data depth methods' performance across:

**Simulated Datasets (Median over $mc_trials MC trials):**
1. Univariate: Isolated spike
2. Univariate: Magnitude I
3. Univariate: Magnitude II
4. Univariate: Shape
5. Multivariate: One anomalous channel
6. Multivariate: Two anomalous channels
7. Multivariate: Three anomalous channels

**Real Datasets (Single run):**
1. AsphaltRegularity
2. CharacterTrajectories
3. Chinatown

**Methods compared (simulated datasets only):**
- WICMAD (Bayesian nonparametric clustering)

**Univariate simulated datasets:**
- depth.FM (Fraiman-Muniz / Integrated depth, fda.usc)
- depth.mode (h-modal depth, fda.usc)
- depth.RT (Random Tukey depth, fda.usc)
- depth.RP (Random projection depth, fda.usc)
- depth.RPD (Double random projection depth, fda.usc)
- depth.FSD (Functional spatial depth, fda.usc)
- depth.KFSD (Kernelized functional spatial depth, fda.usc)

**Multivariate simulated datasets:**
- depth.FMp (Fraiman-Muniz multivariate depth, fda.usc)
- depth.modep (h-modal multivariate depth, fda.usc)
- depth.RPp (Random projection multivariate depth, fda.usc)

**Real datasets:**
- Univariate (AsphaltRegularity, Chinatown): WICMAD + all univariate depth methods
- Multivariate (CharacterTrajectories): WICMAD + all multivariate depth methods

For each depth method, the cutoff that maximizes F1 score was selected.
For each dataset, plots were saved to plots/notebook_output/:
- Ground truth visualizations
- WICMAD clustering visualizations

Performance metrics (Precision, Recall, F1, Accuracy) and confusion matrices
were printed to the console for all methods.
""")
println("="^70)
println("All experiments completed! Check plots/notebook_output/ for PNG files.")

