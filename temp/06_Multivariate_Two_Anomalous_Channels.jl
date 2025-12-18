#!/usr/bin/env julia

# WICMAD: Multivariate Two Anomalous Channels Experiment
# This script runs WICMAD on the Multivariate Two Anomalous Channels simulated dataset.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using WICMAD, Random, Statistics, Printf
using StatsBase: countmap, sample, median, shuffle
using Plots, Interpolations
using WICMAD.PostProcessing: map_from_res
using ProgressMeter
using RCall
using DataFrames, CSV

gr()
include(joinpath(@__DIR__, "..", "src", "sim_core.jl"))
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
if (!require("mrfDepth", quietly = TRUE)) {
    install.packages("mrfDepth", repos = "https://cloud.r-project.org")
    library(mrfDepth)
}
if (!require("fdaoutlier", quietly = TRUE)) {
    install.packages("fdaoutlier", repos = "https://cloud.r-project.org")
    library(fdaoutlier)
}
"""

# Function to find threshold that maximizes F1 score
function find_best_f1_threshold(scores, y_true)
    # Filter out NaN and Inf values
    valid_mask = .!isnan.(scores) .& .!isinf.(scores)
    if !any(valid_mask)
        @warn "All scores are NaN or Inf"
        return nothing
    end
    scores = scores[valid_mask]
    y_true = y_true[valid_mask]
    
    unique_scores = sort(unique(scores), rev=true)
    if isempty(unique_scores)
        @warn "No unique scores found"
        return nothing
    end
    
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
        # data_array dimensions: [N observations, P time points, M channels]
        n_obs <- dim(data_array)[1]      # N: number of observations
        n_time <- dim(data_array)[2]     # P: number of time points
        n_channels <- dim(data_array)[3] # M: number of channels
        
        X_list <- vector("list", n_channels)
        tgrid <- $t
        rangeval <- range(tgrid)
        
        # Create one fdata object per channel
        # Each fdata object should be n_obs x n_time (observations x time points)
        for (j in seq_len(n_channels)) {
            # Extract channel j: this gives us n_obs x n_time matrix
            channel_data <- data_array[,,j]
            X_list[[j]] <- fdata(
                channel_data,
                argvals = tgrid,
                rangeval = rangeval
            )
        }
        
        # Set simple names to avoid variable name length issues
        names(X_list) <- paste0("X", seq_len(n_channels))
        
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

function compute_fdepth_rpp(Y_list, t, y_true)
    try
        data_array = prepare_r_mfdata(Y_list, t)
        @rput data_array t
        R"""
        library(fda.usc)
        # data_array dimensions: [N observations, P time points, M channels]
        n_obs <- dim(data_array)[1]      # N: number of observations
        n_time <- dim(data_array)[2]     # P: number of time points
        n_channels <- dim(data_array)[3] # M: number of channels
        
        X_list <- vector("list", n_channels)
        tgrid <- $t
        rangeval <- range(tgrid)
        
        # Create one fdata object per channel
        # Each fdata object should be n_obs x n_time (observations x time points)
        for (j in seq_len(n_channels)) {
            # Extract channel j: this gives us n_obs x n_time matrix
            channel_data <- data_array[,,j]
            X_list[[j]] <- fdata(
                channel_data,
                argvals = tgrid,
                rangeval = rangeval
            )
        }
        
        # Set simple names to avoid variable name length issues
        names(X_list) <- paste0("X", seq_len(n_channels))
        
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

# mrfDepth: functional outlyingness using fOutl() as feature generator
# Expects array with dimensions (time, curves, components) = (P, N, M)
# Uses fOutlyingnessX as scores and F1-optimal cutoff selection
function compute_fdepth_mrfDepth(Y_list, t, y_true)
    try
        data_array = prepare_r_mfdata(Y_list, t)
        # mrfDepth expects (time, curves, components) = (P, N, M)
        # Our data_array is (N, P, M), so we need to permute dimensions
        N, P, M = size(data_array)
        data_array_transposed = permutedims(data_array, (2, 1, 3))  # (P, N, M)
        @rput data_array_transposed t
        R"""
        library(mrfDepth)
        
        # data_array_transposed dimensions: [P time, N curves, M components]
        # Compute functional outlyingness with type="fAO" (robust to skew)
        # diagnostic=TRUE enables localization features (locOutlX, crossDistsX)
        res <- fOutl(
            x = data_array_transposed,
            type = "fAO",           # functional adjusted outlyingness (robust to skew)
            alpha = 0,              # uniform weights across time
            time = $t,              # optional time grid
            diagnostic = TRUE       # enables locOutlX and crossDistsX for localization
        )
        
        # Extract fOutlyingnessX: length n vector, higher = more outlying
        # This is the scalar outlyingness score per observation
        n_obs <- dim(data_array_transposed)[2]  # Number of observations
        
        if (is.null(res$fOutlyingnessX) || length(res$fOutlyingnessX) != n_obs) {
            stop(paste("fOutlyingnessX has wrong length:", 
                      length(res$fOutlyingnessX), "expected:", n_obs))
        }
        
        anomaly_scores <- as.numeric(res$fOutlyingnessX)
        """
        anomaly_scores = @rget anomaly_scores
        # Ensure we got a valid vector
        if isempty(anomaly_scores) || length(anomaly_scores) != length(y_true)
            @warn "mrfDepth.fOutl returned invalid scores: length=$(length(anomaly_scores)), expected=$(length(y_true))"
            return nothing
        end
        # Use existing F1-optimal cutoff selection
        return find_best_f1_threshold(anomaly_scores, y_true)
    catch e
        @warn "mrfDepth.fOutl failed: $e"
        return nothing
    end
end

# fdaoutlier: MS-plot based multivariate functional outlier detection
# Expects array with dimensions (curves, time, components) = (N, P, M)
function compute_fdepth_fdaoutlier(Y_list, t, y_true)
    try
        data_array = prepare_r_mfdata(Y_list, t)
        # fdaoutlier expects (curves, time, components) = (N, P, M)
        # Our data_array is already (N, P, M), so no transpose needed
        @rput data_array
        R"""
        library(fdaoutlier)
        
        # data_array dimensions: [N curves, P time, M components]
        # MS-plot based multivariate functional outlier detection
        ms_res <- msplot(
            dts = data_array,
            data_depth = "random_projections",
            n_projections = 200,
            plot = FALSE,
            return_mvdir = TRUE
        )
        
        # Use mean directional outlyingness (MO) as anomaly scores
        # Higher MO = more outlying = higher anomaly score
        if (!is.null(ms_res$mvdir) && !is.null(ms_res$mvdir$MO)) {
            anomaly_scores <- ms_res$mvdir$MO
        } else {
            # Fall back to binary outlier flags
            n_obs <- dim(data_array)[1]
            anomaly_scores <- rep(0, n_obs)
            if (length(ms_res$outliers_index) > 0) {
                anomaly_scores[ms_res$outliers_index] <- 1
            }
        }
        """
        anomaly_scores = @rget anomaly_scores
        return find_best_f1_threshold(anomaly_scores, y_true)
    catch e
        @warn "fdaoutlier.msplot failed: $e"
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
        methods["depth.RPp"] = compute_fdepth_rpp(Y_list, t, y_true)
        methods["mrfDepth.fOutl"] = compute_fdepth_mrfDepth(Y_list, t, y_true)
        methods["fdaoutlier.msplot"] = compute_fdepth_fdaoutlier(Y_list, t, y_true)
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
# Experiment: Multivariate Two Anomalous Channels
# ============================================================================

const mc_trials = 100
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
    
    # Store per-trial results for CSV
    csv_rows = []
    
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
        depth_method_names = ["depth.FMp", "depth.RPp", "mrfDepth.fOutl", "fdaoutlier.msplot"]
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
                # Store per-trial result for CSV
                push!(csv_rows, (;
                    mc_trial = mc_idx,
                    method = method_name,
                    precision = result.metrics.precision,
                    recall = result.metrics.recall,
                    f1 = result.metrics.f1,
                    accuracy = result.metrics.accuracy,
                    tp = result.metrics.tp,
                    tn = result.metrics.tn,
                    fp = result.metrics.fp,
                    fn = result.metrics.fn
                ))
            end
        end
        
        # Run WICMAD after depth methods
        normal_indices = findall(==(0), data_mc.y_true)
        revealed_idx = sort(sample(normal_indices, max(1, round(Int, 0.15 * length(normal_indices))), replace=false))
        
        # Wavelet selection
        sel = select_wavelet(data_mc.Y_list, data_mc.t, revealed_idx;
            wf_candidates = nothing, J = nothing, boundary = "periodic",
            mcmc = (n_iter=3000, burnin=1000, thin=1), verbose=false)
        wf_selected = sel.selected_wf
        
        res = wicmad(data_mc.Y_list, data_mc.t; n_iter=5000, burn=2000, thin=1,
                     wf=wf_selected, revealed_idx=revealed_idx, verbose=false)
        mapr = map_from_res(res)
        wicmad_metrics = compute_metrics(mapr.z_hat, data_mc.y_true)
        push!(all_metrics_wicmad, wicmad_metrics)
        # Store WICMAD per-trial result for CSV
        push!(csv_rows, (;
            mc_trial = mc_idx,
            method = "WICMAD",
            precision = wicmad_metrics.precision,
            recall = wicmad_metrics.recall,
            f1 = wicmad_metrics.f1,
            accuracy = wicmad_metrics.accuracy,
            tp = wicmad_metrics.tp,
            tn = wicmad_metrics.tn,
            fp = wicmad_metrics.fp,
            fn = wicmad_metrics.fn
        ))
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
    
    # Save results to CSV
    if !isempty(csv_rows)
        df = DataFrame(csv_rows)
        results_dir = joinpath(project_root, "results")
        mkpath(results_dir)
        csv_path = joinpath(results_dir, "$(dataset_name)_results.csv")
        CSV.write(csv_path, df)
        println("\nSaved results to CSV: $csv_path")
        println("  Total rows: $(nrow(df)) (one per MC trial per method)")
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

# Run the experiment
run_simulated_experiment(nothing, "Multivariate_Two_Anomalous_Channels"; is_multivariate=true, regime=:two)

println("\nExperiment completed!")

