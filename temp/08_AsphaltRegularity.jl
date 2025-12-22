#!/usr/bin/env julia

# WICMAD: AsphaltRegularity Real Dataset Experiment
# This script runs WICMAD on the AsphaltRegularity real dataset.

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
println("  [2.1.1] Checking/installing fda.usc package...")
R"""
if (!require("fda.usc", quietly = TRUE)) {
    cat("[R] Installing fda.usc package...\n")
    install.packages("fda.usc", repos = "https://cloud.r-project.org")
    cat("[R] fda.usc installation complete.\n")
}
cat("[R] Loading fda.usc library...\n")
library(fda.usc)
cat("[R] fda.usc library loaded.\n")
"""
println("  ✓ fda.usc package ready")

println("  [2.1.2] Checking/installing mrfDepth package...")
R"""
if (!require("mrfDepth", quietly = TRUE)) {
    cat("[R] Installing mrfDepth package...\n")
    install.packages("mrfDepth", repos = "https://cloud.r-project.org")
    cat("[R] mrfDepth installation complete.\n")
}
cat("[R] Loading mrfDepth library...\n")
library(mrfDepth)
cat("[R] mrfDepth library loaded.\n")
"""
println("  ✓ mrfDepth package ready")

println("  [2.1.3] Checking/installing fdaoutlier package...")
R"""
if (!require("fdaoutlier", quietly = TRUE)) {
    cat("[R] Installing fdaoutlier package...\n")
    install.packages("fdaoutlier", repos = "https://cloud.r-project.org")
    cat("[R] fdaoutlier installation complete.\n")
}
cat("[R] Loading fdaoutlier library...\n")
library(fdaoutlier)
cat("[R] fdaoutlier library loaded.\n")
"""
println("  ✓ fdaoutlier package ready")

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

println("      [2.2.5.3] compute_fdepth_mrfDepth...")
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
        cat("[R] Starting mrfDepth.fOutl...\n")
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
        cat("[R] mrfDepth.fOutl completed.\n")
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

println("      [2.2.5.4] compute_fdepth_fdaoutlier...")
# fdaoutlier: MS-plot based multivariate functional outlier detection
# Expects array with dimensions (curves, time, components) = (N, P, M)
function compute_fdepth_fdaoutlier(Y_list, t, y_true)
    try
        data_array = prepare_r_mfdata(Y_list, t)
        # fdaoutlier expects (curves, time, components) = (N, P, M)
        # Our data_array is already (N, P, M), so no transpose needed
        @rput data_array
        R"""
        cat("[R] Starting fdaoutlier.msplot...\n")
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
        cat("[R] fdaoutlier.msplot completed.\n")
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
        # Use multivariate functional depth methods (matching simulations)
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
    end
    
    return methods
end

println("R environment and depth functions loaded")

# ============================================================================
# Experiment: AsphaltRegularity Real Dataset
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
    
    # Wavelet selection
    println("Selecting wavelet...")
    sel = select_wavelet(Y_interp, t, revealed_idx;
        wf_candidates = nothing, J = nothing, boundary = "periodic",
        mcmc = (n_iter=3000, burnin=1000, thin=1), verbose=true)
    wf_selected = sel.selected_wf
    println("Selected wavelet: $wf_selected")
    
    # Create progress bar for MCMC iterations
    n_iter_mcmc = 5000
    p_mcmc = Progress(n_iter_mcmc, desc="MCMC Progress: ", color=:green, showspeed=true, output=stdout)
    
    # Run MCMC with progress tracking
    res_channel = Channel(1)
    
    task = @async begin
        res = wicmad(Y_interp, t; n_iter=n_iter_mcmc, burn=2000, thin=1, wf=wf_selected, 
                     revealed_idx=revealed_idx, verbose=true)
        put!(res_channel, res)
    end
    
    # Update progress bar periodically while MCMC runs
    start_time = time()
    update_interval = 0.05
    last_update = 0
    
    while !istaskdone(task)
        elapsed = time() - start_time
        estimated_time_per_iter = 0.015
        estimated_iter = min(n_iter_mcmc, round(Int, elapsed / estimated_time_per_iter))
        
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
    print_metrics("WICMAD (raw) - $dataset_name", metrics_wicmad)
    
    # Store results for CSV
    csv_rows = []
    push!(csv_rows, (;
        method = "WICMAD",
        variant = "raw",
        precision = metrics_wicmad.precision,
        recall = metrics_wicmad.recall,
        f1 = metrics_wicmad.f1,
        accuracy = metrics_wicmad.accuracy,
        tp = metrics_wicmad.tp,
        tn = metrics_wicmad.tn,
        fp = metrics_wicmad.fp,
        fn = metrics_wicmad.fn
    ))
    
    # Determine if multivariate and run appropriate depth methods
    M = size(Y_interp[1], 2)
    is_multivariate_data = M > 1
    
    # For univariate datasets, also test derivative-augmented version
    if !is_multivariate_data
        println("\n" * "="^70)
        println("Running WICMAD on derivative-augmented data (3 channels: raw, d1, d2)")
        println("="^70)
        
        # Create derivative-augmented data
        Y_deriv = derivatives_transform(Y_interp, t)
        
        # Wavelet selection for derivative-augmented data
        println("Selecting wavelet for derivative-augmented data...")
        sel_deriv = select_wavelet(Y_deriv, t, revealed_idx;
            wf_candidates = nothing, J = nothing, boundary = "periodic",
            mcmc = (n_iter=3000, burnin=1000, thin=1), verbose=true)
        wf_deriv = sel_deriv.selected_wf
        println("Selected wavelet (deriv): $wf_deriv")
        
        # Run WICMAD on derivative-augmented data
        p_mcmc_deriv = Progress(n_iter_mcmc, desc="MCMC Progress (deriv): ", color=:green, showspeed=true, output=stdout)
        res_channel_deriv = Channel(1)
        
        task_deriv = @async begin
            res_deriv = wicmad(Y_deriv, t; n_iter=n_iter_mcmc, burn=2000, thin=1, wf=wf_deriv,
                             revealed_idx=revealed_idx, verbose=true)
            put!(res_channel_deriv, res_deriv)
        end
        
        # Update progress bar
        start_time_deriv = time()
        last_update_deriv = 0
        
        while !istaskdone(task_deriv)
            elapsed = time() - start_time_deriv
            estimated_iter = min(n_iter_mcmc, round(Int, elapsed / estimated_time_per_iter))
            
            if estimated_iter > last_update_deriv && estimated_iter <= n_iter_mcmc
                update!(p_mcmc_deriv, estimated_iter)
                last_update_deriv = estimated_iter
            end
            
            sleep(update_interval)
        end
        
        res_deriv = take!(res_channel_deriv)
        update!(p_mcmc_deriv, n_iter_mcmc)
        finish!(p_mcmc_deriv)
        println("✓ WICMAD (derivative-augmented) completed")
        
        # Display metrics for derivative-augmented version
        mapr_deriv = map_from_res(res_deriv)
        metrics_wicmad_deriv = compute_metrics(mapr_deriv.z_hat, gt_binary)
        print_metrics("WICMAD (deriv) - $dataset_name", metrics_wicmad_deriv)
        
        # Store derivative-augmented results for CSV
        push!(csv_rows, (;
            method = "WICMAD",
            variant = "deriv",
            precision = metrics_wicmad_deriv.precision,
            recall = metrics_wicmad_deriv.recall,
            f1 = metrics_wicmad_deriv.f1,
            accuracy = metrics_wicmad_deriv.accuracy,
            tp = metrics_wicmad_deriv.tp,
            tn = metrics_wicmad_deriv.tn,
            fp = metrics_wicmad_deriv.fp,
            fn = metrics_wicmad_deriv.fn
        ))
    end
    
    # Run all depth methods (use appropriate methods for univariate vs multivariate)
    depth_results = run_all_depth_methods(Y_interp, t, gt_binary; is_multivariate=is_multivariate_data)
    
    # Display metrics for each depth method
    println("\n" * "="^70)
    println("Functional Depth Methods - $dataset_name")
    println("="^70)
    for (method_name, result) in depth_results
        if !isnothing(result) && !isnothing(result.metrics)
            print_metrics(method_name, result.metrics)
            # Store result for CSV
            push!(csv_rows, (;
                method = method_name,
                variant = "raw",
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
    
    # Save results to CSV
    if !isempty(csv_rows)
        df = DataFrame(csv_rows)
        results_dir = joinpath(project_root, "results")
        mkpath(results_dir)
        csv_path = joinpath(results_dir, "$(dataset_name)_results.csv")
        CSV.write(csv_path, df)
        println("\nSaved results to CSV: $csv_path")
        println("  Total rows: $(nrow(df)) (one per method)")
    end
    
    # Plot clustering result
    plot_path = joinpath(project_root, "plots", "notebook_output", "$(dataset_name)_clustering.png")
    plot_clustered(Y_interp, mapr.z_hat, t, "$dataset_name - WICMAD Clustering"; save_path=plot_path)
end

# Run the experiment
run_real_experiment("AsphaltRegularity", "AsphaltRegularity/AsphaltRegularity")

println("\nExperiment completed!")

