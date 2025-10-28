#!/usr/bin/env julia

# ------------------------------------------------------------
# WICMAD Simulation Study with Automatic Wavelet Selection
# - Uses new automatic wavelet selection method
# - 100 MC trials with 8 threads
# - Simplified: no grid search, uses defaults + automatic selection
# ------------------------------------------------------------

using Pkg
Pkg.activate(@__DIR__)

# Add the WICMAD package to the environment
Pkg.develop(path=dirname(@__DIR__))

using WICMAD
using Random
using Statistics: mean
using StatsBase: countmap
using Printf
using CSV
using DataFrames
using Dates
using LinearAlgebra
using Distributions
using Plots

# Set random seed for reproducibility
Random.seed!(42)

println("WICMAD Simulation Study with Automatic Wavelet Selection")
println("="^60)

# -----------------------------
# Global controls
# -----------------------------

# Headless plotting
ENV["GKSwstype"] = "100"

# Constants
const P_use        = 64
const t_grid       = collect(range(0, 1, length=P_use))
const Delta        = 1/(P_use - 1)
const sigma_noise  = 0.05

const mc_runs      = 100  # 100 MC trials
const base_seed    = 20240501

# Sampler controls (using defaults)
const n_iter       = 3000
const burnin       = 1000
const thin         = 1
const warmup_iters = 200

# Semi-supervised controls
const reveal_prop  = 0.15

# Output
const timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
const out_root = joinpath("simstudy_results", timestamp)
const metrics_csv = joinpath(out_root, "summary_metrics.csv")

# Create output directories
mkpath(out_root)

println("Writing outputs to: ", abspath(out_root))
println("MC trials: $mc_runs")
println("Using automatic wavelet selection (no grid search)")

# -----------------------------
# GP mean and kernel functions
# -----------------------------
function mean_fun(t)
    0.6 * sin(4 * π * t) + 0.25 * cos(10 * π * t) + 0.1 * t
end

# Quasiperiodic + long-scale SE kernel
function k_qp(t, tprime; ell1=0.15, sig1_sq=1.0, p=0.30, ellp=0.30, ell2=0.60, sig2_sq=0.4)
    dt = t .- tprime'
    se1 = exp.(-(dt.^2) ./ (2 * ell1^2))
    per = exp.(-2 * (sin.(π * dt ./ p)).^2 ./ (ellp^2))
    se2 = exp.(-(dt.^2) ./ (2 * ell2^2))
    sig1_sq * se1 .* per + sig2_sq * se2
end

# Draw GP curves via MVN (Cholesky)
function gp_draw_matrix(N, t, mu_fun, cov_fun, sigma_eps)
    P = length(t)
    mu = mu_fun.(t)
    K = cov_fun(t, t)
    Kc = K + 1e-8 * I(P)
    L = cholesky(Kc).L
    Z = randn(P, N)
    G = mu .+ L' * Z  # P x N (columns are functions)
    X = G' .+ randn(N, P) * sigma_eps  # N x P
    X
end

# -----------------------------
# Anomaly perturbations
# -----------------------------
function add_isolated(xrow, t)
    P = length(t)
    i0 = rand(3:(P-2))
    w = rand(0.3*Delta:0.8*Delta)
    S = rand([-1, +1])
    A = rand(8:12)
    bump = S * A * exp.(-(t .- t[i0]).^2 ./ (2 * w^2))
    xrow .+ bump
end

function add_mag1(xrow)
    S = rand([-1, +1])
    A = rand(12:15)
    xrow .+ S * A
end

function add_mag2(xrow, t)
    P = length(t)
    lambda = floor(Int, 0.10 * P)
    s = rand(1:(P - lambda + 1))
    S = rand([-1, +1])
    A = rand(10:15)
    rfun(u) = 0.5 * (1 - cos(π * u))
    bump = zeros(P)
    idx = s:(s + lambda - 1)
    u = collect(range(0, 1, length=lambda))
    bump[idx] = S * A * rfun.(u)
    xrow .+ bump
end

function add_shape(xrow, t)
    U = rand(0.2:2.0)
    xrow .+ 3 * sin.(2 * π * U * t)
end

# -----------------------------
# Dataset generators
# -----------------------------
function make_univariate_dataset(N=40, anomaly_type="isolated", t=t_grid)
    X = gp_draw_matrix(N, t, mean_fun, k_qp, sigma_noise)  # N x P
    y_true = zeros(Int, N)
    n_anom = max(1, round(Int, 0.10 * N))   # 10%
    idx_anom = randperm(N)[1:n_anom]

    X_pert = copy(X)
    for i in idx_anom
        xi = X[i, :]
        if anomaly_type == "isolated"
            xi = add_isolated(xi, t)
        elseif anomaly_type == "mag1"
            xi = add_mag1(xi)
        elseif anomaly_type == "mag2"
            xi = add_mag2(xi, t)
        elseif anomaly_type == "shape"
            xi = add_shape(xi, t)
        end
        X_pert[i, :] = xi
    end
    y_true[idx_anom] .= 1

    # Wrap as list of P x M matrices with M=1
    Y_list = [reshape(X_pert[i, :], length(t), 1) for i in 1:N]
    (Y_list=Y_list, t=t, y_true=y_true)
end

# Multivariate with 3 channels using 2 latent GP factors
function make_multivariate_dataset(N=40, regime="one", t=t_grid)
    P = length(t)
    U1 = gp_draw_matrix(N, t, mean_fun, k_qp, 0)
    U2 = gp_draw_matrix(N, t, mean_fun, k_qp, 0)
    A = [1.0 0.4;
         0.2 1.0;
         0.7 -0.3]

    Y = Vector{Matrix{Float64}}(undef, N)
    for i in 1:N
        Ui = hcat(U1[i, :], U2[i, :])  # P x 2
        Xi = Ui * A'  # P x 3
        Xi = Xi + randn(P, 3) * sigma_noise
        Y[i] = Xi
    end

    y_true = zeros(Int, N)
    n_anom = max(1, round(Int, 0.10 * N))
    idx_anom = randperm(N)[1:n_anom]
    y_true[idx_anom] .= 1

    types = ["isolated", "mag1", "mag2", "shape"]
    apply_perturb(row, type) = begin
        if type == "isolated"
            add_isolated(row, t)
        elseif type == "mag1"
            add_mag1(row)
        elseif type == "mag2"
            add_mag2(row, t)
        elseif type == "shape"
            add_shape(row, t)
        else
            row
        end
    end

    for i in idx_anom
        Xi = Y[i]
        if regime == "one"
            ch = rand(1:3)
            type = rand(types)
            Xi[:, ch] = apply_perturb(Xi[:, ch], type)
        elseif regime == "two"
            chs = randperm(3)[1:2]
            type = rand(types)  # same type on both
            for ch in chs
                Xi[:, ch] = apply_perturb(Xi[:, ch], type)
            end
        else # "three"
            for ch in 1:3
                type = rand(types)
                Xi[:, ch] = apply_perturb(Xi[:, ch], type)
            end
        end
        Y[i] = Xi
    end

    (Y_list=Y, t=t, y_true=y_true)
end

# -----------------------------
# Derivatives transform
# -----------------------------
function finite_differences(y, t)
    n = length(y)
    dy = zeros(n)
    dt = t[2] - t[1]  # Assuming uniform spacing
    
    # Forward difference for first point
    dy[1] = (y[2] - y[1]) / dt
    
    # Central differences for interior points
    for i in 2:n-1
        dy[i] = (y[i+1] - y[i-1]) / (2 * dt)
    end
    
    # Backward difference for last point
    dy[n] = (y[n] - y[n-1]) / dt
    
    return dy
end

function derivatives_transform(Y_list, t)
    N = length(Y_list)
    P = length(t)
    
    # Compute derivatives for each subject
    Y_deriv = Vector{Matrix{Float64}}(undef, N)
    
    for i in 1:N
        Yi = Y_list[i]
        M = size(Yi, 2)
        deriv_mat = zeros(P, 2 * M)  # Only original + 1st derivative
        
        for m in 1:M
            # Original signal
            deriv_mat[:, 2*(m-1) + 1] = Yi[:, m]
            
            # First derivative using finite differences
            deriv_mat[:, 2*(m-1) + 2] = finite_differences(Yi[:, m], t)
        end
        
        Y_deriv[i] = deriv_mat
    end
    
    Y_deriv
end

# -----------------------------
# Metrics
# -----------------------------
function metrics_from_preds(y_true, pred_anom)
    tn = sum((y_true .== 0) .& (pred_anom .== 0))
    fp = sum((y_true .== 0) .& (pred_anom .== 1))
    fn = sum((y_true .== 1) .& (pred_anom .== 0))
    tp = sum((y_true .== 1) .& (pred_anom .== 1))
    
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = (tp + fp) == 0 ? NaN : tp / (tp + fp)
    rec = (tp + fn) == 0 ? NaN : tp / (tp + fn)
    f1 = (isnan(prec) || isnan(rec) || (prec + rec) == 0) ? NaN : 2 * prec * rec / (prec + rec)
    
    (accuracy=acc, precision=prec, recall=rec, f1=f1)
end

# -----------------------------
# One dataset for one MC seed with automatic wavelet selection
# -----------------------------
function run_one(dataset_label, dataset_title, representation, make_data_fn, mc_idx)
    Random.seed!(base_seed + mc_idx)
    dat = make_data_fn()
    Y_list_raw = dat.Y_list
    t = dat.t
    y_true = dat.y_true
    N = length(Y_list_raw)

    # Transform
    Y_list = representation == "raw" ? Y_list_raw :
             representation == "derivatives" ? derivatives_transform(Y_list_raw, t) :
             throw(ArgumentError("Unknown representation: $representation"))

    # Clean
    for i in 1:length(Y_list)
        if any(.!isfinite.(Y_list[i]))
            @info "Non-finite values in $dataset_label subject $i – zeroing"
            Y_list[i][.!isfinite.(Y_list[i])] .= 0
        end
    end

    # Reveal 15% of normals
    normals_idx = findall(==(0), y_true)
    n_reveal = max(1, floor(Int, reveal_prop * length(normals_idx)))
    reveal_idx = normals_idx[randperm(length(normals_idx))[1:n_reveal]]

    # Run model with automatic wavelet selection
    res = try
        wicmad(Y_list, t;
               n_iter        = n_iter,
               burn          = burnin,
               thin          = thin,
               warmup_iters  = warmup_iters,
               revealed_idx  = reveal_idx,
               diagnostics   = true)
    catch e
        @warn "WICMAD error for $dataset_label: $e"
        (Z = ones(Int, 1, N), K_occ=[1], loglik=[0.0], alpha=[1.0], kern=[1],
         params=[], v=[0.5], pi=[1.0], diagnostics=nothing)
    end

    # Partition + predictions
    z_hat = res.Z[end, :]
    normal_label = begin
        lab_counts = countmap(z_hat[reveal_idx])
        isempty(lab_counts) ? 1 : first(keys(sort(collect(lab_counts), by = x -> x[2], rev = true)))
    end
    pred_anom = [z != normal_label ? 1 : 0 for z in z_hat]

    met = metrics_from_preds(y_true, pred_anom)
    (dataset=dataset_label, representation=representation, mc_run=mc_idx,
     accuracy=met.accuracy, precision=met.precision, recall=met.recall, f1=met.f1)
end

# -----------------------------
# Dataset registry
# -----------------------------
# Univariate datasets with multiple representations
univariate_specs = [
    (id="uni_isolated", title="Isolated", fn=() -> make_univariate_dataset(40, "isolated", t_grid)),
    (id="uni_mag1", title="Magnitude I", fn=() -> make_univariate_dataset(40, "mag1", t_grid)),
    (id="uni_mag2", title="Magnitude II", fn=() -> make_univariate_dataset(40, "mag2", t_grid)),
    (id="uni_shape", title="Shape", fn=() -> make_univariate_dataset(40, "shape", t_grid))
]

# Multivariate datasets (raw only)
multivariate_specs = [
    (id="mv_one_channel", title="One Channel", fn=() -> make_multivariate_dataset(40, "one", t_grid)),
    (id="mv_two_channels", title="Two Channels", fn=() -> make_multivariate_dataset(40, "two", t_grid)),
    (id="mv_three_channels", title="Three Channels", fn=() -> make_multivariate_dataset(40, "three", t_grid))
]

# Create expanded dataset specs with all representations for univariate datasets
dataset_specs = []

# Add univariate datasets with raw and derivatives representations
for spec in univariate_specs
    # Raw representation
    push!(dataset_specs, (id="$(spec.id)_raw", title=spec.title, representation="raw", fn=spec.fn))
    # Derivatives representation
    push!(dataset_specs, (id="$(spec.id)_deriv", title=spec.title, representation="derivatives", fn=spec.fn))
end

# Add multivariate datasets (raw only)
for spec in multivariate_specs
    push!(dataset_specs, (id=spec.id, title=spec.title, representation="raw", fn=spec.fn))
end

# -----------------------------
# MAIN: run all datasets across MC runs with automatic wavelet selection
# -----------------------------
all_results = []

println("\nStarting simulation study with automatic wavelet selection...")
println("Total datasets: $(length(dataset_specs))")
println("MC trials per dataset: $mc_runs")
println("Total experiments: $(length(dataset_specs) * mc_runs)")

for spec in dataset_specs
    id = spec.id
    title = spec.title
    representation = spec.representation
    
    println("\n[Dataset: $id] Testing $representation...")
    
    # Run MC trials for this dataset
    dataset_results = []
    
    # Use Threads.@threads for parallel execution
    Threads.@threads for mc_idx in 1:mc_runs
        result = run_one(id, title, representation, spec.fn, mc_idx)
        push!(dataset_results, result)
    end
    
    # Add results to overall results
    append!(all_results, dataset_results)
    
    # Calculate summary statistics for this dataset
    df_dataset = DataFrame(dataset_results)
    mean_f1 = mean(df_dataset.f1)
    mean_acc = mean(df_dataset.accuracy)
    mean_prec = mean(df_dataset.precision)
    mean_rec = mean(df_dataset.recall)
    
    println("  → $id ($representation): F1 = $(round(mean_f1, digits=4)), Acc = $(round(mean_acc, digits=4))")
end

# -----------------------------
# Aggregate results and save
# -----------------------------
df_results = DataFrame(all_results)

# Summary by dataset and representation
summary_tbl = combine(groupby(df_results, [:dataset, :representation]),
    :accuracy => mean => :accuracy,
    :precision => mean => :precision,
    :recall => mean => :recall,
    :f1 => mean => :f1,
    :accuracy => std => :accuracy_std,
    :precision => std => :precision_std,
    :recall => std => :recall_std,
    :f1 => std => :f1_std
)

# Save results
CSV.write(metrics_csv, summary_tbl)
println("\nSaved summary metrics CSV to: ", abspath(metrics_csv))

# Also save detailed results
detailed_csv = joinpath(out_root, "detailed_results.csv")
CSV.write(detailed_csv, summary_tbl)
println("Saved detailed results CSV to: ", abspath(detailed_csv))

perrun_csv = joinpath(out_root, "metrics_per_run.csv")
CSV.write(perrun_csv, df_results)
println("Saved per-run metrics CSV to: ", abspath(perrun_csv))

println("\nExperiment results saved in: ", abspath(out_root))

# Print summary
println("\n" * "="^80)
println("SIMULATION STUDY SUMMARY")
println("="^80)
for row in eachrow(summary_tbl)
    println("$(row.dataset) ($(row.representation)):")
    println("  F1: $(round(row.f1, digits=4)) ± $(round(row.f1_std, digits=4))")
    println("  Accuracy: $(round(row.accuracy, digits=4)) ± $(round(row.accuracy_std, digits=4))")
    println("  Precision: $(round(row.precision, digits=4)) ± $(round(row.precision_std, digits=4))")
    println("  Recall: $(round(row.recall, digits=4)) ± $(round(row.recall_std, digits=4))")
    println()
end

# Create performance summary text file
performance_summary_path = joinpath(out_root, "performance_summary.txt")
open(performance_summary_path, "w") do io
    println(io, "WICMAD SIMULATION STUDY PERFORMANCE SUMMARY")
    println(io, "="^60)
    println(io, "Generated: $(now())")
    println(io, "Total experiments: $(nrow(df_results))")
    println(io, "Datasets tested: $(length(unique(df_results.dataset)))")
    println(io, "MC trials per dataset: $mc_runs")
    println(io, "Using automatic wavelet selection")
    println(io, "")
    
    println(io, "PERFORMANCE BY DATASET:")
    println(io, "-"^40)
    for row in eachrow(summary_tbl)
        println(io, "$(row.dataset) ($(row.representation)):")
        println(io, "  F1 Score: $(round(row.f1, digits=4)) ± $(round(row.f1_std, digits=4))")
        println(io, "  Accuracy: $(round(row.accuracy, digits=4)) ± $(round(row.accuracy_std, digits=4))")
        println(io, "  Precision: $(round(row.precision, digits=4)) ± $(round(row.precision_std, digits=4))")
        println(io, "  Recall: $(round(row.recall, digits=4)) ± $(round(row.recall_std, digits=4))")
        println(io, "")
    end
    
    println(io, "OVERALL STATISTICS:")
    println(io, "-"^40)
    println(io, "Average F1 Score: $(round(mean(summary_tbl.f1), digits=4))")
    println(io, "Best F1 Score: $(round(maximum(summary_tbl.f1), digits=4))")
    println(io, "Worst F1 Score: $(round(minimum(summary_tbl.f1), digits=4))")
    println(io, "")
    
    println(io, "EXPERIMENT SETTINGS:")
    println(io, "-"^40)
    println(io, "Time points: $P_use")
    println(io, "MC runs: $mc_runs")
    println(io, "MCMC iterations: $n_iter")
    println(io, "Burn-in: $burnin")
    println(io, "Thin: $thin")
    println(io, "Warmup: $warmup_iters")
    println(io, "Reveal proportion: $reveal_prop")
    println(io, "Noise level: $sigma_noise")
    println(io, "Threads: $(Threads.nthreads())")
end

println("\nPerformance summary saved to: ", abspath(performance_summary_path))
println("\nSimulation study completed successfully!")
println("Results saved to: ", abspath(out_root))