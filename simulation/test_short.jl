#!/usr/bin/env julia

# Test script with very short runs: 10 MCMC iterations, 2 MC trials
# This tests that the MAP confusion matrix is being saved correctly

# Ensure we use the repository root project and have deps available
begin
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
end

using Base.Threads
@info "Available Julia threads: $(nthreads())"

using Random, LinearAlgebra, Statistics, Printf, Dates, DelimitedFiles
using StatsBase: countmap
using DataFrames, CSV
using Plots
using WICMAD

# Temporarily modify sim_core.jl by creating a modified version
# We'll read it, modify the constants, write to temp, include it, then delete temp
core_path = joinpath(@__DIR__, "sim_core.jl")
core_code = read(core_path, String)

# Replace the const declarations with shorter values for testing
core_code = replace(core_code, r"const mc_runs\s*=\s*\d+" => "const mc_runs = 2")
core_code = replace(core_code, r"const n_iter\s*=\s*[\d_]+" => "const n_iter = 10")
core_code = replace(core_code, r"const burnin\s*=\s*[\d_]+" => "const burnin = 2")
core_code = replace(core_code, r"const warmup_iters\s*=\s*[\d_]+" => "const warmup_iters = 1")

# Write to temporary file
temp_file = joinpath(@__DIR__, "sim_core_temp.jl")
write(temp_file, core_code)

# Include the temporary file
try
    include(temp_file)
finally
    # Clean up temp file
    if isfile(temp_file)
        rm(temp_file)
    end
end

println("\n" * "="^60)
println("Running short test: 10 MCMC iterations, 2 MC trials")
println("Output directory: $(abspath(out_root))")
println("="^60)

# Run test on just one simple dataset
run_datasets(["uni_isolated_raw"])

println("\n" * "="^60)
println("Test complete!")
println("Check outputs:")
println("  Summary metrics: $(abspath(metrics_csv))")
println("  Per-run metrics: $(abspath(perrun_csv))")

# Verify the output files exist and have the MAP confusion matrix columns
if isfile(metrics_csv)
    df_summary = CSV.read(metrics_csv, DataFrame)
    println("\nSummary metrics columns:")
    println(names(df_summary))
    if "mean_map_tn" in names(df_summary)
        println("\n✓ Mean MAP confusion matrix found in summary!")
        println(df_summary)
    else
        println("\n✗ WARNING: mean_map_tn not found in summary!")
    end
end

if isfile(perrun_csv)
    df_perrun = CSV.read(perrun_csv, DataFrame)
    println("\nPer-run metrics columns:")
    println(names(df_perrun))
    if "map_tn" in names(df_perrun)
        println("\n✓ MAP confusion matrix per run found!")
        println(first(df_perrun, 5))  # Show first 5 rows
    else
        println("\n✗ WARNING: map_tn not found in per-run metrics!")
    end
end
println("="^60)
