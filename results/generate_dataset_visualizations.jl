#!/usr/bin/env julia

# Script to generate visualization plots for all simulation datasets
# Saves PNG files that will be included in the markdown summary

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using WICMAD, Random, Statistics, Printf
using Plots

gr()
include(joinpath(@__DIR__, "..", "src", "sim_core.jl"))
project_root = dirname(@__DIR__)

# Helper function to plot datasets (same as in temp scripts)
function plot_dataset(Y_list, y_true, t, title_str; save_path=nothing)
    M = size(Y_list[1], 2)
    plt = plot(layout=(M,1), size=(1000, 300*M))
    for m in 1:M
        for i in 1:length(Y_list)
            color, alpha_val = y_true[i] == 0 ? (:blue, 0.4) : (:red, 0.6)
            plot!(plt[m], t, Y_list[i][:, m], color=color, alpha=alpha_val, linewidth=1, label="")
        end
        # Only show "Channel X" title for multivariate datasets (M > 1)
        channel_title = M > 1 ? "Channel $m" : ""
        plot!(plt[m], title=channel_title, xlabel="Time", ylabel="Value")
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

# Create output directory for visualizations
viz_dir = joinpath(project_root, "results", "dataset_visualizations")
mkpath(viz_dir)

const base_seed_val = 0x000000000135CFF1

# Dataset configurations
datasets = [
    ("Univariate_Isolated_Spike", :isolated, false, nothing),
    ("Univariate_Magnitude_I", :mag1, false, nothing),
    ("Univariate_Magnitude_II", :mag2, false, nothing),
    ("Univariate_Shape", :shape, false, nothing),
    ("Multivariate_One_Anomalous_Channel", nothing, true, :one),
    ("Multivariate_Two_Anomalous_Channels", nothing, true, :two),
    ("Multivariate_Three_Anomalous_Channels", nothing, true, :three),
]

println("Generating dataset visualizations...")
println("="^70)

for (dataset_name, anomaly_type, is_multivariate, regime) in datasets
    println("\nGenerating visualization for: $dataset_name")
    
    # Generate dataset with fixed seed for reproducibility
    Random.seed!(base_seed_val)
    if is_multivariate
        data = make_multivariate_dataset(; regime=regime, t=t_grid)
    else
        data = make_univariate_dataset(; anomaly_type=anomaly_type, t=t_grid)
    end
    
    # Create plot
    plot_path = joinpath(viz_dir, "$(dataset_name)_visualization.png")
    plot_dataset(data.Y_list, data.y_true, data.t, "$dataset_name - Ground Truth"; save_path=plot_path)
end

println("\n" * "="^70)
println("All visualizations generated successfully!")
println("Output directory: $viz_dir")

