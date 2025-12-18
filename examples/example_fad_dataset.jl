#!/usr/bin/env julia

# Generate and plot an example Functional Anomaly Detection (FAD) dataset
# This script creates a publication-quality plot showing normal vs anomaly patterns

using Pkg
Pkg.activate(@__DIR__)

using Plots
using Random

# Set backend for PNG output
gr()  # GR backend provides good PNG support

Random.seed!(42)

# Generate synthetic functional data
P = 64  # Number of time points
t = collect(range(0, 2π, length=P))

# Normal class: smooth sine wave with small noise
n_normal = 85
normal_curves = []
for i in 1:n_normal
    # Base pattern: sine wave with slight variations
    # Add small random phase shift
    phase = randn() * 0.1
    curve = sin.(t .+ phase) .+ 0.3 .* sin.(2 .* (t .+ phase))
    # Add small noise
    noise = randn(P) * 0.1
    push!(normal_curves, curve .+ noise)
end

# Anomaly class: phase shift deviation from normal pattern
n_anomaly = 15
anomaly_curves = []
for i in 1:n_anomaly
    # Phase shift anomaly
    phase = randn() * 0.8 + π/4
    curve = sin.(t .+ phase) .+ 0.3 .* sin.(2 .* (t .+ phase))
    noise = randn(P) * 0.1
    push!(anomaly_curves, curve .+ noise)
end

# Create publication-quality plot
p = plot(
    size=(800, 500),
    dpi=300,
    xlabel="Time",
    ylabel="Value",
    title="Example FAD Dataset",
    titlefontsize=16,
    xguidefontsize=12,
    yguidefontsize=12,
    tickfontsize=10,
    legend=:topright,
    legendfontsize=11,
    grid=true,
    gridwidth=1,
    gridalpha=0.3,
    framestyle=:box
)

# Plot normal curves (blue, semi-transparent)
for curve in normal_curves
    plot!(p, t, curve, 
          color=:blue, 
          alpha=0.4, 
          linewidth=1.2,
          label="")
end

# Plot anomaly curves (red, semi-transparent)
for curve in anomaly_curves
    plot!(p, t, curve, 
          color=:red, 
          alpha=0.6, 
          linewidth=1.5,
          label="")
end

# Add legend entries (only once)
plot!(p, [NaN], [NaN], 
      color=:blue, 
      linewidth=2, 
      alpha=0.7,
      label="Normal")
plot!(p, [NaN], [NaN], 
      color=:red, 
      linewidth=2, 
      alpha=0.7,
      label="Anomaly")

# Save plot as PNG
plots_dir = joinpath(dirname(@__DIR__), "plots")
isdir(plots_dir) || mkpath(plots_dir)
output_path = joinpath(plots_dir, "example_fad_dataset.png")
savefig(p, output_path)
println("Plot saved as PNG to: ", output_path)

# Display plot
display(p)

