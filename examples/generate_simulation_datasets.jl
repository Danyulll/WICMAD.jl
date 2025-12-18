#!/usr/bin/env julia

# Generate and save all simulation datasets for all Monte Carlo runs as CSVs
# Each dataset includes a column identifying the MC run

using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=dirname(@__DIR__))

# Include simulation core to get dataset generation functions
include(joinpath(dirname(@__DIR__), "src", "sim_core.jl"))

using Random
using DataFrames
using CSV
using Plots

# Set backend for PNG output
gr()

# Override MC runs to 100
const mc_runs = 100

# Create output directory
output_dir = joinpath(dirname(@__DIR__), "datasets_for_other_method_runs")
isdir(output_dir) || mkpath(output_dir)

println("Generating simulation datasets for $mc_runs Monte Carlo runs...")
println("Output directory: $output_dir")

# Get all dataset specs
all_specs = vcat(univariate_specs, multivariate_specs)

# Generate datasets for each dataset type (combining all MC runs)
for spec in all_specs
    println("\nGenerating: $(spec.id)")
    
    # Collect all MC runs for this dataset type
    all_dfs = DataFrame[]
    
    # Determine M (number of channels) from first dataset
    Random.seed!(base_seed)
    first_dataset = spec.fn()
    M = size(first_dataset.Y_list[1], 2)  # Number of dimensions/channels
    
    for mc_idx in 1:mc_runs
        # Set seed for this MC run (same as sim_core.jl)
        Random.seed!(base_seed + UInt64(mc_idx - 1))
        
        if mc_idx % 10 == 0
            println("  MC Run $mc_idx / $mc_runs")
        end
        
        # Generate dataset
        dataset = spec.fn()
        
        # Extract data
        Y_list = dataset.Y_list
        t = dataset.t
        y_true = dataset.y_true
        N = length(Y_list)
        P = length(t)
        
        # Create data matrix
        if M == 1
            # Univariate: N rows × P columns
            data_matrix = zeros(N, P)
            for i in 1:N
                data_matrix[i, :] = vec(Y_list[i])
            end
            
            # Create DataFrame with MC run, observation ID, label, and time points
            df = DataFrame()
            df.mc_run = fill(mc_idx, N)
            df.obs_id = 1:N
            df.label = y_true
            
            # Add time point columns
            for p in 1:P
                df[!, Symbol("t_$p")] = data_matrix[:, p]
            end
            
        else
            # Multivariate: N rows × (P*M) columns (flattened)
            data_matrix = zeros(N, P * M)
            for i in 1:N
                # Flatten each observation: [P×M] -> [P*M]
                data_matrix[i, :] = vec(Y_list[i])
            end
            
            # Create DataFrame with MC run, observation ID, label, and time points
            df = DataFrame()
            df.mc_run = fill(mc_idx, N)
            df.obs_id = 1:N
            df.label = y_true
            
            # Add time point columns for each channel
            for m in 1:M
                for p in 1:P
                    col_idx = (m - 1) * P + p
                    df[!, Symbol("ch$(m)_t_$p")] = data_matrix[:, col_idx]
                end
            end
        end
        
        push!(all_dfs, df)
    end
    
    # Combine all MC runs into single DataFrame
    combined_df = vcat(all_dfs...)
    
    # Save to CSV
    filename = joinpath(output_dir, "$(spec.id).csv")
    CSV.write(filename, combined_df)
    total_obs = nrow(combined_df)
    println("  ✓ Saved: $(spec.id).csv ($total_obs observations across $mc_runs MC runs, $M channel(s))")
    
    # Generate and save visualization for first MC run
    Random.seed!(base_seed)  # Use first MC run for visualization
    viz_dataset = spec.fn()
    viz_Y_list = viz_dataset.Y_list
    viz_t = viz_dataset.t
    viz_y_true = viz_dataset.y_true
    
    # Create plot
    if M == 1
        # Univariate plot
        p = plot(
            size=(800, 500),
            dpi=300,
            xlabel="Time",
            ylabel="Value",
            title="$(spec.title) - MC Run 1",
            titlefontsize=14,
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
        
        # Plot normal curves (blue)
        for i in 1:length(viz_Y_list)
            if viz_y_true[i] == 0
                plot!(p, viz_t, vec(viz_Y_list[i]), 
                      color=:blue, 
                      alpha=0.4, 
                      linewidth=1.2,
                      label="")
            end
        end
        
        # Plot anomaly curves (red)
        for i in 1:length(viz_Y_list)
            if viz_y_true[i] == 1
                plot!(p, viz_t, vec(viz_Y_list[i]), 
                      color=:red, 
                      alpha=0.6, 
                      linewidth=1.5,
                      label="")
            end
        end
        
        # Add legend entries
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
    else
        # Multivariate plot - stacked subplots
        p = plot(
            layout=(M, 1),
            size=(800, 300*M),
            dpi=300,
            xguidefontsize=12,
            yguidefontsize=12,
            tickfontsize=10,
            legendfontsize=11,
            grid=true,
            gridwidth=1,
            gridalpha=0.3,
            framestyle=:box
        )
        
        for m in 1:M
            # Plot normal curves (blue)
            for i in 1:length(viz_Y_list)
                if viz_y_true[i] == 0
                    plot!(p[m], viz_t, viz_Y_list[i][:, m], 
                          color=:blue, 
                          alpha=0.4, 
                          linewidth=1.2,
                          label="")
                end
            end
            
            # Plot anomaly curves (red)
            for i in 1:length(viz_Y_list)
                if viz_y_true[i] == 1
                    plot!(p[m], viz_t, viz_Y_list[i][:, m], 
                          color=:red, 
                          alpha=0.6, 
                          linewidth=1.5,
                          label="")
                end
            end
            
            # Add legend entries (only on first subplot)
            if m == 1
                plot!(p[m], [NaN], [NaN], 
                      color=:blue, 
                      linewidth=2, 
                      alpha=0.7,
                      label="Normal")
                plot!(p[m], [NaN], [NaN], 
                      color=:red, 
                      linewidth=2, 
                      alpha=0.7,
                      label="Anomaly")
            end
            
            plot!(p[m], 
                  title="$(spec.title) - Channel $m (MC Run 1)",
                  titlefontsize=12,
                  xlabel="Time",
                  ylabel="Value")
        end
    end
    
    # Save plot
    plot_filename = joinpath(output_dir, "$(spec.id)_visualization.png")
    savefig(p, plot_filename)
    println("  ✓ Saved visualization: $(spec.id)_visualization.png")
end

println("\n✓ All datasets generated and saved to: $output_dir")
println("Total files: $(length(all_specs))")

