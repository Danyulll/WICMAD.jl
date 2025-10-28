module Plotting

using Plots
using Statistics: mean
using StatsBase: countmap
using ..WaveletOps: compute_mu_from_beta

export plot_dataset_before_clustering, plot_dataset_after_clustering, plot_clustering_comparison,
       plot_cluster_means_diagnostic, plot_kernel_switches_diagnostic, plot_wicmad_diagnostics

"""
    plot_dataset_before_clustering(series, labels, t; title="Dataset Before Clustering", save_path=nothing)

Plot the original dataset before clustering, showing all time series colored by their true labels.

# Arguments
- `series`: Vector of time series matrices
- `labels`: Vector of binary labels (0 for normal, 1 for anomaly)
- `t`: Time vector
- `title`: Plot title
- `save_path`: Optional path to save the plot

# Returns
- Plot object
"""
function plot_dataset_before_clustering(series, labels, t; title="Dataset Before Clustering", save_path=nothing)
    # Create subplots for each dimension
    n_dims = size(series[1], 2)
    plots = []
    
    for dim in 1:n_dims
        p = plot(title="Dimension $dim", xlabel="Time", ylabel="Value", 
                legend=:topright, size=(800, 400))
        
        # Plot normal samples (label = 0)
        normal_indices = findall(==(0), labels)
        for idx in normal_indices
            plot!(p, t, series[idx][:, dim], 
                  color=:blue, alpha=0.6, linewidth=1, 
                  label=idx == normal_indices[1] ? "Normal" : "")
        end
        
        # Plot anomaly samples (label = 1)
        anomaly_indices = findall(==(1), labels)
        for idx in anomaly_indices
            plot!(p, t, series[idx][:, dim], 
                  color=:red, alpha=0.8, linewidth=2, 
                  label=idx == anomaly_indices[1] ? "Anomaly" : "")
        end
        
        push!(plots, p)
    end
    
    # Combine plots
    if n_dims == 1
        final_plot = plots[1]
    else
        final_plot = plot(plots..., layout=(n_dims, 1), size=(800, 400*n_dims))
    end
    
    plot!(final_plot, title=title, plot_titlefontsize=14)
    
    if save_path !== nothing
        savefig(final_plot, save_path)
        println("Plot saved to: $save_path")
    end
    
    return final_plot
end

"""
    plot_dataset_after_clustering(series, clusters, t; title="Dataset After Clustering", save_path=nothing)

Plot the dataset after clustering, showing time series colored by their cluster assignments.

# Arguments
- `series`: Vector of time series matrices
- `clusters`: Vector of cluster assignments
- `t`: Time vector
- `title`: Plot title
- `save_path`: Optional path to save the plot

# Returns
- Plot object
"""
function plot_dataset_after_clustering(series, clusters, t; title="Dataset After Clustering", save_path=nothing)
    # Create subplots for each dimension
    n_dims = size(series[1], 2)
    plots = []
    
    # Get unique clusters and assign colors
    unique_clusters = sort(unique(clusters))
    colors = [:blue, :red, :green, :orange, :purple, :brown, :pink, :gray, :olive, :cyan]
    
    for dim in 1:n_dims
        p = plot(title="Dimension $dim", xlabel="Time", ylabel="Value", 
                legend=:topright, size=(800, 400))
        
        # Plot each cluster
        for (i, cluster) in enumerate(unique_clusters)
            cluster_indices = findall(==(cluster), clusters)
            color = colors[mod(i-1, length(colors)) + 1]
            
            for (j, idx) in enumerate(cluster_indices)
                plot!(p, t, series[idx][:, dim], 
                      color=color, alpha=0.6, linewidth=1,
                      label=j == 1 ? "Cluster $cluster" : "")
            end
        end
        
        push!(plots, p)
    end
    
    # Combine plots
    if n_dims == 1
        final_plot = plots[1]
    else
        final_plot = plot(plots..., layout=(n_dims, 1), size=(800, 400*n_dims))
    end
    
    plot!(final_plot, title=title, plot_titlefontsize=14)
    
    if save_path !== nothing
        savefig(final_plot, save_path)
        println("Plot saved to: $save_path")
    end
    
    return final_plot
end

"""
    plot_clustering_comparison(series, true_labels, clusters, t; save_path=nothing)

Create a side-by-side comparison of the dataset before and after clustering.

# Arguments
- `series`: Vector of time series matrices
- `true_labels`: Vector of true binary labels
- `clusters`: Vector of cluster assignments
- `t`: Time vector
- `save_path`: Optional path to save the plot

# Returns
- Plot object
"""
function plot_clustering_comparison(series, true_labels, clusters, t; save_path=nothing)
    n_dims = size(series[1], 2)
    
    # Create before and after plots for each dimension
    all_plots = []
    
    for dim in 1:n_dims
        # Before clustering plot
        p_before = plot(title="Before Clustering (Dim $dim)", xlabel="Time", ylabel="Value", 
                       legend=:topright, size=(400, 300))
        
        # Plot normal samples
        normal_indices = findall(==(0), true_labels)
        for idx in normal_indices
            plot!(p_before, t, series[idx][:, dim], 
                  color=:blue, alpha=0.6, linewidth=1, 
                  label=idx == normal_indices[1] ? "Normal" : "")
        end
        
        # Plot anomaly samples
        anomaly_indices = findall(==(1), true_labels)
        for idx in anomaly_indices
            plot!(p_before, t, series[idx][:, dim], 
                  color=:red, alpha=0.8, linewidth=2, 
                  label=idx == anomaly_indices[1] ? "Anomaly" : "")
        end
        
        # After clustering plot
        p_after = plot(title="After Clustering (Dim $dim)", xlabel="Time", ylabel="Value", 
                      legend=:topright, size=(400, 300))
        
        # Get unique clusters and assign colors
        unique_clusters = sort(unique(clusters))
        colors = [:blue, :red, :green, :orange, :purple, :brown, :pink, :gray, :olive, :cyan]
        
        for (i, cluster) in enumerate(unique_clusters)
            cluster_indices = findall(==(cluster), clusters)
            color = colors[mod(i-1, length(colors)) + 1]
            
            for (j, idx) in enumerate(cluster_indices)
                plot!(p_after, t, series[idx][:, dim], 
                      color=color, alpha=0.6, linewidth=1,
                      label=j == 1 ? "Cluster $cluster" : "")
            end
        end
        
        push!(all_plots, p_before, p_after)
    end
    
    # Create layout: 2 columns (before/after) × n_dims rows
    layout = @layout([grid(n_dims, 2)])
    final_plot = plot(all_plots..., layout=layout, size=(800, 300*n_dims))
    
    if save_path !== nothing
        savefig(final_plot, save_path)
        println("Comparison plot saved to: $save_path")
    end
    
    return final_plot
end

"""
    plot_clustering_summary(series, true_labels, clusters, t; save_dir=nothing)

Create a comprehensive summary of clustering results including before/after plots and statistics.

# Arguments
- `series`: Vector of time series matrices
- `true_labels`: Vector of true binary labels
- `clusters`: Vector of cluster assignments
- `t`: Time vector
- `save_dir`: Optional directory to save plots

# Returns
- Dictionary of plot objects
"""
function plot_clustering_summary(series, true_labels, clusters, t; save_dir=nothing)
    plots_dict = Dict()
    
    # Create plots directory if saving
    if save_dir !== nothing
        mkpath(save_dir)
    end
    
    # Plot 1: Before clustering
    before_path = save_dir !== nothing ? joinpath(save_dir, "dataset_before_clustering.png") : nothing
    plots_dict[:before] = plot_dataset_before_clustering(series, true_labels, t; 
                                                        title="Dataset Before Clustering",
                                                        save_path=before_path)
    
    # Plot 2: After clustering
    after_path = save_dir !== nothing ? joinpath(save_dir, "dataset_after_clustering.png") : nothing
    plots_dict[:after] = plot_dataset_after_clustering(series, clusters, t; 
                                                      title="Dataset After Clustering",
                                                      save_path=after_path)
    
    # Plot 3: Comparison
    comparison_path = save_dir !== nothing ? joinpath(save_dir, "clustering_comparison.png") : nothing
    plots_dict[:comparison] = plot_clustering_comparison(series, true_labels, clusters, t; 
                                                        save_path=comparison_path)
    
    # Print summary statistics
    println("\nClustering Summary:")
    println("==================")
    println("Total samples: $(length(series))")
    println("True anomalies: $(sum(true_labels))")
    println("True normal: $(sum(1 .- true_labels))")
    println("Number of clusters found: $(length(unique(clusters)))")
    println("Cluster sizes: $(countmap(clusters))")
    
    return plots_dict
end

"""
    plot_cluster_means_diagnostic(series, clusters, wicmad_result, t; wf="la8", J=nothing, boundary="periodic", save_path=nothing)

Plot cluster means before and after smoothing, showing the true empirical means vs WICMAD's smoothed estimates.

# Arguments
- `series`: Vector of time series matrices
- `clusters`: Vector of cluster assignments
- `wicmad_result`: WICMAD result object containing cluster parameters
- `t`: Time vector
- `wf`: Wavelet family used in WICMAD
- `J`: Wavelet decomposition level
- `boundary`: Boundary condition for wavelets
- `save_path`: Optional path to save the plot

# Returns
- Plot object
"""
function plot_cluster_means_diagnostic(series, clusters, wicmad_result, t; wf="la8", J=nothing, boundary="periodic", save_path=nothing)
    P = length(t)
    
    # Determine J if not provided
    if J === nothing
        J = floor(Int, log2(P))
    end
    
    # Ensure J is compatible with P (P must equal 2^J)
    if P != 2^J
        J = floor(Int, log2(P))
    end
    
    # Compute true cluster means from original data
    true_means = Dict{Int, Vector{Float64}}()
    for cluster_id in unique(clusters)
        cluster_indices = findall(==(cluster_id), clusters)
        cluster_data = [vec(series[i]) for i in cluster_indices]
        true_mean = mean(cluster_data)
        true_means[cluster_id] = true_mean
    end
    
    # Compute smoothed cluster means from WICMAD results
    smoothed_means = Dict{Int, Vector{Float64}}()
    for cluster_id in unique(clusters)
        cluster_indices = findall(==(cluster_id), clusters)
        if !isempty(cluster_indices) && cluster_id <= length(wicmad_result.params)
            # Compute smoothed mean using WICMAD's compute_mu_from_beta function
            beta_ch = wicmad_result.params[cluster_id].beta_ch
            mu_wave = compute_mu_from_beta(beta_ch, wf, J, boundary, P)
            smoothed_mean = vec(mu_wave)
            smoothed_means[cluster_id] = smoothed_mean
        end
    end
    
    # Create plot
    p = plot(title="Cluster Means: True vs Smoothed", xlabel="Time", ylabel="Value", 
             legend=:topright, size=(800, 500))
    
    # Plot true and smoothed means
    plot_colors = [:blue, :red, :green, :orange, :purple, :brown, :pink, :gray, :olive, :cyan]
    for (i, cluster_id) in enumerate(sort(collect(keys(true_means))))
        color = plot_colors[mod(i-1, length(plot_colors)) + 1]
        
        # Plot true mean
        if haskey(true_means, cluster_id)
            plot!(p, t, true_means[cluster_id], 
                  color=color, linestyle=:solid, linewidth=2, 
                  label="Cluster $cluster_id (True)", alpha=0.8)
        end
        
        # Plot smoothed mean
        if haskey(smoothed_means, cluster_id)
            plot!(p, t, smoothed_means[cluster_id], 
                  color=color, linestyle=:dash, linewidth=2, 
                  label="Cluster $cluster_id (Smoothed)", alpha=0.8)
        end
    end
    
    if save_path !== nothing
        savefig(p, save_path)
        println("Cluster means diagnostic plot saved to: $save_path")
    end
    
    return p
end

"""
    plot_kernel_switches_diagnostic(wicmad_result; save_path=nothing)

Plot kernel switches tracking throughout the MCMC iterations.

# Arguments
- `wicmad_result`: WICMAD result object containing kernel tracking information
- `save_path`: Optional path to save the plot

# Returns
- Plot object
"""
function plot_kernel_switches_diagnostic(wicmad_result; save_path=nothing)
    p = plot(title="Kernel Switches in ICM", xlabel="Iteration", ylabel="Kernel Index", 
             legend=:topright, size=(800, 400))
    
    # Extract kernel switches from results
    if !isempty(wicmad_result.kern)
        iterations = collect(1:length(wicmad_result.kern))
        plot!(p, iterations, wicmad_result.kern, 
              color=:purple, linewidth=1, alpha=0.7,
              label="Kernel Index")
        
        # Add statistics
        n_switches = sum(diff(wicmad_result.kern) .!= 0)
        println("Kernel switches detected: $n_switches")
        
        # Add annotation with switch count
        annotate!(p, 0.7, 0.9, text("Switches: $n_switches", 12, :right))
    else
        plot!(p, [], [], label="No kernel data available")
    end
    
    if save_path !== nothing
        savefig(p, save_path)
        println("Kernel switches diagnostic plot saved to: $save_path")
    end
    
    return p
end

"""
    plot_wicmad_diagnostics(series, clusters, wicmad_result, t; wf="la8", J=nothing, boundary="periodic", save_dir=nothing)

Create comprehensive diagnostic plots for WICMAD results including cluster means and kernel switches.

# Arguments
- `series`: Vector of time series matrices
- `clusters`: Vector of cluster assignments
- `wicmad_result`: WICMAD result object
- `t`: Time vector
- `wf`: Wavelet family used in WICMAD
- `J`: Wavelet decomposition level
- `boundary`: Boundary condition for wavelets
- `save_dir`: Optional directory to save plots

# Returns
- Dictionary of plot objects
"""
function plot_wicmad_diagnostics(series, clusters, wicmad_result, t; wf="la8", J=nothing, boundary="periodic", save_dir=nothing)
    plots_dict = Dict()
    
    # Create plots directory if saving
    if save_dir !== nothing
        mkpath(save_dir)
    end
    
    # Plot 1: Cluster means diagnostic
    means_path = save_dir !== nothing ? joinpath(save_dir, "cluster_means_diagnostic.png") : nothing
    plots_dict[:cluster_means] = plot_cluster_means_diagnostic(series, clusters, wicmad_result, t; 
                                                              wf=wf, J=J, boundary=boundary,
                                                              save_path=means_path)
    
    # Plot 2: Kernel switches diagnostic
    switches_path = save_dir !== nothing ? joinpath(save_dir, "kernel_switches_diagnostic.png") : nothing
    plots_dict[:kernel_switches] = plot_kernel_switches_diagnostic(wicmad_result; 
                                                                  save_path=switches_path)
    
    # Print diagnostic summary
    println("\nWICMAD Diagnostic Summary:")
    println("=========================")
    println("Number of clusters: $(length(unique(clusters)))")
    println("Total samples: $(length(series))")
    
    if !isempty(wicmad_result.kern)
        n_switches = sum(diff(wicmad_result.kern) .!= 0)
        println("Kernel switches: $n_switches")
        println("Final kernel index: $(wicmad_result.kern[end])")
    end
    
    if !isempty(wicmad_result.loglik)
        println("Final log-likelihood: $(round(wicmad_result.loglik[end], digits=2))")
    end
    
    return plots_dict
end

end # module
