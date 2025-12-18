#!/usr/bin/env julia

using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=dirname(@__DIR__))

using WICMAD
using Random
using Statistics
using StatsBase: countmap, sample
using Interpolations
using Plots

Random.seed!(42)

data_dir = joinpath(dirname(@__DIR__), "data")
plots_dir = joinpath(dirname(@__DIR__), "plots")
isdir(plots_dir) || mkpath(plots_dir)

# Include simulation core to get dataset generation functions
include(joinpath(dirname(@__DIR__), "src", "sim_core.jl"))

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

# Function to convert reshaped arrays to matrices
function to_matrices(Y_list)
    return [Matrix(Y) for Y in Y_list]
end

# Function to interpolate to power-of-two grid
function interpolate_to_length(series::Vector{Matrix{Float64}}; target_len::Int = 32)
    out = Vector{Matrix{Float64}}(undef, length(series))
    for (i, X) in pairs(series)
        P0, M0 = size(X)
        x_old = collect(1:P0)
        x_new = collect(range(1, P0, length=target_len))
        Xn = Array{Float64}(undef, target_len, M0)
        for m in 1:M0
            itp = linear_interpolation(x_old, @view X[:, m])
            Xn[:, m] = itp.(x_new)
        end
        out[i] = Xn
    end
    return out, target_len
end

# Derivative utilities (from chinatown.jl)
function finite_diff_pad(v::AbstractVector{<:Real})
    d = diff(v)
    out = similar(v, length(v))
    out[1:end-1] = d
    out[end] = d[end]
    return out
end

function build_derivatives_matrix(Xm::AbstractMatrix{<:Real}; orders::Vector{Int})
    N, P = size(Xm)
    M = 1 + length(orders)
    Ymulti = Vector{Matrix{Float64}}(undef, N)
    for i in 1:N
        raw = Float64.(Xm[i, :])
        cols = Vector{Vector{Float64}}()
        push!(cols, raw)
        dprev = raw
        for ord in orders
            dprev = finite_diff_pad(dprev)
            push!(cols, dprev)
        end
        Yi = hcat(cols...)
        Ymulti[i] = Yi
    end
    return Ymulti
end

# Function to create publication-ready plot
function create_publication_plot(series::Vector{<:AbstractMatrix}, labels::Vector{Int}, 
                                 t::Vector, dataset_name::String, save_path::String;
                                 channel_titles::Union{Vector{String}, Nothing}=nothing)
    n_dims = size(series[1], 2)
    
    if n_dims == 1
        # Univariate: single plot
        p = plot(title=dataset_name, xlabel="Time", ylabel="Value", 
                legend=:topright, size=(1000, 600), 
                titlefontsize=16, xguidefontsize=12, yguidefontsize=12,
                legendfontsize=11)
        
        # Plot normal samples (label = 0)
        normal_indices = findall(==(0), labels)
        for (idx_count, idx) in enumerate(normal_indices)
            plot!(p, t, series[idx][:, 1], 
                  color=:blue, alpha=0.6, linewidth=1.5, 
                  label=idx_count == 1 ? "Normal" : "")
        end
        
        # Plot anomaly samples (label = 1)
        anomaly_indices = findall(==(1), labels)
        for (idx_count, idx) in enumerate(anomaly_indices)
            plot!(p, t, series[idx][:, 1], 
                  color=:red, alpha=0.3, linewidth=1.5, 
                  label=idx_count == 1 ? "Anomalous" : "")
        end
        
        savefig(p, save_path)
        println("Saved: $save_path")
        return p
    else
        # Multivariate: stacked subplots
        plt = plot(layout=(n_dims, 1), size=(1000, 400*n_dims), 
                  plot_title=dataset_name, plot_titlefontsize=16)
        
        # Channel titles: use provided titles, or default to "Raw", "Derivative 1", "Derivative 2", etc.
        if channel_titles === nothing
            channel_titles = ["Raw"; ["Derivative $(k)" for k in 1:(n_dims-1)]...]
        else
            length(channel_titles) == n_dims || error("Number of channel titles ($(length(channel_titles))) must match number of dimensions ($n_dims)")
        end
        
        for dim in 1:n_dims
            # Plot normal samples
            normal_indices = findall(==(0), labels)
            for (idx_count, idx) in enumerate(normal_indices)
                plot!(plt[dim], t, series[idx][:, dim], 
                      color=:blue, alpha=0.6, linewidth=1.5, 
                      label=idx_count == 1 ? "Normal" : "",
                      xlabel="Time", ylabel="Value")
            end
            
            # Plot anomaly samples
            anomaly_indices = findall(==(1), labels)
            for (idx_count, idx) in enumerate(anomaly_indices)
                plot!(plt[dim], t, series[idx][:, dim], 
                      color=:red, alpha=0.3, linewidth=1.5, 
                      label=idx_count == 1 ? "Anomalous" : "",
                      xlabel="Time", ylabel="Value")
            end
            
            plot!(plt[dim], title=channel_titles[dim], legend=:topright, legendfontsize=11,
                  xguidefontsize=11, yguidefontsize=11, titlefontsize=12)
        end
        
        savefig(plt, save_path)
        println("Saved: $save_path")
        return plt
    end
end

# ============================================================================
# REAL DATASETS
# ============================================================================

println("\n" * "="^70)
println("REAL DATASETS")
println("="^70)

# ----------------------------------------------------------------------------
# 1. CHINATOWN
# ----------------------------------------------------------------------------
println("\n" * "="^70)
println("Processing Chinatown dataset...")
println("="^70)

ds_chinatown = WICMAD.Utils.load_ucr_dataset("Chinatown/Chinatown", data_dir)
println("Chinatown dataset: $(length(ds_chinatown.series)) samples")
println("Class distribution: ", countmap(ds_chinatown.labels))

# Select classes: largest = normal, second largest = anomaly
cm_chinatown = countmap(ds_chinatown.labels)
classes_sorted_chinatown = sort(collect(keys(cm_chinatown)); by = c -> -cm_chinatown[c])
normal_class_chinatown = classes_sorted_chinatown[1]
anomaly_class_chinatown = classes_sorted_chinatown[2]

println("Normal class: $normal_class_chinatown (Weekday, $(cm_chinatown[normal_class_chinatown]) cases)")
println("Anomaly class: $anomaly_class_chinatown (Weekend, $(cm_chinatown[anomaly_class_chinatown]) cases)")

# Sample to achieve 15% anomalies
normal_indices_chinatown = findall(==(normal_class_chinatown), ds_chinatown.labels)
anomaly_indices_chinatown = findall(==(anomaly_class_chinatown), ds_chinatown.labels)

p_anom = 0.15
ratio = p_anom / (1 - p_anom)
avail_norm_chinatown = length(normal_indices_chinatown)
avail_anom_chinatown = length(anomaly_indices_chinatown)

n_anom_max_by_norm = floor(Int, ratio * avail_norm_chinatown)
n_anomaly_used_chinatown = min(avail_anom_chinatown, n_anom_max_by_norm)
n_normal_used_chinatown = min(avail_norm_chinatown, round(Int, n_anomaly_used_chinatown * (1 - p_anom) / p_anom))

used_normal_indices_chinatown = sample(normal_indices_chinatown, n_normal_used_chinatown, replace=false)
used_anomaly_indices_chinatown = sample(anomaly_indices_chinatown, n_anomaly_used_chinatown, replace=false)
all_used_indices_chinatown = vcat(used_normal_indices_chinatown, used_anomaly_indices_chinatown)
shuffle!(all_used_indices_chinatown)

# Subset data
Y_chinatown_subset = ds_chinatown.series[all_used_indices_chinatown]
y_chinatown_subset = [ds_chinatown.labels[i] for i in all_used_indices_chinatown]

# Create binary labels
gt_binary_chinatown = [label == normal_class_chinatown ? 0 : 1 for label in y_chinatown_subset]

println("Using N=$(length(Y_chinatown_subset)) with ~15% anomalies. Counts: ", countmap(y_chinatown_subset))
actual_anom_pct_chinatown = round(100 * count(==(anomaly_class_chinatown), y_chinatown_subset) / length(y_chinatown_subset), digits=1)
println("Actual anomaly percentage: $(actual_anom_pct_chinatown)%")

# Interpolate
Y_chinatown_interp, P_chinatown = interpolate_to_length(Y_chinatown_subset; target_len=32)
t_chinatown = collect(1:P_chinatown)

# Convert to matrix format for derivative computation
X_chinatown = zeros(length(Y_chinatown_interp), P_chinatown)
for i in 1:length(Y_chinatown_interp)
    X_chinatown[i, :] = vec(Y_chinatown_interp[i])  # Extract univariate series
end

# Build raw + first derivative matrix
Y_chinatown_d1 = build_derivatives_matrix(X_chinatown; orders=[1])

# Create plot with raw + first derivative
create_publication_plot(Y_chinatown_d1, gt_binary_chinatown, t_chinatown,
                       "Chinatown", 
                       joinpath(plots_dir, "chinatown_publication.png"))

# ----------------------------------------------------------------------------
# 2. ASPHALT REGULARITY
# ----------------------------------------------------------------------------
println("\n" * "="^70)
println("Processing Asphalt Regularity dataset...")
println("="^70)

ds_asphalt = WICMAD.Utils.load_ucr_dataset("AsphaltRegularity/AsphaltRegularity", data_dir)
println("Asphalt Regularity dataset: $(length(ds_asphalt.series)) samples")
println("Class distribution: ", countmap(ds_asphalt.labels))

# Select classes: largest = normal, second largest = anomaly
cm_asphalt = countmap(ds_asphalt.labels)
classes_sorted_asphalt = sort(collect(keys(cm_asphalt)); by = c -> -cm_asphalt[c])
normal_class_asphalt = classes_sorted_asphalt[1]
anomaly_class_asphalt = classes_sorted_asphalt[2]

println("Normal class: $normal_class_asphalt ($(cm_asphalt[normal_class_asphalt]) cases)")
println("Anomaly class: $anomaly_class_asphalt ($(cm_asphalt[anomaly_class_asphalt]) cases)")

# Sample to achieve 15% anomalies
normal_indices_asphalt = findall(==(normal_class_asphalt), ds_asphalt.labels)
anomaly_indices_asphalt = findall(==(anomaly_class_asphalt), ds_asphalt.labels)

p_anom = 0.15
ratio = p_anom / (1 - p_anom)
avail_norm_asphalt = length(normal_indices_asphalt)
avail_anom_asphalt = length(anomaly_indices_asphalt)

n_anom_max_by_norm = floor(Int, ratio * avail_norm_asphalt)
n_anomaly_used_asphalt = min(avail_anom_asphalt, n_anom_max_by_norm)
n_normal_used_asphalt = min(avail_norm_asphalt, round(Int, n_anomaly_used_asphalt * (1 - p_anom) / p_anom))

used_normal_indices_asphalt = sample(normal_indices_asphalt, n_normal_used_asphalt, replace=false)
used_anomaly_indices_asphalt = sample(anomaly_indices_asphalt, n_anomaly_used_asphalt, replace=false)
all_used_indices_asphalt = vcat(used_normal_indices_asphalt, used_anomaly_indices_asphalt)
shuffle!(all_used_indices_asphalt)

# Subset data
Y_asphalt_subset = ds_asphalt.series[all_used_indices_asphalt]
y_asphalt_subset = [ds_asphalt.labels[i] for i in all_used_indices_asphalt]

# Create binary labels
gt_binary_asphalt = [label == normal_class_asphalt ? 0 : 1 for label in y_asphalt_subset]

println("Using N=$(length(Y_asphalt_subset)) with ~15% anomalies. Counts: ", countmap(y_asphalt_subset))
actual_anom_pct_asphalt = round(100 * count(==(anomaly_class_asphalt), y_asphalt_subset) / length(y_asphalt_subset), digits=1)
println("Actual anomaly percentage: $(actual_anom_pct_asphalt)%")

# Interpolate
Y_asphalt_interp, P_asphalt = interpolate_to_length(Y_asphalt_subset; target_len=32)
t_asphalt = collect(1:P_asphalt)

# Create plot
create_publication_plot(Y_asphalt_interp, gt_binary_asphalt, t_asphalt,
                       "Asphalt Regularity", 
                       joinpath(plots_dir, "asphaltregularity_publication.png"))

# ----------------------------------------------------------------------------
# 3. CHARACTER TRAJECTORIES (classes 1 and 2 only)
# ----------------------------------------------------------------------------
println("\n" * "="^70)
println("Processing Character Trajectories dataset (classes 1 and 2 only)...")
println("="^70)

ds_char = WICMAD.Utils.load_ucr_dataset("CharacterTrajectories/CharacterTrajectories", data_dir)
println("Character Trajectories dataset: $(length(ds_char.series)) samples")
println("Class distribution: ", countmap(ds_char.labels))

# Filter to only classes 1 and 2
# Class 1 = 'a' (normal), Class 2 = anomaly (based on script logic)
normal_class_char = "1"
anomaly_class_char = "2"

normal_indices_char_full = findall(==(normal_class_char), ds_char.labels)
anomaly_indices_char_full = findall(==(anomaly_class_char), ds_char.labels)

println("Class 1 ('a'): $(length(normal_indices_char_full)) cases")
println("Class 2: $(length(anomaly_indices_char_full)) cases")

# Sample to achieve 15% anomalies
p_anom = 0.15
p_normal = 0.85
avail_norm_char = length(normal_indices_char_full)
avail_anom_char = length(anomaly_indices_char_full)

# Start with available normals and calculate how many anomalies we can use
n_normal_used_char = avail_norm_char  # Use all available normals
n_anomaly_needed = round(Int, n_normal_used_char * p_anom / p_normal)  # How many anomalies needed for 15%
n_anomaly_used_char = min(avail_anom_char, n_anomaly_needed)  # Use what's available, up to what's needed

# If we have more anomalies than needed, we might need to adjust normals
if n_anomaly_used_char < n_anomaly_needed
    # If we don't have enough anomalies, adjust normals to match available anomalies
    n_normal_used_char = min(avail_norm_char, round(Int, n_anomaly_used_char * p_normal / p_anom))
end

used_normal_indices_char = sample(normal_indices_char_full, n_normal_used_char, replace=false)
used_anomaly_indices_char = sample(anomaly_indices_char_full, n_anomaly_used_char, replace=false)
all_used_indices_char = vcat(used_normal_indices_char, used_anomaly_indices_char)
shuffle!(all_used_indices_char)

# Subset data
Y_char_subset = ds_char.series[all_used_indices_char]
labels_char_subset = [ds_char.labels[i] for i in all_used_indices_char]

# Create binary labels: class 1 = normal (0), class 2 = anomaly (1)
gt_binary_char = [label == normal_class_char ? 0 : 1 for label in labels_char_subset]

println("Using N=$(length(Y_char_subset)) with ~15% anomalies. Counts: ", countmap(labels_char_subset))
actual_anom_pct_char = round(100 * count(==(anomaly_class_char), labels_char_subset) / length(labels_char_subset), digits=1)
println("Actual anomaly percentage: $(actual_anom_pct_char)%")

# Interpolate
Y_char_interp, P_char = interpolate_to_length(Y_char_subset; target_len=32)
t_char = collect(1:P_char)

# Create plot with correct dimension labels (x, y, pen tip force)
create_publication_plot(Y_char_interp, gt_binary_char, t_char,
                       "Character Trajectories", 
                       joinpath(plots_dir, "charactertrajectories_publication.png");
                       channel_titles=["X-coordinate", "Y-coordinate", "Pen tip force"])

# ============================================================================
# SIMULATION DATASETS
# ============================================================================

println("\n" * "="^70)
println("SIMULATION DATASETS")
println("="^70)

# ----------------------------------------------------------------------------
# 1. UNIVARIATE ISOLATED RAW
# ----------------------------------------------------------------------------
println("\n" * "="^70)
println("Processing Univariate Isolated (Raw)...")
println("="^70)

dat_isolated = make_univariate_dataset(; anomaly_type=:isolated, t=t_grid)
Y_isolated = to_matrices(dat_isolated.Y_list)
y_isolated = dat_isolated.y_true
t_isolated = dat_isolated.t

println("Dataset: $(length(Y_isolated)) samples")
println("Anomaly percentage: $(round(100 * sum(y_isolated) / length(y_isolated), digits=1))%")

create_publication_plot(Y_isolated, y_isolated, t_isolated,
                       "Isolated", 
                       joinpath(plots_dir, "simulation_isolated_publication.png"))

# ----------------------------------------------------------------------------
# 2. UNIVARIATE MAGNITUDE I RAW
# ----------------------------------------------------------------------------
println("\n" * "="^70)
println("Processing Univariate Magnitude I (Raw)...")
println("="^70)

dat_mag1 = make_univariate_dataset(; anomaly_type=:mag1, t=t_grid)
Y_mag1 = to_matrices(dat_mag1.Y_list)
y_mag1 = dat_mag1.y_true
t_mag1 = dat_mag1.t

println("Dataset: $(length(Y_mag1)) samples")
println("Anomaly percentage: $(round(100 * sum(y_mag1) / length(y_mag1), digits=1))%")

create_publication_plot(Y_mag1, y_mag1, t_mag1,
                       "Magnitude I", 
                       joinpath(plots_dir, "simulation_magnitude1_publication.png"))

# ----------------------------------------------------------------------------
# 3. UNIVARIATE MAGNITUDE II RAW
# ----------------------------------------------------------------------------
println("\n" * "="^70)
println("Processing Univariate Magnitude II (Raw)...")
println("="^70)

dat_mag2 = make_univariate_dataset(; anomaly_type=:mag2, t=t_grid)
Y_mag2 = to_matrices(dat_mag2.Y_list)
y_mag2 = dat_mag2.y_true
t_mag2 = dat_mag2.t

println("Dataset: $(length(Y_mag2)) samples")
println("Anomaly percentage: $(round(100 * sum(y_mag2) / length(y_mag2), digits=1))%")

create_publication_plot(Y_mag2, y_mag2, t_mag2,
                       "Magnitude II", 
                       joinpath(plots_dir, "simulation_magnitude2_publication.png"))

# ----------------------------------------------------------------------------
# 4. UNIVARIATE SHAPE RAW
# ----------------------------------------------------------------------------
println("\n" * "="^70)
println("Processing Univariate Shape (Raw)...")
println("="^70)

dat_shape = make_univariate_dataset(; anomaly_type=:shape, t=t_grid)
Y_shape = to_matrices(dat_shape.Y_list)
y_shape = dat_shape.y_true
t_shape = dat_shape.t

println("Dataset: $(length(Y_shape)) samples")
println("Anomaly percentage: $(round(100 * sum(y_shape) / length(y_shape), digits=1))%")

create_publication_plot(Y_shape, y_shape, t_shape,
                       "Shape", 
                       joinpath(plots_dir, "simulation_shape_publication.png"))

# ----------------------------------------------------------------------------
# 5. MULTIVARIATE ONE CHANNEL
# ----------------------------------------------------------------------------
println("\n" * "="^70)
println("Processing Multivariate One Channel...")
println("="^70)

dat_one = make_multivariate_dataset(; regime=:one, t=t_grid)
Y_one = to_matrices(dat_one.Y_list)
y_one = dat_one.y_true
t_one = dat_one.t

println("Dataset: $(length(Y_one)) samples")
println("Anomaly percentage: $(round(100 * sum(y_one) / length(y_one), digits=1))%")

create_publication_plot(Y_one, y_one, t_one,
                       "One Anomalous Dimension", 
                       joinpath(plots_dir, "simulation_one_channel_publication.png"))

# ----------------------------------------------------------------------------
# 6. MULTIVARIATE TWO CHANNELS
# ----------------------------------------------------------------------------
println("\n" * "="^70)
println("Processing Multivariate Two Channels...")
println("="^70)

dat_two = make_multivariate_dataset(; regime=:two, t=t_grid)
Y_two = to_matrices(dat_two.Y_list)
y_two = dat_two.y_true
t_two = dat_two.t

println("Dataset: $(length(Y_two)) samples")
println("Anomaly percentage: $(round(100 * sum(y_two) / length(y_two), digits=1))%")

create_publication_plot(Y_two, y_two, t_two,
                       "Two Anomalous Dimensions", 
                       joinpath(plots_dir, "simulation_two_channels_publication.png"))

# ----------------------------------------------------------------------------
# 7. MULTIVARIATE THREE CHANNELS
# ----------------------------------------------------------------------------
println("\n" * "="^70)
println("Processing Multivariate Three Channels...")
println("="^70)

dat_three = make_multivariate_dataset(; regime=:three, t=t_grid)
Y_three = to_matrices(dat_three.Y_list)
y_three = dat_three.y_true
t_three = dat_three.t

println("Dataset: $(length(Y_three)) samples")
println("Anomaly percentage: $(round(100 * sum(y_three) / length(y_three), digits=1))%")

create_publication_plot(Y_three, y_three, t_three,
                       "Three Anomalous Dimensions", 
                       joinpath(plots_dir, "simulation_three_channels_publication.png"))

# ============================================================================
# SUMMARY
# ============================================================================

println("\n" * "="^70)
println("All publication plots created successfully!")
println("="^70)
println("\nReal dataset plots:")
println("  - $(joinpath(plots_dir, "chinatown_publication.png"))")
println("  - $(joinpath(plots_dir, "asphaltregularity_publication.png"))")
println("  - $(joinpath(plots_dir, "charactertrajectories_publication.png"))")
println("\nSimulation dataset plots:")
println("  - $(joinpath(plots_dir, "simulation_isolated_publication.png"))")
println("  - $(joinpath(plots_dir, "simulation_magnitude1_publication.png"))")
println("  - $(joinpath(plots_dir, "simulation_magnitude2_publication.png"))")
println("  - $(joinpath(plots_dir, "simulation_shape_publication.png"))")
println("  - $(joinpath(plots_dir, "simulation_one_channel_publication.png"))")
println("  - $(joinpath(plots_dir, "simulation_two_channels_publication.png"))")
println("  - $(joinpath(plots_dir, "simulation_three_channels_publication.png"))")

