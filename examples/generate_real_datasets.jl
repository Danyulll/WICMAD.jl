#!/usr/bin/env julia

# Generate and save real datasets as CSVs for R scripts
# Each dataset is saved in the same format as simulated datasets

using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=dirname(@__DIR__))

using WICMAD
using Random
using DataFrames
using CSV
using StatsBase: countmap, sample
using Interpolations

Random.seed!(42)

# Create output directory
output_dir = joinpath(dirname(@__DIR__), "datasets_for_other_method_runs")
isdir(output_dir) || mkpath(output_dir)

data_dir = joinpath(dirname(@__DIR__), "data")

println("Generating real datasets for R scripts...")
println("Output directory: $output_dir")

# Interpolate to power-of-two grid
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

# Process AsphaltRegularity dataset
println("\nProcessing AsphaltRegularity dataset...")
ds_asphalt = WICMAD.Utils.load_ucr_dataset("AsphaltRegularity/AsphaltRegularity", data_dir)
cm_asphalt = countmap(ds_asphalt.labels)
classes_sorted_asphalt = sort(collect(keys(cm_asphalt)); by = c -> -cm_asphalt[c])
normal_class_asphalt = classes_sorted_asphalt[1]
anomaly_class_asphalt = classes_sorted_asphalt[2]

normal_indices_asphalt = findall(==(normal_class_asphalt), ds_asphalt.labels)
anomaly_indices_asphalt = findall(==(anomaly_class_asphalt), ds_asphalt.labels)

p_anom = 0.15
ratio = p_anom / (1 - p_anom)
avail_norm_asphalt = length(normal_indices_asphalt)
avail_anom_asphalt = length(anomaly_indices_asphalt)

n_anom_max_by_norm_asphalt = floor(Int, ratio * avail_norm_asphalt)
n_anomaly_used_asphalt = min(avail_anom_asphalt, n_anom_max_by_norm_asphalt)
n_normal_used_asphalt = min(avail_norm_asphalt, round(Int, n_anomaly_used_asphalt * (1 - p_anom) / p_anom))

used_normal_indices_asphalt = sample(normal_indices_asphalt, n_normal_used_asphalt, replace=false)
used_anomaly_indices_asphalt = sample(anomaly_indices_asphalt, n_anomaly_used_asphalt, replace=false)
all_used_indices_asphalt = vcat(used_normal_indices_asphalt, used_anomaly_indices_asphalt)
shuffle!(all_used_indices_asphalt)

Y_full_asphalt = ds_asphalt.series[all_used_indices_asphalt]
y_full_asphalt = [ds_asphalt.labels[i] for i in all_used_indices_asphalt]

Y_interp_asphalt, P_asphalt = interpolate_to_length(Y_full_asphalt; target_len=32)

# Convert to matrix format (univariate)
X_asphalt = zeros(length(Y_interp_asphalt), P_asphalt)
for i in 1:length(Y_interp_asphalt)
    X_asphalt[i, :] = vec(Y_interp_asphalt[i])
end

# Create DataFrame
N_asphalt = size(X_asphalt, 1)
df_asphalt = DataFrame()
df_asphalt.mc_run = fill(1, N_asphalt)  # Single "run" for real data
df_asphalt.obs_id = 1:N_asphalt
df_asphalt.label = [yi == normal_class_asphalt ? 0 : 1 for yi in y_full_asphalt]

# Add time point columns
for p in 1:P_asphalt
    df_asphalt[!, Symbol("t_$p")] = X_asphalt[:, p]
end

# Save to CSV
filename_asphalt = joinpath(output_dir, "real_asphaltregularity.csv")
CSV.write(filename_asphalt, df_asphalt)
println("  ✓ Saved: real_asphaltregularity.csv ($N_asphalt observations, univariate)")

# Process CharacterTrajectories dataset
println("\nProcessing CharacterTrajectories dataset...")
ds_char = WICMAD.Utils.load_ucr_dataset("CharacterTrajectories/CharacterTrajectories", data_dir)
normal_class_char = "1"  # Corresponds to 'a'
cm_char = countmap(ds_char.labels)
classes_sorted_char = sort(collect(keys(cm_char)); by = c -> -cm_char[c])
other_classes_char = filter(c -> c != normal_class_char, classes_sorted_char)
anomaly_class_char = isempty(other_classes_char) ? error("No other classes found") : other_classes_char[1]

normal_indices_char = findall(==(normal_class_char), ds_char.labels)
anomaly_indices_char = findall(==(anomaly_class_char), ds_char.labels)

p_normal = 0.85
avail_norm_char = length(normal_indices_char)
avail_anom_char = length(anomaly_indices_char)

n_normal_used_char = avail_norm_char
n_anomaly_needed_char = round(Int, n_normal_used_char * p_anom / p_normal)
n_anomaly_used_char = min(avail_anom_char, n_anomaly_needed_char)

if n_anomaly_used_char < n_anomaly_needed_char
    n_normal_used_char = min(avail_norm_char, round(Int, n_anomaly_used_char * p_normal / p_anom))
end

used_normal_indices_char = sample(normal_indices_char, n_normal_used_char, replace=false)
used_anomaly_indices_char = sample(anomaly_indices_char, n_anomaly_used_char, replace=false)
all_used_indices_char = vcat(used_normal_indices_char, used_anomaly_indices_char)
shuffle!(all_used_indices_char)

Y_full_char = ds_char.series[all_used_indices_char]
y_full_char = [ds_char.labels[i] for i in all_used_indices_char]

Y_interp_char, P_char = interpolate_to_length(Y_full_char; target_len=32)

# Create DataFrame (multivariate)
N_char = length(Y_interp_char)
M_char = size(Y_interp_char[1], 2)
df_char = DataFrame()
df_char.mc_run = fill(1, N_char)
df_char.obs_id = 1:N_char
df_char.label = [yi == normal_class_char ? 0 : 1 for yi in y_full_char]

# Add time point columns for each channel
for m in 1:M_char
    for p in 1:P_char
        col_vals = [Y_interp_char[i][p, m] for i in 1:N_char]
        df_char[!, Symbol("ch$(m)_t_$p")] = col_vals
    end
end

# Save to CSV
filename_char = joinpath(output_dir, "real_charactertrajectories.csv")
CSV.write(filename_char, df_char)
println("  ✓ Saved: real_charactertrajectories.csv ($N_char observations, $M_char channels)")

# Process Chinatown dataset
println("\nProcessing Chinatown dataset...")
ds_chinatown = WICMAD.Utils.load_ucr_dataset("Chinatown/Chinatown", data_dir)
cm_chinatown = countmap(ds_chinatown.labels)
classes_sorted_chinatown = sort(collect(keys(cm_chinatown)); by = c -> -cm_chinatown[c])
normal_class_chinatown = classes_sorted_chinatown[1]
anomaly_class_chinatown = classes_sorted_chinatown[2]

normal_indices_chinatown = findall(==(normal_class_chinatown), ds_chinatown.labels)
anomaly_indices_chinatown = findall(==(anomaly_class_chinatown), ds_chinatown.labels)

avail_norm_chinatown = length(normal_indices_chinatown)
avail_anom_chinatown = length(anomaly_indices_chinatown)

n_anom_max_by_norm_chinatown = floor(Int, ratio * avail_norm_chinatown)
n_anomaly_used_chinatown = min(avail_anom_chinatown, n_anom_max_by_norm_chinatown)
n_normal_used_chinatown = min(avail_norm_chinatown, round(Int, n_anomaly_used_chinatown * (1 - p_anom) / p_anom))

used_normal_indices_chinatown = sample(normal_indices_chinatown, n_normal_used_chinatown, replace=false)
used_anomaly_indices_chinatown = sample(anomaly_indices_chinatown, n_anomaly_used_chinatown, replace=false)
all_used_indices_chinatown = vcat(used_normal_indices_chinatown, used_anomaly_indices_chinatown)
shuffle!(all_used_indices_chinatown)

Y_full_chinatown = ds_chinatown.series[all_used_indices_chinatown]
y_full_chinatown = [ds_chinatown.labels[i] for i in all_used_indices_chinatown]

Y_interp_chinatown, P_chinatown = interpolate_to_length(Y_full_chinatown; target_len=32)

# Convert to matrix format (univariate)
X_chinatown = zeros(length(Y_interp_chinatown), P_chinatown)
for i in 1:length(Y_interp_chinatown)
    X_chinatown[i, :] = vec(Y_interp_chinatown[i])
end

# Create DataFrame
N_chinatown = size(X_chinatown, 1)
df_chinatown = DataFrame()
df_chinatown.mc_run = fill(1, N_chinatown)
df_chinatown.obs_id = 1:N_chinatown
df_chinatown.label = [yi == normal_class_chinatown ? 0 : 1 for yi in y_full_chinatown]

# Add time point columns
for p in 1:P_chinatown
    df_chinatown[!, Symbol("t_$p")] = X_chinatown[:, p]
end

# Save to CSV
filename_chinatown = joinpath(output_dir, "real_chinatown.csv")
CSV.write(filename_chinatown, df_chinatown)
println("  ✓ Saved: real_chinatown.csv ($N_chinatown observations, univariate)")

println("\nDone. All real datasets exported to CSV format.")

