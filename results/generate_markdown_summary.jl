#!/usr/bin/env julia

# Script to generate markdown summary with bolding only on classification metrics
# (Precision, Recall, F1, Accuracy) - NOT on TP, TN, FP, FN

using CSV, DataFrames

project_root = dirname(@__DIR__)
results_dir = joinpath(project_root, "results")

# Dataset names in order
datasets = [
    "Univariate_Isolated_Spike",
    "Univariate_Magnitude_I",
    "Univariate_Magnitude_II",
    "Univariate_Shape",
    "Multivariate_One_Anomalous_Channel",
    "Multivariate_Two_Anomalous_Channels",
    "Multivariate_Three_Anomalous_Channels"
]

function format_value(val, is_best, is_classification_metric)
    str = isa(val, Real) ? (isinteger(val) ? string(Int(val)) : string(round(val, digits=4))) : string(val)
    if is_best && is_classification_metric
        return "**$str**"
    else
        return str
    end
end

function find_best_values(df, stat_type)
    # Only find best for classification metrics (precision, recall, f1, accuracy)
    best = Dict()
    for col in [:precision, :recall, :f1, :accuracy]
        best[col] = maximum(skipmissing(df[!, col]))
    end
    return best
end

function generate_table(df, stat_type)
    # Filter for the statistic type
    df_stat = filter(row -> row.statistic == stat_type, df)
    
    # Find best values (only for classification metrics)
    best = find_best_values(df_stat, stat_type)
    
    # Format method names
    df_stat = transform(df_stat, [:method, :variant] => ByRow((m, v) -> 
        if m == "WICMAD"
            "$m $v"
        else
            string(m)
        end) => :method_display)
    
    # Sort: WICMAD methods last, others alphabetically
    df_stat = sort(df_stat, :method_display, by=x -> (startswith(x, "WICMAD") ? 1 : 0, x))
    
    # Build table
    lines = String[]
    push!(lines, "| Method | Precision | Recall | F1 | Accuracy | TP | TN | FP | FN |")
    push!(lines, "|--------|-----------|--------|----|----------|----|----|----|----|")
    
    for row in eachrow(df_stat)
        prec_str = format_value(row.precision, row.precision == best[:precision], true)
        recall_str = format_value(row.recall, row.recall == best[:recall], true)
        f1_str = format_value(row.f1, row.f1 == best[:f1], true)
        acc_str = format_value(row.accuracy, row.accuracy == best[:accuracy], true)
        
        # TP, TN, FP, FN - no bolding (always format without bolding)
        tp_str = isinteger(row.tp) ? string(Int(row.tp)) : string(round(row.tp, digits=4))
        tn_str = isinteger(row.tn) ? string(Int(row.tn)) : string(round(row.tn, digits=4))
        fp_str = isinteger(row.fp) ? string(Int(row.fp)) : string(round(row.fp, digits=4))
        fn_str = isinteger(row.fn) ? string(Int(row.fn)) : string(round(row.fn, digits=4))
        
        push!(lines, "| $(row.method_display) | $prec_str | $recall_str | $f1_str | $acc_str | $tp_str | $tn_str | $fp_str | $fn_str |")
    end
    
    return join(lines, "\n")
end

# Generate markdown content
md_lines = String[]
push!(md_lines, "# Simulation Summary")
push!(md_lines, "")

for (idx, dataset) in enumerate(datasets)
    if idx > 1
        push!(md_lines, "<div style=\"page-break-after: always;\"></div>")
        push!(md_lines, "")
    end
    
    # Dataset title
    title = replace(dataset, "_" => " ")
    push!(md_lines, "## $title")
    push!(md_lines, "")
    
    # Image
    img_name = "$(dataset)_visualization.png"
    push!(md_lines, "![$title Dataset](dataset_visualizations/$img_name)")
    push!(md_lines, "")
    
    # Load CSV
    csv_path = joinpath(results_dir, "$(dataset)_summary.csv")
    if !isfile(csv_path)
        @warn "CSV file not found: $csv_path"
        continue
    end
    
    df = CSV.read(csv_path, DataFrame)
    
    # Mean Results
    push!(md_lines, "### Mean Results")
    push!(md_lines, "")
    push!(md_lines, generate_table(df, "mean"))
    push!(md_lines, "")
    
    # Median Results
    push!(md_lines, "### Median Results")
    push!(md_lines, "")
    push!(md_lines, generate_table(df, "median"))
    push!(md_lines, "")
end

# Write markdown file
md_path = joinpath(results_dir, "sim_results.md")
write(md_path, join(md_lines, "\n"))
println("Markdown summary generated: $md_path")

