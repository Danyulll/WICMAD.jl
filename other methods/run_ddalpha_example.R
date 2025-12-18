# R script using ddalpha library for functional data anomaly detection
# Runs on all univariate datasets across all MC trials
# Reports median classification metrics
# Documentation: https://cran.r-project.org/web/packages/ddalpha/ddalpha.pdf

cat("=== ddalpha Analysis Script ===\n")
cat("Script loaded, starting execution...\n\n")

# Load required libraries
if (!require("ddalpha")) {
  install.packages("ddalpha")
  library(ddalpha)
}

if (!require("readr")) {
  install.packages("readr")
  library(readr)
}

if (!require("dplyr")) {
  install.packages("dplyr")
  library(dplyr)
}

# Configuration
dataset_folder <- "datasets_for_other_method_runs"
depth_type <- "BD"  # Options: "BD" (band depth), "RP1" (random projection 1), "RP2" (random projection 2)

# Helper function: Convert matrix to ddalpha functional object format
# X: n x T (rows = curves, cols = time grid) for univariate
# OR X: n x T x p (observations x time x channels) for multivariate
# t_grid: length-T vector of time points
matrix_to_dataf <- function(X, t_grid, is_multivariate = FALSE) {
  if (is_multivariate) {
    # X is an array: n x T x p
    n <- dim(X)[1]
    Tt <- dim(X)[2]
    p <- dim(X)[3]
    
    dataf_list <- vector("list", n)
    for (i in 1:n) {
      # For multivariate, vals should be a matrix: T x p
      dataf_list[[i]] <- list(
        args = t_grid,
        vals = matrix(X[i,,], nrow = Tt, ncol = p)
      )
    }
  } else {
    # Univariate case
    X <- as.matrix(X)
    n <- nrow(X)
    
    dataf_list <- vector("list", n)
    for (i in 1:n) {
      dataf_list[[i]] <- list(
        args = t_grid,
        vals = X[i, ]
      )
    }
  }
  
  return(dataf_list)
}

# Helper function to detect if dataset is multivariate
detect_multivariate <- function(data_run) {
  # Check if there are channel columns (ch1_t_1, ch2_t_1, etc.)
  channel_cols <- grep("^ch[0-9]+_t_", names(data_run), value = TRUE)
  return(length(channel_cols) > 0)
}

# Helper function to extract channels and time points for multivariate data
extract_multivariate_data <- function(data_run) {
  # Find all unique channels
  all_cols <- names(data_run)
  channel_pattern <- "^ch([0-9]+)_t_([0-9]+)$"
  matching_cols <- grep(channel_pattern, all_cols, value = TRUE)
  
  if (length(matching_cols) == 0) {
    return(NULL)
  }
  
  # Extract channel numbers and time point numbers
  channel_nums <- unique(sub(channel_pattern, "\\1", matching_cols))
  time_nums <- unique(sub(channel_pattern, "\\2", matching_cols))
  time_nums <- time_nums[order(as.numeric(time_nums))]
  
  n_channels <- length(channel_nums)
  n_timepoints <- length(time_nums)
  n_observations <- nrow(data_run)
  
  # Create array: n x T x p (observations x time x channels)
  X_array <- array(NA, dim = c(n_observations, n_timepoints, n_channels))
  
  for (ch_idx in 1:n_channels) {
    ch_num <- channel_nums[ch_idx]
    ch_cols <- paste0("ch", ch_num, "_t_", time_nums)
    X_array[,,ch_idx] <- as.matrix(data_run[, ch_cols])
  }
  
  return(list(
    array = X_array,
    n_channels = n_channels,
    n_timepoints = n_timepoints,
    n_observations = n_observations
  ))
}

# Function to compute functional depth per curve
# Uses ddalpha's depthf.* functions
compute_fdepth <- function(dataf_list, depth_type = "BD") {
  if (depth_type == "BD") {
    d <- depthf.BD(dataf_list, dataf_list)
  } else if (depth_type == "RP1") {
    d <- depthf.RP1(dataf_list, dataf_list)
  } else if (depth_type == "RP2") {
    d <- depthf.RP2(dataf_list, dataf_list)
  } else {
    stop("Unknown depth_type. Use 'BD', 'RP1', or 'RP2'")
  }
  
  # depthf.* returns a numeric vector of depths (one per function)
  as.numeric(d)
}

# Function to get threshold for top 5% of scores
get_top_percent_threshold <- function(scores, top_percent = 5.0) {
  # Get threshold for top 5% (95th percentile)
  threshold <- quantile(scores, 1 - (top_percent / 100), type = 1)
  return(threshold)
}

# Function to compute classification metrics
compute_metrics <- function(true_labels, predicted_labels) {
  TP <- sum((predicted_labels == 1) & (true_labels == 1))
  FP <- sum((predicted_labels == 1) & (true_labels == 0))
  TN <- sum((predicted_labels == 0) & (true_labels == 0))
  FN <- sum((predicted_labels == 0) & (true_labels == 1))
  
  accuracy <- (TP + TN) / (TP + TN + FP + FN)
  precision <- ifelse(TP + FP > 0, TP / (TP + FP), 0)
  recall <- ifelse(TP + FN > 0, TP / (TP + FN), 0)
  f1_score <- ifelse(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)
  
  return(list(
    TP = TP, FP = FP, TN = TN, FN = FN,
    Accuracy = accuracy,
    Precision = precision,
    Recall = recall,
    F1_score = f1_score
  ))
}

# Function to process one MC run (handles both univariate and multivariate)
process_mc_run <- function(csv_file, mc_run, depth_type = "BD", verbose = FALSE) {
  if (verbose) {
    cat("  Processing MC run", mc_run, "...", sep="")
  }
  
  # Read CSV file
  data <- read_csv(csv_file, show_col_types = FALSE)
  
  # Filter for the specified MC run
  data_run <- data %>% filter(mc_run == !!mc_run)
  
  if (verbose) {
    cat(" Loaded", nrow(data_run), "observations")
  }
  
  # Extract true labels
  true_labels <- data_run$label
  n_anomalies <- sum(true_labels == 1)
  
  if (verbose) {
    cat(",", n_anomalies, "true anomalies")
  }
  
  # Check if multivariate
  is_multivariate <- detect_multivariate(data_run)
  
  if (is_multivariate) {
    # Process multivariate data
    mv_data <- extract_multivariate_data(data_run)
    if (is.null(mv_data)) {
      if (verbose) cat(" Error: Could not extract multivariate data\n")
      return(NULL)
    }
    
    X_array <- mv_data$array  # n x T x p
    n_channels <- mv_data$n_channels
    n_timepoints <- mv_data$n_timepoints
    t_grid <- seq(0, 1, length.out = n_timepoints)
    
    if (verbose) {
      cat(",", n_channels, "channels,", n_timepoints, "time points")
      cat(", computing functional depth (", depth_type, ", multivariate)...", sep="")
    }
  } else {
    # Process univariate data
    time_cols <- grep("^t_", names(data_run), value = TRUE)
    time_cols <- time_cols[order(as.numeric(sub("t_", "", time_cols)))]
    
    X_matrix <- as.matrix(data_run[, time_cols])
    n_timepoints <- length(time_cols)
    n_observations <- nrow(X_matrix)
    t_grid <- seq(0, 1, length.out = n_timepoints)
    
    if (verbose) {
      cat(",", n_timepoints, "time points, computing functional depth (", depth_type, ")...", sep="")
    }
  }
  
  # Run ddalpha functional depth computation
  tryCatch({
    # Convert to ddalpha format
    if (is_multivariate) {
      dataf_list <- matrix_to_dataf(X_array, t_grid, is_multivariate = TRUE)
    } else {
      dataf_list <- matrix_to_dataf(X_matrix, t_grid, is_multivariate = FALSE)
    }
    
    # Compute functional depth
    depth_vals <- compute_fdepth(dataf_list, depth_type = depth_type)
    
    # Convert depth to anomaly score (lower depth = higher anomaly score)
    # Use 1 - depth so higher score = more anomalous
    anomaly_scores <- 1 - depth_vals
    
    if (verbose) {
      cat(" depth computed, using top 5% threshold...")
    }
    
    # Use top 5% as anomalies
    threshold <- get_top_percent_threshold(anomaly_scores, top_percent = 5.0)
    predicted_labels <- as.integer(anomaly_scores >= threshold)
    
    # Compute metrics
    metrics <- compute_metrics(true_labels, predicted_labels)
    
    if (verbose) {
      cat(" Done\n")
      cat("    Threshold (top 5%):", round(threshold, 4),
          "| TP:", metrics$TP, "FP:", metrics$FP, "FN:", metrics$FN, "TN:", metrics$TN, "\n")
      cat("    Accuracy:", round(metrics$Accuracy, 4),
          "| Precision:", round(metrics$Precision, 4),
          "| Recall:", round(metrics$Recall, 4),
          "| F1:", round(metrics$F1_score, 4), "\n")
    }
    
    return(metrics)
    
  }, error = function(e) {
    if (verbose) {
      cat(" Error:", conditionMessage(e), "\n")
    }
    return(NULL)
  })
}

# Main function to process all datasets
process_all_datasets <- function(dataset_folder, depth_type = "BD", verbose = TRUE) {
  # Find all univariate CSV files
  if (!dir.exists(dataset_folder)) {
    stop(paste("Error: Folder", dataset_folder, "not found!"))
  }
  
  # Find both univariate and multivariate CSV files
  uni_files <- list.files(dataset_folder, pattern = "^uni_.*\\.csv$", full.names = TRUE)
  mv_files <- list.files(dataset_folder, pattern = "^mv_.*\\.csv$", full.names = TRUE)
  csv_files <- sort(c(uni_files, mv_files))
  
  if (length(csv_files) == 0) {
    stop(paste("No CSV files found in", dataset_folder))
  }
  
  cat("Found", length(uni_files), "univariate and", length(mv_files), "multivariate datasets\n")
  cat("Using", depth_type, "functional depth\n")
  cat("Using top 5% of anomaly scores as anomalies for each MC run\n")
  cat("Processing datasets:", paste(basename(csv_files), collapse=", "), "\n\n")
  
  results <- list()
  
  # Process each dataset
  for (csv_file in csv_files) {
    dataset_name <- basename(csv_file)
    dataset_name <- sub("\\.csv$", "", dataset_name)
    
    if (verbose) {
      cat("Processing", dataset_name, "...\n")
    }
    
    # Get all MC runs for this dataset
    data <- read_csv(csv_file, show_col_types = FALSE)
    mc_runs <- sort(unique(data$mc_run))
    n_runs <- length(mc_runs)
    
    if (verbose) {
      cat("  Found", n_runs, "MC runs\n")
    }
    
    # Process each MC run
    all_metrics <- list()
    successful_runs <- 0
    failed_runs <- 0
    
    for (mc_run in mc_runs) {
      metrics <- process_mc_run(csv_file, mc_run, depth_type = depth_type, verbose = verbose)
      if (!is.null(metrics)) {
        all_metrics[[length(all_metrics) + 1]] <- metrics
        successful_runs <- successful_runs + 1
      } else {
        failed_runs <- failed_runs + 1
        if (verbose) {
          cat("    MC run", mc_run, "failed\n")
        }
      }
    }
    
    if (verbose) {
      cat("  Summary:",
          successful_runs, "successful runs,",
          failed_runs, "failed runs\n")
    }
    
    if (length(all_metrics) > 0) {
      # Aggregate results across MC runs (using median)
      aggregated <- list(
        n_runs = length(all_metrics),
        Accuracy_median = median(sapply(all_metrics, function(m) m$Accuracy)),
        Accuracy_std = sd(sapply(all_metrics, function(m) m$Accuracy)),
        Precision_median = median(sapply(all_metrics, function(m) m$Precision)),
        Precision_std = sd(sapply(all_metrics, function(m) m$Precision)),
        Recall_median = median(sapply(all_metrics, function(m) m$Recall)),
        Recall_std = sd(sapply(all_metrics, function(m) m$Recall)),
        F1_score_median = median(sapply(all_metrics, function(m) m$F1_score)),
        F1_score_std = sd(sapply(all_metrics, function(m) m$F1_score))
      )
      
      results[[dataset_name]] <- list(
        aggregated = aggregated,
        per_run = all_metrics
      )
      
      if (verbose) {
        cat("  Completed", dataset_name, "\n")
        cat("  Median metrics: Accuracy=", round(aggregated$Accuracy_median, 3),
            " Precision=", round(aggregated$Precision_median, 3),
            " Recall=", round(aggregated$Recall_median, 3),
            " F1=", round(aggregated$F1_score_median, 3), "\n\n", sep="")
      }
    } else {
      if (verbose) {
        cat("  Warning: No successful runs for", dataset_name, "\n\n")
      }
    }
  }
  
  return(results)
}

# Function to print results
print_results <- function(results) {
  cat(paste(rep("=", 100), collapse=""), "\n")
  cat("ddalpha RESULTS SUMMARY (Median across MC runs)\n")
  cat(paste(rep("=", 100), collapse=""), "\n")
  cat(sprintf("%-20s %-10s %-20s %-20s %-20s %-20s\n",
              "Dataset", "MC Runs", "Accuracy", "Precision", "Recall", "F1-score"))
  cat(paste(rep("-", 100), collapse=""), "\n")
  
  for (dataset_name in names(results)) {
    agg <- results[[dataset_name]]$aggregated
    cat(sprintf("%-20s %-10d %-20s %-20s %-20s %-20s\n",
                dataset_name,
                agg$n_runs,
                paste0(sprintf("%.3f", agg$Accuracy_median), "±", sprintf("%.3f", agg$Accuracy_std)),
                paste0(sprintf("%.3f", agg$Precision_median), "±", sprintf("%.3f", agg$Precision_std)),
                paste0(sprintf("%.3f", agg$Recall_median), "±", sprintf("%.3f", agg$Recall_std)),
                paste0(sprintf("%.3f", agg$F1_score_median), "±", sprintf("%.3f", agg$F1_score_std))))
  }
  
  cat(paste(rep("=", 100), collapse=""), "\n")
  cat("\nDetailed per-run results:\n")
  cat(paste(rep("=", 100), collapse=""), "\n")
  
  for (dataset_name in names(results)) {
    cat("\n", dataset_name, ":\n", sep="")
    cat(sprintf("  %-6s %-12s %-12s %-12s %-12s\n",
                "Run", "Accuracy", "Precision", "Recall", "F1-score"))
    cat("  ", paste(rep("-", 60), collapse=""), "\n", sep="")
    for (i in seq_along(results[[dataset_name]]$per_run)) {
      m <- results[[dataset_name]]$per_run[[i]]
      cat(sprintf("  %-6d %-12.3f %-12.3f %-12.3f %-12.3f\n",
                  i, m$Accuracy, m$Precision, m$Recall, m$F1_score))
    }
  }
}

# Function to save results to CSV
save_results_to_csv <- function(results, output_file = "ddalpha_results.csv") {
  rows <- list()
  for (dataset_name in names(results)) {
    agg <- results[[dataset_name]]$aggregated
    rows[[length(rows) + 1]] <- data.frame(
      Dataset = dataset_name,
      MC_Runs = agg$n_runs,
      Accuracy_median = agg$Accuracy_median,
      Accuracy_std = agg$Accuracy_std,
      Precision_median = agg$Precision_median,
      Precision_std = agg$Precision_std,
      Recall_median = agg$Recall_median,
      Recall_std = agg$Recall_std,
      F1_score_median = agg$F1_score_median,
      F1_score_std = agg$F1_score_std
    )
  }
  
  df <- do.call(rbind, rows)
  write_csv(df, output_file)
  cat("\nResults saved to", output_file, "\n")
}

# Main execution
cat("\n=== Starting Main Execution ===\n")
cat("Dataset folder:", dataset_folder, "\n")
cat("Depth type:", depth_type, "\n\n")

# Run on all datasets
results <- tryCatch({
  process_all_datasets(dataset_folder, depth_type = depth_type, verbose = TRUE)
}, error = function(e) {
  cat("ERROR in process_all_datasets:", conditionMessage(e), "\n")
  cat("Stack trace:\n")
  print(traceback())
  return(list())
})

# Print results
cat("\n=== Final Results ===\n")
if (length(results) > 0) {
  print_results(results)
  save_results_to_csv(results)
  cat("\n=== Analysis Complete ===\n")
} else {
  cat("No results to report.\n")
  cat("Check for errors above.\n")
}

