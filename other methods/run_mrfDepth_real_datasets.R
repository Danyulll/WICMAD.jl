# R script using mrfDepth library for functional data anomaly detection
# Runs on all real datasets
# Reports classification metrics and confusion matrix for each dataset
# Documentation: https://cran.r-project.org/web/packages/mrfDepth/mrfDepth.pdf

cat("=== mrfDepth Analysis Script (Real Datasets) ===\n")
cat("Script loaded, starting execution...\n\n")

# Load required libraries
if (!require("mrfDepth")) {
  install.packages("mrfDepth")
  library(mrfDepth)
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

# Function to find threshold that maximizes F1 score
find_best_f1_threshold <- function(scores, true_labels) {
  # Try all unique score values as thresholds
  unique_scores <- sort(unique(scores), decreasing = TRUE)
  
  best_f1 <- -Inf
  best_threshold <- unique_scores[1]
  best_metrics <- NULL
  
  for (threshold in unique_scores) {
    predicted_labels <- as.integer(scores >= threshold)
    metrics <- compute_metrics(true_labels, predicted_labels)
    
    if (metrics$F1_score > best_f1) {
      best_f1 <- metrics$F1_score
      best_threshold <- threshold
      best_metrics <- metrics
    }
  }
  
  return(list(threshold = best_threshold, f1 = best_f1, metrics = best_metrics))
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

# Function to print confusion matrix
print_confusion_matrix <- function(dataset_name, metrics) {
  cat("\n")
  cat(paste(rep("=", 80), collapse=""), "\n")
  cat("Confusion Matrix -", dataset_name, "\n")
  cat("(rows = true class, cols = predicted class; normal=0, anomaly=1)\n")
  cat(paste(rep("-", 80), collapse=""), "\n")
  cat(sprintf("%-15s %-15s %-15s\n", "", "Pred=0", "Pred=1"))
  cat(sprintf("%-15s %-15d %-15d\n", "True=0", metrics$TN, metrics$FP))
  cat(sprintf("%-15s %-15d %-15d\n", "True=1", metrics$FN, metrics$TP))
  cat(paste(rep("-", 80), collapse=""), "\n")
  cat(sprintf("N = %d   Accuracy = %.4f   Precision = %.4f   Recall = %.4f   F1 = %.4f\n",
              metrics$TP + metrics$TN + metrics$FP + metrics$FN,
              metrics$Accuracy, metrics$Precision, metrics$Recall, metrics$F1_score))
  cat(paste(rep("=", 80), collapse=""), "\n")
  cat("\n")
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
  
  # Create array: T x n x p (time x observations x channels) for mrfDepth
  X_array <- array(NA, dim = c(n_timepoints, n_observations, n_channels))
  
  for (ch_idx in 1:n_channels) {
    ch_num <- channel_nums[ch_idx]
    ch_cols <- paste0("ch", ch_num, "_t_", time_nums)
    X_array[,,ch_idx] <- t(as.matrix(data_run[, ch_cols]))
  }
  
  return(list(
    array = X_array,
    n_channels = n_channels,
    n_timepoints = n_timepoints,
    n_observations = n_observations
  ))
}

# Function to process one real dataset
process_real_dataset <- function(csv_file, verbose = TRUE) {
  dataset_name <- basename(csv_file)
  dataset_name <- sub("\\.csv$", "", dataset_name)
  
  if (verbose) {
    cat("\nProcessing", dataset_name, "...\n")
  }
  
  # Read CSV file
  data <- read_csv(csv_file, show_col_types = FALSE)
  
  # For real datasets, there's only one MC run (mc_run = 1)
  data_run <- data %>% filter(mc_run == 1)
  
  if (verbose) {
    cat("  Loaded", nrow(data_run), "observations")
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
    
    X_array <- mv_data$array
    n_channels <- mv_data$n_channels
    
    if (verbose) {
      cat(",", n_channels, "channels,", mv_data$n_timepoints, "time points")
      cat(", running fOutl (multivariate)...")
    }
  } else {
    # Process univariate data
    time_cols <- grep("^t_", names(data_run), value = TRUE)
    time_cols <- time_cols[order(as.numeric(sub("t_", "", time_cols)))]
    
    X_matrix <- as.matrix(data_run[, time_cols])
    n_timepoints <- length(time_cols)
    n_observations <- nrow(X_matrix)
    
    # Convert to mrfDepth format: t x n x p array
    X_array <- array(NA, dim = c(n_timepoints, n_observations, 1))
    X_array[,,1] <- t(X_matrix)
    
    if (verbose) {
      cat(",", n_timepoints, "time points, running fOutl...")
    }
  }
  
  # Run mrfDepth fOutl
  tryCatch({
    outlier_result <- fOutl(X_array, type = "fAO")
    
    # Extract outlyingness scores from $mean
    if ("mean" %in% names(outlier_result)) {
      outlyingness_scores <- outlier_result$mean
    } else {
      # Fallback
      numeric_components <- sapply(outlier_result, is.numeric)
      if (any(numeric_components)) {
        outlyingness_scores <- outlier_result[[which(numeric_components)[1]]]
      } else {
        stop("Could not extract scores from fOutl")
      }
    }
    
    if (verbose) {
      cat(" scores computed, finding threshold that maximizes F1...")
    }
    
    # Find threshold that maximizes F1 score
    best_result <- find_best_f1_threshold(outlyingness_scores, true_labels)
    threshold <- best_result$threshold
    predicted_labels <- as.integer(outlyingness_scores >= threshold)
    metrics <- best_result$metrics
    
    if (verbose) {
      cat(" Done\n")
    }
    
    # Print confusion matrix
    print_confusion_matrix(dataset_name, metrics)
    
    return(list(
      dataset_name = dataset_name,
      metrics = metrics,
      threshold = threshold
    ))
    
  }, error = function(e) {
    if (verbose) {
      cat(" Error:", conditionMessage(e), "\n")
    }
    return(NULL)
  })
}

# Main function to process all real datasets
process_all_real_datasets <- function(dataset_folder, verbose = TRUE) {
  if (!dir.exists(dataset_folder)) {
    stop(paste("Error: Folder", dataset_folder, "not found!"))
  }
  
  # Find all real dataset CSV files
  real_files <- list.files(dataset_folder, pattern = "^real_.*\\.csv$", full.names = TRUE)
  
  if (length(real_files) == 0) {
    stop(paste("No real dataset CSV files found in", dataset_folder))
  }
  
  cat("Found", length(real_files), "real dataset(s)\n")
  cat("Using threshold that maximizes F1 score\n")
  cat("Processing datasets:", paste(basename(real_files), collapse=", "), "\n")
  
  results <- list()
  
  # Process each dataset
  for (csv_file in real_files) {
    result <- process_real_dataset(csv_file, verbose = verbose)
    if (!is.null(result)) {
      results[[result$dataset_name]] <- result
    }
  }
  
  return(results)
}

# Function to print summary results
print_summary <- function(results) {
  cat("\n")
  cat(paste(rep("=", 100), collapse=""), "\n")
  cat("mrfDepth RESULTS SUMMARY (Real Datasets)\n")
  cat(paste(rep("=", 100), collapse=""), "\n")
  cat(sprintf("%-25s %-12s %-12s %-12s %-12s %-12s\n",
              "Dataset", "Accuracy", "Precision", "Recall", "F1-score", "Threshold"))
  cat(paste(rep("-", 100), collapse=""), "\n")
  
  for (dataset_name in names(results)) {
    m <- results[[dataset_name]]$metrics
    cat(sprintf("%-25s %-12.4f %-12.4f %-12.4f %-12.4f %-12.4f\n",
                dataset_name, m$Accuracy, m$Precision, m$Recall, m$F1_score, results[[dataset_name]]$threshold))
  }
  
  cat(paste(rep("=", 100), collapse=""), "\n")
}

# Function to save results to CSV
save_results_to_csv <- function(results, output_file = "mrfDepth_real_results.csv") {
  rows <- list()
  for (dataset_name in names(results)) {
    m <- results[[dataset_name]]$metrics
    rows[[length(rows) + 1]] <- data.frame(
      Dataset = dataset_name,
      TP = m$TP,
      TN = m$TN,
      FP = m$FP,
      FN = m$FN,
      Accuracy = m$Accuracy,
      Precision = m$Precision,
      Recall = m$Recall,
      F1_score = m$F1_score,
      Threshold = results[[dataset_name]]$threshold
    )
  }
  
  df <- do.call(rbind, rows)
  write_csv(df, output_file)
  cat("\nResults saved to", output_file, "\n")
}

# Main execution
cat("\n=== Starting Main Execution ===\n")
cat("Dataset folder:", dataset_folder, "\n\n")

# Run on all real datasets
results <- tryCatch({
  process_all_real_datasets(dataset_folder, verbose = TRUE)
}, error = function(e) {
  cat("ERROR in process_all_real_datasets:", conditionMessage(e), "\n")
  cat("Stack trace:\n")
  print(traceback())
  return(list())
})

# Print summary
if (length(results) > 0) {
  print_summary(results)
  save_results_to_csv(results)
  cat("\n=== Analysis Complete ===\n")
} else {
  cat("No results to report.\n")
  cat("Check for errors above.\n")
}

