library(MASS)  # linear regression imputation
library(missForest)  # missForest imputation
library(VIM)  # KNN imputation
library(mice)  # Bayesian regression imputation
library(caret)  # Split the dataset
setwd("XXXXX")#data_path
datana <- read.csv("XXXXX")#File_name
# NRMSE calculation function
calculate_nrmse <- function(original, imputed) {
  sigma <- sd(original, na.rm = TRUE)  
  if (sigma == 0) {
    return(NA)  
  }
  rmse <- sqrt(mean((original - imputed)^2, na.rm = TRUE))  # 计算 RMSE
  nrmse <- rmse / sigma  
  return(nrmse)
}

# 5-fold cross-validation to evaluate the NRMSE of linear regression imputation
evaluate_regression_imputation_cv <- function(data, target_column_name, cv = 5, missing_rate = 0.2) {
  nrmse_list <- c()  
  complete_data <- data[complete.cases(data), ]  
  folds <- createFolds(complete_data[[target_column_name]], k = cv, list = TRUE)
  for (i in 1:cv) {
    validation_indices <- folds[[i]]
    training_indices <- setdiff(1:nrow(complete_data), validation_indices)
    training_set <- complete_data[training_indices, ]  
    validation_set <- complete_data[validation_indices, ]  
    validation_with_na <- validation_set
    target_column <- validation_with_na[[target_column_name]]
    non_na_indices <- seq_along(target_column)  
    na_indices <- sample(non_na_indices, size = floor(length(non_na_indices) * missing_rate))  
    validation_with_na[[target_column_name]][na_indices] <- NA
    missing_indices <- which(is.na(validation_with_na[[target_column_name]]))
    non_missing_indices <- which(!is.na(validation_with_na[[target_column_name]]))
    if (length(missing_indices) > 0) {
      lm_model <- lm(as.formula(paste(target_column_name, "~ .")), data = validation_with_na[non_missing_indices, ])
      predicted_values <- predict(lm_model, newdata = validation_with_na[missing_indices, ])
      validation_with_na[[target_column_name]][missing_indices] <- predicted_values
    }
    
    imputed_target <- validation_with_na[[target_column_name]]
    original_values <- validation_set[[target_column_name]][na_indices] 
    imputed_values <- imputed_target[na_indices]
    nrmse <- calculate_nrmse(original_values, imputed_values)
    if (!is.na(nrmse)) {
      nrmse_list <- c(nrmse_list, nrmse)
    }
  }
  
  mean_nrmse <- mean(nrmse_list, na.rm = TRUE)
  return(mean_nrmse)
}
# Set the target column
target_column_name <- "XXXXX"

# NRMSE of linear regression imputation
set.seed(123)
nrmse_regression <- evaluate_regression_imputation_cv(datana, target_column_name = target_column_name, cv = 5, missing_rate = 0.2)
cat("NRMSE of linear regression imputation:", nrmse_regression, "\n")
# 5-fold cross-validation to evaluate the NRMSE of missForest imputation
evaluate_missforest_cv <- function(data, target_column_name, cv = 5, missing_rate = 0.2) {
  nrmse_list <- c() 
  complete_data <- data[complete.cases(data), ]  
  folds <- createFolds(complete_data[[target_column_name]], k = cv, list = TRUE)
  for (i in 1:cv) {
    validation_indices <- folds[[i]]
    training_indices <- setdiff(1:nrow(complete_data), validation_indices)
    training_set <- complete_data[training_indices, ] 
    validation_set <- complete_data[validation_indices, ] 
    validation_with_na <- validation_set
    target_column <- validation_with_na[[target_column_name]]
    non_na_indices <- seq_along(target_column)
    na_indices <- sample(non_na_indices, size = floor(length(non_na_indices) * missing_rate)) 
    validation_with_na[[target_column_name]][na_indices] <- NA
    imputed_data <- missForest(validation_with_na, verbose = FALSE)$ximp 
    imputed_target <- imputed_data[[target_column_name]]
    original_values <- validation_set[[target_column_name]][na_indices]
    imputed_values <- imputed_target[na_indices]
    nrmse <- calculate_nrmse(original_values, imputed_values)
    if (!is.na(nrmse)) {
      nrmse_list <- c(nrmse_list, nrmse)
    }
  }
  mean_nrmse <- mean(nrmse_list, na.rm = TRUE)
  return(mean_nrmse)
}
# NRMSE of missForest imputation
set.seed(123)
nrmse_missforest <- evaluate_missforest_cv(datana, target_column_name = target_column_name, cv = 5, missing_rate = 0.2)
cat("NRMSE of missForest imputation:", nrmse_missforest, "\n")
# Mean imputation function
mean_imputation <- function(column, training_set) {
  mean_value <- mean(training_set, na.rm = TRUE) 
  column_imputed <- ifelse(is.na(column), mean_value, column) 
  return(column_imputed)
}
# Median imputation function
median_imputation <- function(column, training_set) {
  median_value <- median(training_set, na.rm = TRUE)  
  column_imputed <- ifelse(is.na(column), median_value, column)  
  return(column_imputed)
}
# 5-fold cross-validation to evaluate the NRMSE of mean imputation and medican imputation
evaluate_imputation_cv <- function(column, method, k = 5, missing_rate = 0.2) {
  nrmse_list <- c()  
  complete_indices <- which(!is.na(column))  
  complete_data <- column[complete_indices]  
  folds <- createFolds(complete_data, k = k, list = TRUE)
  for (i in 1:k) {
    validation_indices <- folds[[i]]
    training_indices <- setdiff(1:length(complete_data), validation_indices)
    
    training_set <- complete_data[training_indices]  
    validation_set <- complete_data[validation_indices]  
    validation_with_na <- validation_set
    non_na_indices <- seq_along(validation_with_na)  
    na_indices <- sample(non_na_indices, size = floor(length(non_na_indices) * missing_rate))  
    validation_with_na[na_indices] <- NA
    if (method == "mean") {
      validation_imputed <- mean_imputation(validation_with_na, training_set)
    } else if (method == "median") {
      validation_imputed <- median_imputation(validation_with_na, training_set)
    } else {
      stop("Unsupported method. Choose from 'mean' or 'median'.")
    }
    original_values <- validation_set[na_indices]  
    imputed_values <- validation_imputed[na_indices]  
    nrmse <- calculate_nrmse(original_values, imputed_values)
    if (!is.na(nrmse)) {
      nrmse_list <- c(nrmse_list, nrmse)
    }
  }
  mean_nrmse <- mean(nrmse_list, na.rm = TRUE)
  return(mean_nrmse)
}
# NRMSE of mean imputation
set.seed(123)
nrmse_mean <- evaluate_imputation_cv(complete_skull_defect_area, method = "mean", k = 5, missing_rate = 0.2)
cat("NRMSE of mean imputation:", nrmse_mean, "\n")
# NRMSE of median imputation
nrmse_median <- evaluate_imputation_cv(complete_skull_defect_area, method = "median", k = 5, missing_rate = 0.2)
cat("NRMSE of mean imputation:", nrmse_median, "\n")

# 5-fold cross-validation to evaluate the NRMSE of KNN imputation
evaluate_knn_imputation_cv <- function(data, target_column_name, k = 5, cv = 5, missing_rate = 0.2) {
  nrmse_list <- c()  
  complete_data <- data[complete.cases(data), ]  
  folds <- createFolds(complete_data[[target_column_name]], k = cv, list = TRUE)
  for (i in 1:cv) {
    validation_indices <- folds[[i]]
    training_indices <- setdiff(1:nrow(complete_data), validation_indices)
    training_set <- complete_data[training_indices, ]  
    validation_set <- complete_data[validation_indices, ]  
    validation_with_na <- validation_set
    target_column <- validation_with_na[[target_column_name]]
    non_na_indices <- seq_along(target_column)  
    na_indices <- sample(non_na_indices, size = floor(length(non_na_indices) * missing_rate))  
    validation_with_na[[target_column_name]][na_indices] <- NA
    imputed_data <- kNN(validation_with_na, k = k, imp_var = FALSE)  
    imputed_target <- imputed_data[[target_column_name]]
    original_values <- validation_set[[target_column_name]][na_indices] 
    imputed_values <- imputed_target[na_indices]  
    nrmse <- calculate_nrmse(original_values, imputed_values)
    if (!is.na(nrmse)) {
      nrmse_list <- c(nrmse_list, nrmse)
    }
  }
  mean_nrmse <- mean(nrmse_list, na.rm = TRUE)
  return(mean_nrmse)
}
# NRMSE of KNN imputation
set.seed(123)
nrmse_knn <- evaluate_knn_imputation_cv(datana, target_column_name = target_column_name, k = 5, cv = 5, missing_rate = 0.2)
cat("NRMSE of KNN imputation:", nrmse_knn, "\n")
# 5-fold cross-validation to evaluate the NRMSE of Bayesian regression imputation
evaluate_bayesian_imputation_cv <- function(data, target_column_name, cv = 5, missing_rate = 0.2) {
  nrmse_list <- c()  
  complete_data <- data[complete.cases(data), ] 
  folds <- createFolds(complete_data[[target_column_name]], k = cv, list = TRUE)
  for (i in 1:cv) {
    validation_indices <- folds[[i]]
    training_indices <- setdiff(1:nrow(complete_data), validation_indices)
    training_set <- complete_data[training_indices, ]
    validation_set <- complete_data[validation_indices, ] 
    validation_with_na <- validation_set
    target_column <- validation_with_na[[target_column_name]]
    non_na_indices <- seq_along(target_column)
    na_indices <- sample(non_na_indices, size = floor(length(non_na_indices) * missing_rate)) 
    validation_with_na[[target_column_name]][na_indices] <- NA
    imputed_data <- mice(validation_with_na, m = 1, method = "norm.nob", maxit = 5, printFlag = FALSE) 
    completed_data <- complete(imputed_data) 
    imputed_target <- completed_data[[target_column_name]]
    original_values <- validation_set[[target_column_name]][na_indices]  
    imputed_values <- imputed_target[na_indices]  
    nrmse <- calculate_nrmse(original_values, imputed_values)
    if (!is.na(nrmse)) {
      nrmse_list <- c(nrmse_list, nrmse)
    }
  }
  mean_nrmse <- mean(nrmse_list, na.rm = TRUE)
  return(mean_nrmse)
}
#  NRMSE of Bayesian regression imputation
set.seed(123)
nrmse_bayesian <- evaluate_bayesian_imputation_cv(datana, target_column_name = target_column_name, cv = 5, missing_rate = 0.2)
cat("NRMSE of Bayesian regression imputation:", nrmse_bayesian, "\n")

# load data
categories <- c("BRIM", "LRIM", "MedIM", "MIM", "KNNIM", "mFIM")# different imputation method
values <- c(xx, xx, xx, xx, xx, xx)# NRMSE of different imputation method
# column chart
bar_heights <- barplot(
  values, 
  names.arg = categories, 
  main = "Bar Plot with NRMSE", 
  xlab = "Missing Value Imputation Methods", 
  ylab = "NRMSE", 
  col = "skyblue", 
  ylim = c(0, 1.6)
)
text(
  x = bar_heights, 
  y = values, 
  labels = round(values, 3), 
  pos = 3, 
  cex = 0.8, 
  col = "black"
)