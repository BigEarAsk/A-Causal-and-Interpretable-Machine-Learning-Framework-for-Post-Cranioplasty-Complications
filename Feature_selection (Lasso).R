setwd("XXXXX")#data_path
data <- read.csv("XXXXX.csv", header = TRUE)#File_name

library(glmnet)
library(ggplot2)

# Define response variables and predictor variables
response_col <- ncol(data)  
predictor_cols <- 1:(ncol(data) - 1) 
response <- data[, response_col]  
predictors <- as.matrix(data[, predictor_cols])  

# Feature selection using LASSO
set.seed(1234)  # Setting a random seed ensures the repeatability of results.
lasso_model <- cv.glmnet(
  predictors, response, 
  alpha = 1, 
  family = "binomial", 
  standardize = FALSE  
)

# Save the regression coefficient path plot
pdf("lasso_coefficient_path.pdf", width = 10, height = 8)
plot(lasso_model$glmnet.fit, xvar = "lambda", label = TRUE)
title("Coefficient Path Plot")
dev.off()

# Save the cross-validation error plot
pdf("lasso_cv_error_plot.pdf", width = 10, height = 8)
plot(lasso_model)
title("Cross-Validation Error Plot")
dev.off()

# Extracting the optimal lambda value
best_lambda_min <- lasso_model$lambda.min  #
cat("Optimal lambda value (minimum error):", best_lambda_min, "\n")

# Extract important features (features with non-zero coefficients, based on lambda.min).
lasso_coefficients_min <- coef(lasso_model, s = best_lambda_min)  # Coefficient extraction
important_features_min <- rownames(lasso_coefficients_min)[lasso_coefficients_min[, 1] != 0]  # Features with non-zero coefficients
important_features_min <- important_features_min[-1]  # Remove intercept term
cat("Important features (minimum error):\n", important_features_min, "\n")

# Save important features to a CSV file
write.csv(important_features_min, "important_features_lasso_min.csv", row.names = FALSE)