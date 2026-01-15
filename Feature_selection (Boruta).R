setwd("XXXXX")#data_path
data <- read.csv("XXXXX", header = TRUE)#File_name
library(Boruta)
library(ggplot2)
library(tidyverse)
library(ranger)
library(ggridges)
library(viridis)
library(scales)
library(forcats)

# Define response variables and predictor variables
response_col <- ncol(data)
predictor_cols <- 1:(ncol(data) - 1)

# Run Boruta for feature selection
set.seed(1234)  # Set a random seed to ensure the repeatability of results
Var_Boruta <- Boruta(x = data[, predictor_cols], y = data[, response_col], 
                     maxRuns = 500, pValue = 0.05, getImp = getImpRfZ)
print(Var_Boruta)

# History of Feature Importance Extraction
imp_history <- as.data.frame(Var_Boruta$ImpHistory)
imp_history <- imp_history[, colSums(is.na(imp_history)) == 0]  # Remove NA column

# Final decision 
decision_all <- tibble(
  feature = names(Var_Boruta$finalDecision),
  decision = Var_Boruta$finalDecision
)

# Processing feature importance
data_long <- imp_history %>%
  pivot_longer(cols = everything(), names_to = "feature", values_to = "importance") %>%
  left_join(decision_all, by = "feature") %>%
  filter(is.finite(importance))

# Determine feature ranking
feature_order <- data_long %>%
  group_by(feature) %>%
  summarise(median_imp = median(importance, na.rm = TRUE)) %>%
  arrange(median_imp) %>%
  pull(feature)

# Save the feature importance map as a PDF
pdf("XXXXX", width = 10, height = 8)#File_name
ggplot(data_long, aes(x = importance, y = fct_relevel(feature, feature_order), fill = decision)) +
  geom_density_ridges_gradient(scale = 2, rel_min_height = 0.01, color = "black") +
  scale_fill_manual(values = c(
    "Confirmed" = alpha("#02BBC1", 0.6),
    "Tentative" = alpha("#FFC107", 0.6),
    "Rejected" = alpha("#E53935", 0.6),
    "Shadow" = alpha("#757575", 0.6)
  )) +
  labs(
    title = "Feature Importance Ridge Plot Based on Boruta",
    x = "Importance (Z-Score)",
    y = "Features"
  ) +
  theme_bw() +
  theme(
    legend.position = "right",
    panel.spacing = unit(0.1, "lines"),
    axis.text.y = element_text(size = 10)
  )
dev.off()
