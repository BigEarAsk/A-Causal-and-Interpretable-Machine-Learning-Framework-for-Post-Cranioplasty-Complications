library(ggplot2)
library(missForest)
library(readxl)
setwd("XXXXX")#data_path
datana <- read.csv("XXXXX")#file_name

#Select the column where the missing data is located
column_of_interest1 <- datana$"Before"

# Density map 
density_plot_before <- ggplot(datana, aes(x = column_of_interest1, fill = "Before Imputation")) +
  geom_density(alpha = 0.5) +
  labs(title = "Density Plot (Before and After Imputation)", fill = "") +
  theme_minimal()
column_of_interest2 <- datana$"After"
density_plot_combined <- ggplot(datana, aes(x = Before, fill = "Before Imputation")) +
  geom_density(alpha = 0.5) +
  geom_density(aes(x = After, fill = "After Imputation"), alpha = 0.5) +
  labs(title = "XXXXX Density Plot (Before and After Imputation)", fill = "") +
  theme_minimal()
density_plot_combined
#Boxplot map
boxplot_before <- ggplot(datana, aes(y = column_of_interest1, fill = "Before Imputation")) +
  geom_boxplot() +
  labs(title = "Boxplot (Before and After Imputation)", fill = "") +
  theme_minimal()
boxplot_combined <- ggplot(datana, aes(y = Before, fill = "Before Imputation")) +
  geom_boxplot(alpha = 0.5) +
  geom_boxplot(aes(y = After, fill = "After Imputation"), alpha = 0.5) +
  labs(title = "XXXXX Boxplot (Before and After Imputation)", fill = "") +
  theme_minimal()
print(boxplot_combined)
#scatter plot
combined_data <- data.frame(
  Skull.defect.area = c(datana$Before, datana$After),
  Imputation_Status = factor(rep(c("Before Imputation", "After Imputation"), each = nrow(datana)))
)
combined_data$Imputation_Status <- factor(combined_data$Imputation_Status, levels = c("Before Imputation", "After Imputation"))
combined_data$Imputation_Status <- as.character(combined_data$Imputation_Status)
scatter_plot <- ggplot(combined_data, aes(x = Imputation_Status, y = Skull.defect.area, color = Imputation_Status)) +
  geom_jitter(width = 0.2, alpha = 0.7) +  
  labs(title = "XXXXX Scatter Plot (Before and After Imputation)", x = "Imputation Status", y = "Skull.defect.area", color = "Imputation Status") +
  theme_minimal() +
  scale_color_manual(values = c("Before Imputation" = "blue", "After Imputation" = "red")) +
  theme(legend.position = "none")  
print(scatter_plot)

