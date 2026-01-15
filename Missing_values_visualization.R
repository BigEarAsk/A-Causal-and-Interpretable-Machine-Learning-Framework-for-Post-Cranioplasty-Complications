library(ggplot2)
library(dplyr)
library(tidyr)
library(showtext)

setwd("XXXXX")#data_path
datana <- read.csv("XXXXX")#File_name

# Calculating the Proportion of Missing Values for Each Variable
missing_data <- datana %>%
  summarise(across(everything(), ~sum(is.na(.)) / n())) %>%
  pivot_longer(cols = everything(), names_to = "Variables", values_to = "MissingProportion")

# Define the classification of missing values by scale
missing_data <- missing_data %>%
  mutate(Category = case_when(
    MissingProportion <= 0.05 ~ "Minor",
    MissingProportion <= 0.10 ~ "Moderate",
    TRUE ~ "High"
  ))

# Creating scaled missing value visualisations
plot <- ggplot(missing_data, aes(x = reorder(Variables, -MissingProportion), y = MissingProportion, fill = Category)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(
    title = "XXXXX",
    x = "Variables",
    y = "Proportion of Missing Values",
    fill = "Degree of Missingness"
  ) +
  geom_text(aes(label = scales::percent(MissingProportion, accuracy = 0.01)), 
            vjust = -0.5, color = "black", size = 4) +
  theme(
    plot.title = element_text(size = 18, face = "bold", hjust = 0.5, family = "Arial"),
    axis.title = element_text(size = 14, face = "bold", family = "Arial"),
    axis.text = element_text(size = 12, family = "Arial"),
    axis.text.x = element_text(angle = 45, hjust = 1, family = "Arial"),
    legend.position = "bottom",
    legend.title = element_text(size = 12, face = "bold", family = "Arial"),
    legend.text = element_text(size = 10, family = "Arial")
  )

# Save images as PDF
cairo_pdf("XXXXX", width = 16, height = 8)#File_name
print(plot)
dev.off()
