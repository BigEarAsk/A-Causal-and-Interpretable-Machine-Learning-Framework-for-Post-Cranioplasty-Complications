install.packages("ggplot2")
install.packages("RColorBrewer")
library(ggplot2)
library(RColorBrewer)
setwd("XXXXX")#data_path
data <- data.frame(
  Category = c("GAM",
               "LR",
               "GBDT",
               "KNN",
               "AdaBoost",
               "LightGBM",
               "RotF",
               "RF",
               "XGBoost",
               "GPC",
               "ExtraTrees",
               "DT",
               "NB",
               "MLP",
               "SVM"),
  AB_score = c("xx")#calculated AB_score
)

# Convert AB_score to numeric type
data$AB_score <- as.numeric(data$AB_score)

p <- ggplot(data, aes(x = reorder(Category, AB_score), y = AB_score)) +
  geom_point(aes(size = AB_score, color = AB_score), alpha = 0.8) +
  scale_color_gradient(low = "lightblue", high = "darkblue", guide = guide_legend(reverse = TRUE)) +
  scale_size_continuous(guide = guide_legend(reverse = TRUE)) + 
  theme_minimal() +
  coord_flip() +
  labs(
    title = "Model performance of XXXXX",
    x = "Model",
    y = "AB_score"
  ) +
  geom_text(aes(label = round(AB_score, 4)), hjust = -0.3, color = "black", size = 4) +
  scale_y_continuous(limits = c(xx,xx))#Suitable values for the figure
print(p)
ggsave("XXXXX.pdf", plot = p, width = 6, height = 5)#File_name
