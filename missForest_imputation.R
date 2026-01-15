install.packages("readxl")
install.packages("ggplot2")
install.packages("missForest")
library(ggplot2)
library(missForest)
library(readxl)
setwd("XXXXX")#data_path

datana <- read.csv("XXXXX")#File_name

#set seed
set.seed(123)
#imputation
imputed_data <- missForest(datana,xtrue = datana, verbose = TRUE)
imputed_data$OOBerror
imputed_data$error
#Extract fill data from missForest objects
imputed_data <- imputed_data$ximp
write.csv(imputed_data,"XXXXX")# File_name