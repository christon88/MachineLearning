# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Data.csv')
#dataset = dataset[, 2:3]

# Feature Scaling
#dataset[, 2:3] = scale(dataset[, 2:3])

# Splitting dataset
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)
