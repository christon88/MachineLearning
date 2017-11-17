# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Salary_Data.csv')
#dataset = dataset[, 2:3]

# Feature Scaling
#dataset[, 2:3] = scale(dataset[, 2:3])

# Splitting dataset
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Fitting Simple linear regression
regressor = lm(formula = Salary ~ YearsExperience, training_set)

#Predicting test results
y_pred = predict.lm(regressor, newdata = test_set)

#Visualizing training results
library(ggplot2)
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') + 
  geom_line(aes(x = training_set$YearsExperience, predict(regressor, newdata = training_set)),
            colour = 'blue')