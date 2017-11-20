# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('50_Startups.csv')


# Encoding categorical data
dataset$State = factor(dataset$State, 
                         levels = c('California', 'Florida', 'New York'),
                         labels = c(1, 2, 3))


# Feature Scaling
#dataset[, 2:3] = scale(dataset[, 2:3])

# Splitting dataset
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting regression model to training set
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset)

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = dataset)

regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = dataset)

regressor = lm(formula = Profit ~ R.D.Spend,
               data = dataset)

summary(regressor)

# Predicting test results
y_pred = predict(regressor, test_set)