# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
no_vars = 1
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, no_vars].values


# Feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x[:, 3:] = sc_x.fit_transform(x[:, 1:2])"""

# Splitting data into taining and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

#Fitting Linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Predicting test set result
y_pred = regressor.predict(x_test)

#Visualizing the training results
plt.scatter(x_train, y_train, color ='red')
plt.plot(x_train, regressor.predict(x_train), color = 'green')
plt.show()
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_test, y_pred, color = 'blue')
