# -*- coding: utf-8 -*-

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Dataset.csv')
x = dataset.iloc[:, -1].values
y = dataset.iloc[:, 3].values

# Feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x[:, 3:] = sc_x.fit_transform(x[:, 1:2])
"""
# Splitting data into taining and test set
"""from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
"""

#Fitting Regression to dataset



#Predicting results with fitted model
y_pred = regressor.predict()

#Visualizing regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.show()

#Visualizing regression results (high resolution)
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.show()
