# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
no_vars = 2
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, no_vars].values

#Fitting Linear Regression to dataset

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)
y_pred_l = lin_reg.predict([6.5])

#Fitting Polynomial Regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

#Visualizing linear results
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x))
plt.show()

#Visualizing polynomial results
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)))

lin_salary = lin_reg.predict(6.5)

poly_salary = lin_reg_2.predict(poly_reg.fit_transform(6.5))

