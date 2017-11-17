# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
no_vars = 3
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, no_vars].values


# Feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x[:, 3:] = sc_x.fit_transform(x[:, 1:2])"""

# Splitting data into taining and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


