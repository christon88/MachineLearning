# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
no_vars = 4
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, no_vars].values

#Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_x = LabelEncoder()
x[:, 3] = le_x.fit_transform(x[:, 3])
ohe = OneHotEncoder(categorical_features = [3])
x = ohe.fit_transform(x).toarray()

le_y = LabelEncoder()
y = le_y.fit_transform(y)

# Feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x[:, 3:] = sc_x.fit_transform(x[:, 1:2])"""

# Splitting data into taining and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
