# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#Fix missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

#Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_x = LabelEncoder()
x[:, 0] = le_x.fit_transform(x[:, 0])
ohe = OneHotEncoder(categorical_features = [0])
x = ohe.fit_transform(x).toarray()

le_y = LabelEncoder()
y = le_y.fit_transform(y)