# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#np.set_printoptions(threshold=np.inf)

#Importing the dataset
train = pd.read_csv('C:\\Users\\Eier\\Dropbox\\Christian\\MachineLearning\\Kaggle\\Titanic\\train.csv')
test = pd.read_csv('C:\\Users\\Eier\\Dropbox\\Christian\\MachineLearning\\Kaggle\\Titanic\\test.csv')
x = train.iloc[:, [2, 4, 5, 6, 7, 9, 11]].values
y = train.iloc[:, 1].values
x_test = test.iloc[:, [1, 3, 4, 5, 6, 8, 10]].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:, 2:3])
x[:, 2:3] = imputer.transform(x[:, 2:3])
imputer = imputer.fit(x_test[:, 2:6])
x_test[:, 2:6] = imputer.transform(x_test[:, 2:6])

x[pd.isna(x[:, 6])]  = '9'


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
x[:, 1] = labelencoder_X.fit_transform(x[:, 1])
x[:, 6] = labelencoder_X.fit_transform(x[:, 6])
onehotencoder = OneHotEncoder(categorical_features = [1, 6])
x = onehotencoder.fit_transform(x).toarray()

labelencoder_X = LabelEncoder()
x_test[:, 1] = labelencoder_X.fit_transform(x_test[:, 1])
x_test[:, 6] = labelencoder_X.fit_transform(x_test[:, 6])
onehotencoder = OneHotEncoder(categorical_features = [1, 6])
x_test = onehotencoder.fit_transform(x_test).toarray()

x = x[:, [2, 5, 6, 7, 8, 9, 10, 11]]
x_test = x_test[:, 2:11]

"""
# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x = sc_x.fit_transform(x)
"""
# Splitting data into taining and test set
x_train = x
y_train = y
#y_test = test.iloc[:, 1].values

# Fitting logistic regression to training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)

# Predicting test set results
y_pred = classifier.predict(x_test) 
results = x_test = test.iloc[:, 0:2].values
results[:, 1] = y_pred
np.savetxt("results.csv", results, delimiter=",", fmt='%d')
