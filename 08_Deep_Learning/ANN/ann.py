# Artificial Neural Network

# Installing Tensorflow
# coda install tensorflow

# Installing Keras
# coda install keras

# Data Preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_1 = LabelEncoder()
x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])
labelencoder_x_2 = LabelEncoder()
x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()
x = x[:, 1:]

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x = sc_x.fit_transform(x)

# Splitting data into taining and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# Building ANN
# Importing Keras
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initializing ANN
classifier = Sequential()
classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

classifier.fit(x_train, y_train, epochs=40, batch_size=2)


# Predicting test set results
y_pred = classifier.predict(x_test) 
y_pred[y_pred > 0.5] = 1
y_pred[y_pred <= 0.5] = 0
# Confusion matrix
from sklearn.metrics import confusion_matrix
c = confusion_matrix(y_test, y_pred)

print(classifier.summary())

from keras.utils.vis_utils import plot_model
plot_model(classifier, show_shapes=True, show_layer_names=True)