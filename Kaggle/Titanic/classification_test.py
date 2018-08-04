# Data Preprocessing Template

# Importing the libraries
import pandas as pd
#np.set_printoptions(threshold=np.inf)

#Importing the dataset
train = pd.read_csv('C:\\Users\\Eier\\Dropbox\\Christian\\MachineLearning\\Kaggle\\Titanic\\train.csv')
test = pd.read_csv('C:\\Users\\Eier\\Dropbox\\Christian\\MachineLearning\\Kaggle\\Titanic\\test.csv')

train = train.drop(labels = ['PassengerId'], axis = 1)

import re
deck = {"A": 1, "B":2, "C":3, "D":4, "E":5, "F":6, "G":7, "U":8}
data = [train, test]

for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int) 
    

# Taking care of missing data
for dataset in data:
    mean_age = dataset['Age'].mean()
    mean_fare = dataset['Fare'].mean()
    genders = {"male":0, "female":1}
    dataset['Age'] = dataset['Age'].fillna(mean_age)
    dataset['Fare'] = dataset['Fare'].fillna(mean_age)
    dataset['Embarked'] = dataset['Embarked'].fillna("S")
    dataset['Sex'] = dataset['Sex'].map(genders)

train = train.drop(['Name', 'Ticket', 'Cabin'], axis=1)
test = test.drop(['Name', 'Ticket', 'Cabin'], axis=1)

x = train.drop("Survived", axis = 1)
y = train["Survived"]
x_test = test.drop(labels = ['PassengerId'], axis = 1)


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_1 = LabelEncoder()
x.iloc[:, 6] = labelencoder_x_1.fit_transform(x.iloc[:, 6])
onehotencoder = OneHotEncoder(categorical_features = [6])
x = onehotencoder.fit_transform(x).toarray()
x = x[:, 1:]

labelencoder_x_2 = LabelEncoder()
x_test.iloc[:, 6] = labelencoder_x_2.fit_transform(x_test.iloc[:, 6])
onehotencoder = OneHotEncoder(categorical_features = [6])
x_test = onehotencoder.fit_transform(x_test).toarray()
x_test = x_test[:, 1:]


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x = sc_x.fit_transform(x)
sc_x_test = StandardScaler()
x_test = sc_x_test.fit_transform(x_test)

"""
# Splitting data into taining and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)
"""
x_train = x
y_train = y


# Building ANN
# Importing Keras
from keras.models import Sequential
from keras.layers import Dense

#Initializing ANN
classifier = Sequential()
classifier.add(Dense(activation="relu", input_dim=9, units=5, kernel_initializer="uniform"))
classifier.add(Dense(activation="relu", units=5, kernel_initializer="uniform"))
classifier.add(Dense(activation="relu", units=5, kernel_initializer="uniform"))
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

classifier.fit(x_train, y_train, epochs=50, batch_size=5)


# Predicting test set results
y_pred = classifier.predict(x_test) 
y_pred[y_pred > 0.5] = int(1)
y_pred[y_pred <= 0.5] = int(0)

"""
# Confusion matrix
from sklearn.metrics import confusion_matrix
c = confusion_matrix(y_test, y_pred)
"""

print(classifier.summary())

from keras.utils.vis_utils import plot_model
plot_model(classifier, show_shapes=True, show_layer_names=True)

submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred.ravel()
        })

    
submission.to_csv("Results.csv", index = False, float_format = '%d')