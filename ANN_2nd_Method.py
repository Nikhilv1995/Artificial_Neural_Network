# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 11:20:25 2023

@author: nikhilve
"""

#importing libraries
import pandas as pd
import numpy as np

##Data pre processing
#Importing Data Set

dataset=pd.read_csv("C:/Users/NIKHILVE/ML_ANN.CG/ML_Projects/ANN_Projects/Artificial_Neural_Networks_Project/Churn_Modelling.csv")

#deciding matrix of IV(X) and vector of DV(y).
X=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13:].values

#Encoding the text or categorical data in columns.
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

#making an object of label encoder and performing encoding for 1st column.
labelencoder_X_1=LabelEncoder()
X[:,1]= labelencoder_X_1.fit_transform(X[:,1])

#making an object of label encoder and performing encoding for 2nd column.
labelencoder_X_2=LabelEncoder()
X[:,2]= labelencoder_X_1.fit_transform(X[:,2])

# Perform one-hot encoding
onehotencoder = OneHotEncoder(drop='first', sparse_output=False)
X_encoded = onehotencoder.fit_transform(X[:, [1]])
#X_endoded contains one hot encoded values and also deals with dummy variable trap by dropping the first column.
##OneHotEncoding is not required for gender column because it hase only 2 genders.the diff in the formula due to expression becoming 0 will be adjusted in b0
# multiple regression formula : b0 + b1*x1 + b2*x2 + ... + bn*xn
##where b0= bias coeffecient, which will adjust if any expression in the formula becomes 0. it will adjust the value of the expression becoming 0 in b0.


# Concatenate the original X with the one-hot encoded features
X = np.concatenate((X[:, [0]], X_encoded, X[:, 2:]), axis=1)
#(X[:, [0] selects the first column of the original X. (X_encoded) contains the onehot encoded values.X[:, 2:] selects all rows and columns from 2nd column onwards.
##We are concatenating the first row from X, the onehot encoded values and the columns of X from 2nd column.  
###now X contains the one hot encoded values and has also dealt with the dummy variable trap by dropping the first column of X_encoded.

#Feature Scaling- Here we are scaling all the data into the same scale.
#can be done before train test split, then it would be easy.
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

#Using train-test split to break the data into training and testing data. test_size= 20%data is reserved for testing  
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Construction of our ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initializing our ANN
classifier = Sequential()

#Input layer and 1st hidden layer
classifier.add(Dense(input_dim = 11, units = 6, activation = "relu", kernel_initializer="uniform"))

#Adding 2nd hidden layer
classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform"))

#Adding 3rd hidden layer
classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform"))


#Adding output layer
classifier.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))
#classifier.add(Dense(units=4, kernel_initializer="uniform", activation="softmax"))
##4 values in the output layer, softmax activation function gives the probability of the 4 output values.

#Compiling the ANN.
classifier.compile(optimizer= keras.optimizers.Adam(learning_rate=0.10) , loss="binary_crossentropy", metrics=["accuracy"] )#loss="catrgorical_crossentropy" for multiple output values

#Training the ANN
classifier.fit(X_train, y_train, epochs=50, batch_size=40, validation_data=(X_test, y_test))


#predicting

y_pred= classifier.predict(X_test)
y_pred=(y_pred >= 0.5)

#Confusion matrix for checking the accuracy.
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# multiple regression formula : b0 + b1*x1 + b2*x2 + ... + bn*xn
##where b0= bias coeffecient, which will adjust if a term in the formula becomes 0. it will adjust the value of the trem becoming 0 in b0.




























