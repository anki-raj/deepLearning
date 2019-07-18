# Pre processing of data

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset

data = pd.read_csv('Data.csv')

#seperate the dependent and independent variables

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

#Replace the missing data with the mean of the other values

from sklearn.preprocessing import Imputer #imputation transformer for completing missing value
imputer = Imputer(missing_values = 'NaN',strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,1:]) #to fit it in our matrix X
X[:,1:] = imputer.transform(X[:,1:])

# encoding categorical variables

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])

# use onehotencoder so that the variables do not have a relative value

onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Splitting the dataset into train and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=0)

#feature scaling (Since ML models uses Euclidian distance between the features, the scale may become large)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


