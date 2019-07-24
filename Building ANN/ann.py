#ANN to solve a classification problem - check who is likely to leave the company

# Data Preprocessing of the dataset
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values

#Encoding the categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])
labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
#Splitting into test and train sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Making the ANN
import keras #wraps the above two library. Runs on Theano and tensorflow. Numerical computational library
from keras.models import Sequential #to initialize the ANN
from keras.layers import Dense #to create layers in ANN

#initializing ANN - 2 ways : Defining sequence of layers / Defining Graph
#We'll Define sequence of layers.
classifier = Sequential()

#Adding input layer and hidden layer : adding no. of hidden layer - average of input and output layer, initializing weights, adding
#Rectifier activation function and the input dimentions.
classifier.add(Dense(units=6,activation='relu',kernel_initializer='uniform',input_dim=11))

#Adding hidden layer
classifier.add(Dense(units=6,activation='relu',kernel_initializer='uniform'))

#Adding output layer
classifier.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))

#Compiling the ANN - Adding Schtocastic gradient descent.
classifier.compile(optimizer='adam',loss= 'binary_crossentropy',metrics = ['accuracy'])

#Fit the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, epochs=100)

# Predict the test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# making a confusion matrix to check the accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Final Accuracy - 86.5%


