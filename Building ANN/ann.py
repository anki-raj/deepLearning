#ANN to solve a classification problem - check who is likely to leave the company

# Data Preprocessing of the dataset
import pandas as pd
import numpy as np

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
import tensorflow as tf
from keras import backend as K


from keras.models import Sequential #to initialize the ANN
from keras.layers import Dense #to create layers in ANN
from keras.layers import Dropout # To reduce overfitting


#initializing ANN - 2 ways : Defining sequence of layers / Defining Graph
#We'll Define sequence of layers.
classifier = Sequential()

#Adding input layer and hidden layer : adding no. of hidden layer - average of input and output layer, initializing weights, adding
#Rectifier activation function and the input dimentions.
classifier.add(Dense(units=6,activation='relu',kernel_initializer='uniform',input_dim=11))
classifier.add(Dropout(p=0.1))

#Adding hidden layer
classifier.add(Dense(units=6,activation='relu',kernel_initializer='uniform'))
classifier.add(Dropout(p=0.1))

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
predict_val = classifier.predict(sc_X.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
predict_val = (predict_val > 0.5)

#Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential #to initialize the ANN
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()    
    classifier.add(Dense(units=6,activation='relu',kernel_initializer='uniform',input_dim=11))
    classifier.add(Dense(units=6,activation='relu',kernel_initializer='uniform'))
    classifier.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))
    classifier.compile(optimizer='adam',loss= 'binary_crossentropy',metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10,nb_epoch=100)
accuricies = cross_val_score(estimator=classifier, X= X_train, y=y_train,cv=10,n_jobs=-1)
mean = accuricies.mean()
variance = accuricies.std()
#accuracy achieved = 83.6%

# Improving the ANN using Dropout regularization(To prevent overfitting) - It randomly disables neurons so that it's not too dependent of other neurons.
#It improves the configuration.

#Tuning the ANN - change the hyperparameter(epochs, batch_size, optimizer etc.) to improve the accuracy.
#SearchGrid helps to find the best match.

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential #to initialize the ANN
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()    
    classifier.add(Dense(units=6,activation='relu',kernel_initializer='uniform',input_dim=11))
    classifier.add(Dense(units=6,activation='relu',kernel_initializer='uniform'))
    classifier.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))
    classifier.compile(optimizer=optimizer,loss= 'binary_crossentropy',metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size' : [25,32],
              'epochs' : [100,500],
              'optimizer' : ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator=classifier, param_grid= parameters,scoring='accuracy',cv=10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
#Accuracy acheived - 85.1625%


