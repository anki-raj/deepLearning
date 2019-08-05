# Building Deep learning models

The repository is categorized into building various models using deep learning

## Artificial Neural Network Model

An ANN model is built to predict whether a customer is likely to leave the bank based on many independent attributes like salary, age, tenure etc.

### Libraries required
    Pandas
    Sklearn
    Keras - Using Tensorflow backend

### Steps to be followed
    Data preprocessing - Diving the dataset, categorical encoding, feature scaling
    Initialize the ANN - using sequencial() class.
    Add the input,hidden and output layer - using Dense() class.
    compile the ANN - Used stochastic gradient descent
    Train it on the train data set.
    test it on the test data set.
    Find the accuracy using confusion matrix
    
    accuracy achieved - 85.5%

### Evaluating the ANN
    K-fold cross validation is used to evaluate the ANN
    Use keras wrapper to wrap the cross validation of scikit-learn
    Use 10 folds to spilt the training set by 10 and do parallel training of these subsets
    Accuracy achieved - 83.6%

### Improving The ANN
    Dropout regularization is used to prevent overfitting
    It randomly disables the neurons at a specified state to improve the configuaration

### Tune the ANN
    Adjust the hyperparameters like epochs, number of batches, optimizer to improve the accuracy
    GridSearchCV is used to search for the best parameter and accuracy.
    

### Final Accuracy Achieved : 85.1625%




