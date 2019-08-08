#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 21:20:36 2019

@author: ankit
"""
## Building CNN to classify cats and dogs

from keras.models import Sequential # To initialize the neural network
from keras.layers import Convolution2D # To Convolute the input image
from keras.layers import MaxPooling2D # Max pooling to for dimension flexibility
from keras.layers import Flatten # To give it to the ANN
from keras.layers import Dense # To add layers to the network

#Initializing the CNN
classifier = Sequential()

# Convoluting
classifier.add(Convolution2D(32,3,3,activation='relu',input_shape = (64,64,3)))

# Max pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Flattening
classifier.add(Flatten())

# Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid')) # softmax activation if more than 2 outputs

# compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # Stochastic gradient descent, loss function and metric of result

# fitting the CNN to the images
## Data augmentation

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         epochs=25,
                         validation_data= test_set,
                         validation_steps= 2000)
