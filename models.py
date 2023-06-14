# University of Stavanger
# Authors: Bruna Atamanczuk and Kurt Arve Skipenes Karadas
# 
# Code for Evolving Deep Neural Networks for Continuous Learning
# Delivered as part of master thesis in Applied Data Science
# June 2023

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential 
from keras.layers import Dense, Flatten, Conv2D, Activation, MaxPooling2D, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint

def model_mnist_mlp():
    # Creating model
    model_mlp = Sequential()

    # first hidden layer
    model_mlp.add(Flatten(input_shape=(28,28)))
    model_mlp.add(Dense(512, input_shape=(784,)))
    model_mlp.add(Activation('relu'))
    model_mlp.add(Dense(10))
    model_mlp.add(Activation('softmax'))

    return model_mlp

def model_mnist_cnn():
    model_cnn = Sequential()

    # First layer, which has a 2D Convolutional layer with 
    # kernel size as 3x3 and Max pooling operation 
    model_cnn.add(Conv2D(32, (3,3), padding='same', input_shape=(28,28, 1)))
    model_cnn.add(MaxPooling2D(pool_size=(2, 2)))

    # Second layer, which has a 2D Convolutional layer with 
    # kernel size as 3x3 & ReLU activation and Max pooling operation 
    model_cnn.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model_cnn.add(MaxPooling2D(pool_size=(2, 2)))

    # Fully connected layer with ReLU activation function 
    model_cnn.add(Flatten())
    model_cnn.add(Dense(128, activation='relu'))

    # Output layer with softmax activation function
    model_cnn.add(Dense(10, activation='softmax'))
    return model_cnn

def model_cifar_cnn():
    model_cnn = Sequential()

    # First layer, which has a 2D Convolutional layer with 
    # kernel size as 3x3 and Max pooling operation 
    model_cnn.add(Conv2D(32, 3, padding='same', 
                        input_shape=(32, 32, 3), 
                        activation= "relu"))
    model_cnn.add(Conv2D(32, 3, activation='relu'))
    model_cnn.add(MaxPooling2D())
    model_cnn.add(Dropout(0.25))

    # Second layer, which has a 2D Convolutional layer with 
    # kernel size as 3x3 & ReLU activation and Max pooling operation 
    model_cnn.add(Conv2D(64, 3, padding='same', activation='relu'))
    model_cnn.add(Conv2D(64, 3, activation='relu'))
    model_cnn.add(MaxPooling2D())
    model_cnn.add(Dropout(0.25))

    # Fully connected layer with ReLU activation function 
    model_cnn.add(Flatten())
    model_cnn.add(Dense(512, activation='relu'))
    model_cnn.add(Dropout(0.5))
    # Output layer with softmax activation function
    model_cnn.add(Dense(10, activation='softmax'))

    return model_cnn
