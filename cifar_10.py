import itertools
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import Sequential 
from keras.layers import Dense, Flatten, Conv2D, Activation, MaxPooling2D, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
import os
from keras.callbacks import ModelCheckpoint
from helper_functions import remove_class, clone_a_model, create_row
from helper_functions import get_confusion_matrix
from helper_functions import evaluate_evolutionary_strategy, naming_figures

# set seed
np.random.seed(3446)
tf.random.set_seed(345)

# -------------------------------------------
# Define global variables
# -------------------------------------------
DATASET = 'CIFAR-10'
MODEL = 'CNN'
STRATIFY = True
HIDE = False
MUTATIONS = [3, 5, 10, 15, 50]


# -------------------------------------------
# Params to save figures
# -------------------------------------------
mod_title = naming_figures(DATASET, MODEL, STRATIFY, HIDE)
filepath = f'confusion_matrix/{DATASET}'


# -------------------------------------------
# Load data
# -------------------------------------------
cifar = tf.keras.datasets.cifar10
(X_train, y_train), (X_test, y_test) = cifar.load_data()
classes = np.unique(y_train)


# -------------------------------------------
# Normalizing data 
# -------------------------------------------
X_train = X_train / 255
X_test = X_test / 255


# -------------------------------------------
# Split data
# -------------------------------------------
(X_largeT, y_largeT), (X_smallT, y_smallT), class_no = remove_class(
    X_train, 
    y_train, 
    classes,
    hide = HIDE, 
    stratitify=STRATIFY)


# -------------------------------------------
# One-hot encoding the labels
# -------------------------------------------
num_classes = len(np.unique(y_train))

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
y_largeT = tf.keras.utils.to_categorical(y_largeT, num_classes)
y_smallT = tf.keras.utils.to_categorical(y_smallT, num_classes)

print(f"X_train shape: {X_train.shape} \ny_train shape: {y_train.shape}\n")
print(f"X_test shape: {X_test.shape} \ny_test shape: {y_test.shape}\n")
print('-'*50)
print(f"X_largeT shape: {X_largeT.shape} \ny_largeT shape: {y_largeT.shape}\n")
print(f"X_smallT shape: {X_smallT.shape} \ny_smallT shape: {y_smallT.shape}\n")


# -------------------------------------------
# Creating model
# -------------------------------------------
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

model_cnn.summary()

# -------------------------------------------
# Compiling the model 
# -------------------------------------------
model_cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)


# -------------------------------------------
# Baseline model
# -------------------------------------------
m_base = clone_a_model(model_cnn)
m_base.compile(loss='categorical_crossentropy', 
               optimizer='adam', 
               metrics=['accuracy'])
m_base.fit(X_train, y_train,
                  batch_size=128,
                  epochs=100,
                  verbose=0,
                  validation_split= 0.1,
                  callbacks=[es_callback],
                  )


# -------------------------------------------
# Evaluate baseline model
# -------------------------------------------
loss, accuracy = m_base.evaluate(X_test, y_test)
print('Test loss baseline:', loss)
print('Test accuracy baseline:', accuracy)


# -------------------------------------------
# Write results to file
# -------------------------------------------
row_txt1 = create_row(DATASET, MODEL, STRATIFY, HIDE,
                      'm_baseline', 'accuracy', round(accuracy, 4))
row_txt2 = create_row(DATASET, MODEL, STRATIFY, HIDE,
                      'm_baseline', 'loss', round(loss, 4))
with open('results/results.csv', 'a') as f:
    for txt in (row_txt1,row_txt2):
        f.write("%s\n" %str(txt))


# -------------------------------------------
# Get confusion matrix for baseline
# -------------------------------------------
report_m = get_confusion_matrix(m_base,
                              X_test, 
                              y_test, 
                              classes,
                              title = "Confusion Matrix - baseline model",
                              prefix= f"{DATASET}-{MODEL}",
                              filepath=filepath,
                        )


# -------------------------------------------
# Creating and training model m0 (the first stream of data)
# -------------------------------------------
m0 = clone_a_model(model_cnn)
m0.compile(loss='categorical_crossentropy', 
           optimizer='adam', 
           metrics=['accuracy'])
history = m0.fit(X_largeT, y_largeT,
        batch_size=128,
        epochs=100,
        verbose=0,
        validation_split= 0.1,
        callbacks=[es_callback],
        )


# -------------------------------------------
# Evaluate m0
# -------------------------------------------
loss, accuracy = m0.evaluate(X_test, y_test)
print('Test loss m0:', loss)
print('Test accuracy m0:', accuracy)


# -------------------------------------------
# Write results to file
# -------------------------------------------
row_txt1 = create_row(DATASET, MODEL, STRATIFY, HIDE,
                      'm0', 'accuracy', round(accuracy, 4))
row_txt2 = create_row(DATASET, MODEL, STRATIFY, HIDE,
                      'm0', 'loss', round(loss, 4))
with open('results/results.csv', 'a') as f:
    for txt in (row_txt1,row_txt2):
        f.write("%s\n" %str(txt))


# -------------------------------------------
# Get confusion matrix
# -------------------------------------------
if class_no is not None: 
    title  = f"Confusion Matrix - missing class no. {class_no}"
else: 
    title = f"Confusion Matrix - all classes"

report_m0 = get_confusion_matrix(m0,
                              X_test, 
                              y_test, 
                              classes,
                              title = title,
                              prefix= mod_title,
                              filepath=filepath,
                              )


# -------------------------------------------
# Evolutionary strategy
# -------------------------------------------
results = evaluate_evolutionary_strategy(m0, MUTATIONS, 
                                         X_smallT, y_smallT, X_test, y_test)


# -------------------------------------------
# Write results
# -------------------------------------------
for key, val in results.items():
    for k, v in val.items():
        if k != 'model':
            row_txt = create_row(DATASET, MODEL, STRATIFY, HIDE,
                                 f'mes{key}', k, round(v, 4))
            with open('results/results.csv', 'a') as f:
                f.write("%s\n" %str(row_txt))


# -------------------------------------------
# Plot confusion matrix for ES
# -------------------------------------------
x = [get_confusion_matrix(results[i]["model"],
                              X_test, 
                              y_test, 
                              classes,
                              title = f"Confusion Matrix, ES - {i} mutations",
                              prefix = mod_title, 
                              filepath=filepath,
                        ) for i in results.keys()]