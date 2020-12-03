# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 18:00:39 2019

@author: sazid
"""

'''In my setup, I:
- created a data/ folder
- created DATADIR/ and test/ subfolders inside data/
- created target species/ and background/ subfolders inside DATADIR/ and test/ folder and put images inside conrresponing folder 
-the images under DATADIR/ will be used to train and validate the network, and the data splitting will be done randomly in the code 
In summary, this is my directory structure:

data/
    DATADIR/    
        snake/
           ...
           ...
        toad/
           ...
           ...
        lizard/
           ...
           ...
    test/
        snake/
           ...
           ...
        toad/
           ...
           ...
        lizard/
           ...
           ...
'''
#######################################################
#Import Libraries

import cv2
import numpy as np
import os
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
from tensorflow.keras import initializers
import itertools 

start = datetime.now()

#####################################################
# Set the paths for training,validation and testing 

DATADIR= "C:\\Users\\data\\online data\\CNN1\\all3species_CNN1\\DATADIR" 
test="C:\\Users\\data_proff\\data\\online data\\CNN1\\all3species_CNN1\\test" 

####################################################
# Set image category

CATEGORIES = [ "snake", "toad", "lizard"]
test_categories=["snake", "toad", "lizard"]

####################################################
# Set image size
IMG_SIZE = 100

####################################################
#building our training data
training_data = []

def create_training_data():
    for category in CATEGORIES:  # do "snake", "lizard", "toad"

        path = os.path.join(DATADIR,category)  # create path to "snake", "lizard", "toad"
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=snake,1=lizard,2=toad
        print (class_num)
        for img in tqdm(os.listdir(path)):  # iterate over each image per "snake", "lizard", "toad"
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            
create_training_data()

#print(len(training_data))
#print(len(test))

##########################################################################
#building our testing data
test_data = []

def create_test_data():
    for category in test_categories:   # do "snake", "lizard", "toad"

        path = os.path.join(test,category)  # create path to "snake", "lizard", "toad"
        class_num = test_categories.index(category)  # get the classification  (0 or a 1). 0=snake,1=lizard,2=toad

        for img in tqdm(os.listdir(path)):  # iterate over each image per "snake", "lizard", "toad"
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                test_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                test_data.append([test_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_test_data()
#print(len(test_data))

#shuffle the data
import random

random.shuffle(training_data)
#for sample in training_data[:5]:
    #print(sample[1])

X_train = []
y_train = []

for features,label in training_data:
    X_train.append(features)
    y_train.append(label)

print(X_train[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = np.array(y_train)

#rescales the images pixel values.
X_train = X_train/255.0

X_test = []
y_test = []

for features,label in test_data:
    X_test.append(features)
    y_test.append(label)

print(X_test[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = np.array(y_test)

#rescales the images pixel values.
X_test = X_test/255.0

######################################################################
#Model Creation / Sequential

model = Sequential()

model.add(Conv2D((32), (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 1), kernel_initializer="glorot_uniform", bias_initializer="zeros"))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D((32), (3, 3),kernel_initializer="glorot_uniform", bias_initializer="zeros"))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D((64), (3, 3),kernel_initializer="glorot_uniform", bias_initializer="zeros"))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D((128), (3, 3),kernel_initializer="glorot_uniform", bias_initializer="zeros"))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(128))

model.add(Dense(3))
model.add(Activation('softmax'))

#Get summary of the model
model.summary()

#Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

##################################################################################
#Traning data needs to split into training and validation data here by providing instrauction. 
#batch size and epoch size is defined

# callback can be used that will stop the training when there is no improvement in
#monitor the validation loss for twenty consecutive epochs and stops the computaion if the model stop improving
#number of patience can be changed to any number.
#CB = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True )
history=model.fit(X_train, y_train, batch_size=32, epochs=250, validation_split=0.25) #,callbacks=[CB]))

##########################################
#You can use model.save(filepath) to save a Keras model into a single HDF5 file which will contain:
#the weights of the model.
#the training configuration (loss, optimizer)
#the state of the optimizer, allowing to resume training exactly where you left off.

model.save("all3_CNN1_online.h5")

#############################################################################
#Plot the acc and loss Graph

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.figure(1)
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy of lizard, snake and toad')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.savefig('all3_acc.png')
plt.show()

plt.figure(2)
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss of lizard, snake and toad')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.savefig('all3_loss.png')
plt.show()

##################################################################################################
#Get the accuracy score
result=model.evaluate(X_test, y_test, verbose=1)
print(result)

##################################################################################################
#Confution Matrix 
#Plot the confusion matrix. Set Normalize = True/False

def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix of snake,lizard,toad, cmap=plt.cm.Blues', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    #plt.figure(figsize=(8,8))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
   
y_pred = model.predict_classes(X_test)
#print (y_pred)
cm = confusion_matrix(y_true=y_test,y_pred=y_pred)
print(cm)

plt.figure(3)
plot_confusion_matrix(cm, test_categories, title='Confusion Matrix')
plt.savefig('all3_conf_CNN1.png')
plt.show()

#######################################
#Print Classification Report
print('Classification Report')
print(classification_report(y_test, y_pred, target_names=test_categories))

#######################################
#calculating computation time
end = datetime.now()
time_taken = end - start
print('Time: ',time_taken)