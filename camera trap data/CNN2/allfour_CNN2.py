# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:38:25 2020

@author: sazid
"""

'''In our setup, we:
- created a data/ folder
- created train/ validation/ and test subfolders inside data/
- created target species/ and background/ subfolders and inside train/ validation/ and test/ folder and put images inside conrresponing folder 
In summary, this is our directory structure:
```
data/
    train/
        lizard/
           ...
           ...
        snake/
           ...
           ...
        toad/
           ...
           ...
        background/
           ...
           ...
    validation/
        lizard/
           ...
           ...
        snake/
           ...
           ...
        toad/
           ...
           ...
        background/
           ...
           ...
    test/
        lizard/
           ...
           ...
        snake/
           ...
           ...
        toad/
           ...
           ...
        background/
           ...
           ...
'''
#######################################################
#Import Libraries

import numpy as np
from keras import backend as K
from keras.models import Sequential
#from keras.layers.core import Dense, Dropout, Activation, Flatten
#from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from datetime import datetime

######################################################
# Set the paths for training, testing and validation 

train_data_path = "C:\\Users\\camera trap\\CNN2\\3Species_background_CNN2\\train"
valid_data_path = "C:\\Users\\camera trap\\CNN2\\3Species_background_CNN2\\valid"
test_data_path = "C:\\Users\\camera trap\\CNN2\\3Species_background_CNN2\\test"

######################################################
# Set image size, batch size, sample number and epochs

img_rows = 150
img_cols = 150
epochs = 100
batch_size = 32
num_of_train_samples = 5742 
num_of_valid_samples = 1400


############################################
# Set Data Generator for training, testing and validataion.
# Note for testing, set shuffle = false (For proper Confusion matrix)

 # This will do preprocessing and realtime data augmentation:
 # this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40, # randomly rotate images in the range 
                                                      # (degrees, 0 to 40)
                                   width_shift_range=0.2, # randomly shift images horizontally 
                                                          # (fraction of total width)
                                   height_shift_range=0.2,# randomly shift images vertically 
                                                          # (fraction of total height)
                                   shear_range=30, #shear_range is for randomly applying shearing transformations
                                   zoom_range=0.3, #randomly zooming inside pictures
                                   horizontal_flip=True, # randomly flip images
                                   fill_mode='nearest')
###***Note 1: We can modify the augmenation specification and range. We can also ignore the whole augmenation process. In that case we just need to rescale the data as like the validation and test dataset shown below. More information is available in https://keras.io/api/preprocessing/image/ *****

###***Note 2: The Keras ImageDataGenerator class is not an “additive” operation. The ImageDataGenerator accepts the original data, randomly transforms it, and returns only the new, transformed data.In this process, model will see a slightly modified set of trainng samples (e.g., zoomed, shifted, rotated) in every iteration. more info: https://www.pyimagesearch.com/2019/07/08/keras-imagedatagenerator-and-data-augmentation/

#**************
# this is the augmentation configuration we will use for validation and testing:
# for validation and testing we do only rescaling
#The rescale parameter rescales the images pixel values between zero and one.
test_datagen = ImageDataGenerator(rescale=1. / 255)
valid_datagen = ImageDataGenerator(rescale=1. / 255)

# this is a generator that will read pictures found in subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(train_data_path, # this is the target directory
                                                    target_size=(img_rows, img_cols),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')
                                                    
# this is a similar generator, for validation data
validation_generator = valid_datagen.flow_from_directory(valid_data_path,
                                                        target_size=(img_rows, img_cols),
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        class_mode='categorical')
                                                        
 # this is a similar generator, for test data                                                       
test_generator = test_datagen.flow_from_directory(test_data_path,
                                                        target_size=(img_rows, img_cols),
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        class_mode='categorical')
                               

############################################
#Model Creation / Sequential

model = Sequential()

model.add(Conv2D((32), (3, 3), input_shape=(img_rows, img_cols, 3), kernel_initializer="glorot_uniform"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D((32), (3, 3),kernel_initializer="glorot_uniform"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D((64), (3, 3),kernel_initializer="glorot_uniform"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dropout(0.5))

model.add(Dense(128))

model.add(Dense(4))
model.add(Activation('softmax'))

#Get summary of the model
model.summary()

#Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

#Train the model

history=model.fit_generator(train_generator,
                    steps_per_epoch=num_of_train_samples // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=num_of_valid_samples // batch_size)
                    
#You can use model.save(filepath) to save a Keras model into a single HDF5 file which will contain:

#the weights of the model.
#the training configuration (loss, optimizer)
#the state of the optimizer, allowing to resume training exactly where you left off.
                    
model.save("all_four_CNN2.h5")

############################################
#Plot the Graph

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

# Accuracy Curves
plt.figure(1)
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy for all four class')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.savefig('acc7.png')
plt.show()

# loss Curves
plt.figure(2)
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss all four class')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.savefig('loss7.png')
plt.show()

##################################################################################################
#Print the Target names
#The class_indices attribute from the training set will help us in getting the class labels.

target_names = []
for key in train_generator.class_indices:
    target_names.append(key)
    
print(target_names)

##################################################################################################
#Confution Matrix 
#Get the accuracy score
Y_pred = model.predict_generator(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
#print('Confusion Matrix')

#Plot the confusion matrix. Set Normalize = True/False
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix of all four class', cmap=plt.cm.Blues):
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
   

#thresh = cm.max() / 2.
#Matplotlib’s matshow
cm = confusion_matrix(test_generator.classes, y_pred)

plt.figure(3)
plot_confusion_matrix(cm, target_names, title='Confusion Matrix')
#print(cm)
plt.savefig('allfour_Conf_CNN2.png')
plt.show()

#######################################
#Print Classification Report
print('Classification Report')
print(classification_report(test_generator.classes, y_pred, target_names=target_names))

#######################################
#calculating computation time

end = datetime.now()
time_taken = end - start
print('Time: ',time_taken)