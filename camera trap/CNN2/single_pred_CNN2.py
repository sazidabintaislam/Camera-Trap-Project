#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.models import load_model
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from PIL import Image
from skimage import transform
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt

#####################################################
# Set the paths for training

train_data_path ="C:\\Users\\sazid\\Desktop\\camera_trap\\snake_background_CNN2\\train"

#####################################################
# Set image size, batch size

img_rows = 150
img_cols = 150
batch_size = 32

#######################################################
#this partis using just to check class lebel
#no training is going on here. Just checking the class label of traning dataset
#Image Generator
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(train_data_path,
                                                    target_size=(img_rows, img_cols),
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    class_mode='categorical')

# Function for plots images with labels within jupyter notebook

def plots(ims, figsize=(12,12), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1

    for i in range(len(ims)):
        sp = f.add_subplot(cols, rows, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=12)
        #plt.imshow(ims[i], interpolation=None if interp else 'none')


#Check the training set (with batch of 10 as defined above
imgs, labels = next(train_generator)

#Images are shown in the output
#plots(imgs, titles=labels)

#Images Classes with index
print(train_generator.class_indices)


##################################################################################################
#Print the Target names
#The class_indices attribute from the training set will help us in getting the class labels.

target_names = []
for key in train_generator.class_indices:
       target_names.append(key)
    
#print(target_names)

#image size of Test data
img_rows = 150
img_cols = 150


#load the saved model
model = load_model("C:\\Users\\sazid\\Desktop\\spyder\\aug8.h5")

#image file path of single image
file = "C:\\Users\\sazid\\Desktop\\spyder\\sb1.JPG"

img = cv2.cvtColor(cv2.imread(file),cv2.COLOR_BGR2RGB)# read in the image, convert to grayscale
img = cv2.resize(img, (img_rows,img_cols)) # resize image to match model's expected sizing

test_image = image.img_to_array(img)
test_image = np.expand_dims(test_image, axis=0)
pred = model.predict_classes(test_image)
print(pred)
#print(labels)
#print(pred, labels[np.argmax(pred)])





