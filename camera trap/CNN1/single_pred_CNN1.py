# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 00:45:25 2020

@author: sazid
"""


import cv2
import tensorflow as tf
import numpy as np
import random

#CATEGORIES = ["background", "snake", "toad", "lizard"]  # will use this to convert prediction num to string value
CATEGORIES = ["background",  "snake"] 


def prepare(filepath):
    IMG_SIZE = 150 

    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) # read in the image, convert to grayscale
    img_array = img_array / 255.0 # resize image to match model's expected sizing
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1) # return the image with shaping that TF wants.

#load the saved model
model = tf.keras.models.load_model("test8.h5", compile=False)

#image file path of single image
prediction = model.predict_classes(prepare('sb11.JPG'))
print(prediction)
