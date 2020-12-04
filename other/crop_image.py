# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 00:19:22 2020

@author: Sazida Binta Islam

Objective: cropping a folder of images at once

For more understanding :https://www.tutorialspoint.com/python_pillow/python_pillow_cropping_an_image.htm
"""

#Import required Image library
from PIL import Image
import os

#Create an Image Object from an Image
#source directory
dir="C:\\Users\\sazid\\Desktop\\crop"
#directory for cropped image
output_dir = "C:\\Users\\sazid\\Desktop\\crop1"
file_names = os.listdir(dir)

for file_name in file_names:
    file_path = dir +"\{}".format(file_name)
    im = Image.open(r"{}".format(file_path))
    output_file= output_dir+"\{}".format(file_name)
    #**********************************************************
    # Setting the points for cropped image 
    #4-tuple defining the left, upper, right, and lower pixel coordinate
    #Set the cropping area with box=(left, upper, right, lower).
    #The top left coordinates correspond to (x, y) = (left, upper), and the bottom right coordinates correspond to (x, y) = (right, lower)
    cropped = im.crop((0,70,2045,1450)) 
    
    #Save the cropped image
    cropped.save(r"{}".format(output_file))
    