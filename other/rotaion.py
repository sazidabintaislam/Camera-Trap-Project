# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 18:56:19 2020

@author: sazida

Objective: Rotating a folder of images at once
"""
#this code is collected from this link: https://www.geeksforgeeks.org/apply-changes-to-all-the-images-in-given-folder-using-python-pil/
# Code to apply operations on all the images 
# present in a folder one by one 
# operations such as rotating, cropping,  

from PIL import Image 
from PIL import ImageFilter 
import os 
  
def main(): 
    # path of the folder containing the raw images 
    inPath ="C:\\Users\\sazid\\Desktop\\rotate\\toad"
  
    # path of the folder that will contain the modified image 
    outPath ="C:\\Users\\sazid\\Desktop\\rotate\\toad90"
  
    for imagePath in os.listdir(inPath): 
        # imagePath contains name of the image  
        inputPath = os.path.join(inPath, imagePath) 
  
        # inputPath contains the full directory name 
        img = Image.open(inputPath) 
  
        fullOutPath = os.path.join(outPath, 'invert_'+imagePath) 
        # fullOutPath contains the path of the output 
        # image that needs to be generated 
        img.rotate(90).save(fullOutPath) 
        print(fullOutPath) 
  
# Driver Function 
if __name__ == '__main__': 
    main() 