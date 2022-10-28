import numpy as np
import cv2
import os

dir_path = os.getcwd()

for filename in os.listdir(dir_path):
    # If the images are not .JPG images, change the line below to match the image type.
    if filename.endswith(".JPG"):
        image = cv2.imread(filename)
        resized = cv2.resize(image,None,fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
        cv2.imwrite(filename,resized)
