# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 09:41:45 2019

@author: ganze
"""
import numpy as np
from PIL import Image, ImageDraw
import os
import cv2
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import random
import matplotlib.pyplot as plt

# Open the input image as numpy array, convert to RGB

IMAGE_PATH = 'C:\\Users\\MDDC\\Desktop\\data\\origin'

data = os.listdir(IMAGE_PATH)
data.sort()

img = [cv2.imread(IMAGE_PATH + '\\' + s, cv2.IMREAD_GRAYSCALE) for s in os.listdir(IMAGE_PATH)]
# plot image in color
"""
plt.imshow(img[0], cmap="jet_r")
#save image in color
plt.imsave("C:\\Users\\ganze\\Desktop\\motion blurring\\phz\\input\\color.png", img[0], cmap="jet_r")
plt.show()
"""

idx = 0

COLOR_PATH = 'C:\\Users\\MDDC\\Desktop\\data\\color'
for idx in range(len(img)):
    images = np.array(img[idx])
    plt.imsave("C:\\Users\\MDDC\\Desktop\\data\\color\\color" + str(idx) + ".png", img[idx], cmap="jet_r")
    
        
    rgb = (cv2.imread(COLOR_PATH + '\\color' + str(idx) + ".png", cv2.IMREAD_COLOR))
    #행, 열 불러오기
    #height, width = images.shape[:2]    
    img_rgb = np.array(rgb)
    #이미지 resize
    input_images = cv2.resize(img_rgb, (256, 256), interpolation = cv2.INTER_CUBIC)
    
    #zeros = np.zeros((images.shape[0], images.shape[1]), dtype="uint8")
    #input_images = cv2.merge([images_r, images_r, images_r])
  
    
    #MotionBlur(K, A, D, O)
# Blurs an image using a motion blur kernel with size K. 
# A is the angle of the blur in degrees to the y-axis (value range: 0 to 360, clockwise). 
# D is the blur direction (value range: -1.0 to 1.0, 1.0 is forward from the center). 
# O is the interpolation order (O=0 is fast, O=1 slightly slower but more accurate).
    
    #k = random.randrange(3, 16, 3)
    k = 45
    a = random.choice([0, 180])
    
    #d = random.random() * random.choice([-1, 1])
    d = -1
    #o = random.choice([0, 1])
    o = 0
    motion_blurer = iaa.MotionBlur(k, a, d, o)
    motion_images_aug = motion_blurer.augment_images(input_images)
    #plt.imshow(images)
    #plt.imshow(motion_images_aug)

    # resize
    #input_images.resize(128, 128, 3)
    #motion_images_aug.resize(128, 128, 3)  
    
    
    # Save with alpha
    #Image.fromarray(input_images[:, :, 0]).save('C:\\Users\\MIT-DGMIF\\Desktop\\data\\origin\\origin' + str(idx) + '.png')
    #Image.fromarray(motion_images_aug[:, :, 0]).save('C:\\Users\\MIT-DGMIF\\Desktop\\data\\blurred\\blurred' + str(idx) + '.png')
    Image.fromarray(input_images).save('C:\\Users\\MDDC\\Desktop\\data\\origin_rgb\\original' + str(idx) + '.png')
    Image.fromarray(motion_images_aug).save('C:\\Users\\MDDC\\Desktop\\data\\blurred_rgb\\blurred' + str(idx) + '.png')        
    
    