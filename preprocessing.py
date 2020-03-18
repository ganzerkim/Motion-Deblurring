# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:05:17 2019

@author: ganze
"""

#motion_blurring_preprocess

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import pyramid_reduce, resize

import os, glob

motion_list = sorted(glob.glob('C:\\Users\\MDDC\\Desktop\\data\\blurred_rgb\\*.png'))
origin_list = sorted(glob.glob('C:\\Users\\MDDC\\Desktop\\data\\origin_rgb\\*.png'))

print(len(motion_list), len(origin_list))

IMG_SIZE = 64

x_data, y_data = np.empty((2, len(motion_list), IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

for i, img_path in enumerate(motion_list):
    img = imread(img_path)
    img = resize(img, output_shape=(IMG_SIZE, IMG_SIZE, 3), preserve_range=True)
    x_data[i] = img
    
for i, img_path in enumerate(origin_list):
    img = imread(img_path)
    img = resize(img, output_shape=(IMG_SIZE, IMG_SIZE, 3), preserve_range=True)
    y_data[i] = img

x_data = np.array(x_data).astype('float32')
y_data = np.array(y_data).astype('float32')    

x_data /= 255.
y_data /= 255.

fig, ax = plt.subplots(1, 2)
ax[0].imshow(x_data[12].squeeze(), cmap='gray')
ax[1].imshow(y_data[12].squeeze(), cmap='gray')

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size = 0.1)

base_path = 'C:\\Users\\MDDC\\Desktop\\data\\'

if not(os.path.exists(base_path + 'dataset')):
    os.mkdir(base_path + 'dataset')
    
np.save(base_path + 'dataset\\x_train.npy', x_train)
np.save(base_path +'dataset\\y_train.npy', y_train)
np.save(base_path +'dataset\\x_val.npy', x_val)
np.save(base_path +'dataset\\y_val.npy', y_val)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)