# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 10:30:24 2019

@author: MIT-DGMIF
"""

import pydicom # for reading dicom files
import os # for doing directory operations 
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
import numpy as np

data_dir = 'D:\\김민건\\DSB3\\stage1\\stage1\\stage1\\'
patients = os.listdir(data_dir)

labels_df = pd.read_csv('D:\\김민건\\DSB3\\stage1_labels.csv', index_col=0, engine='python')

#labels_df.head()

######################
ii = 0
for patient in patients[:]:
    label = labels_df.get_value(patient, 'cancer')
    path = data_dir + patient
    
    # a couple great 1-liners from: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    
    print(len(slices))
    print(slices[0].pixel_array)
   
    plt.imshow(slices[0].pixel_array)
    plt.show()
    
    os.makedirs('C:\\Users\\MIT-DGMIF\\Desktop\\aaa\\' + str(ii))
    idx = 0

    for idx in range(len(slices)):
        img = slices[idx].pixel_array
        norm = np.zeros((512, 512))
        img_norm = cv2.normalize(img, norm, 0, 255, cv2.NORM_MINMAX)
        img_norm = img_norm.astype('uint8')
      
        Image.fromarray(img_norm[:,:]).save('C:\\Users\\MIT-DGMIF\\Desktop\\aaa\\' +str(ii) + '\\input' + str(idx) + '.png')
        #Image.fromarray(motion_images_aug[:, :, 0]).save('C:\\Users\\MIT-DGMIF\\Desktop\\data\\CTblurr\\result' + str(idx) + '.png')
        
        #plt.imshow(img_norm)
    ii += 1