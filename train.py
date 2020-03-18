# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 09:07:54 2019

@author: MIT-DGMIF
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image, ImageDraw
import cv2

from keras.layers import Input, Activation, Conv2D, Flatten, Dense, MaxPooling2D, Dropout, Add, LeakyReLU, UpSampling2D, BatchNormalization
from keras.models import Model, load_model, Input
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam

import scipy.misc
from skimage.io import imread
from glob import glob

import keras.backend as K
import os

#load dataset
base_path = 'D:\\AI code\\deeplearning\\Final_CNN_Code\\motion deblurring\\'
x_train = np.load(base_path + 'dataset\\x_train.npy')
y_train = np.load(base_path + 'dataset\\y_train.npy')
x_val = np.load(base_path + 'dataset\\x_val.npy')
y_val = np.load(base_path + 'dataset\\y_val.npy')

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)

# 학습 모델을 위한 CNN Model 정의 ##
deblur_CNN_input = Input(shape=(64, 64, 3))

#HIDDEN LAYERS
deblur_CNN_layer1 = Conv2D(filters=128, kernel_size=7, strides = 1, padding='same')(deblur_CNN_input)
deblur_CNN_layer1 = BatchNormalization()(deblur_CNN_layer1)
deblur_CNN_layer1 = Activation('relu')(deblur_CNN_layer1)

deblur_CNN_layer2 = Conv2D(filters=320, kernel_size=1, strides = 1, padding='same')(deblur_CNN_layer1)
deblur_CNN_layer2 = BatchNormalization()(deblur_CNN_layer2)
deblur_CNN_layer2 = Activation('relu')(deblur_CNN_layer2)

deblur_CNN_layer3 = Conv2D(filters=320, kernel_size=1, strides = 1, padding='same')(deblur_CNN_layer2)
deblur_CNN_layer1 = BatchNormalization()(deblur_CNN_layer3)
deblur_CNN_layer1 = Activation('relu')(deblur_CNN_layer3)

deblur_CNN_layer4 = Conv2D(filters=320, kernel_size=1, strides = 1, padding='same')(deblur_CNN_layer3)
deblur_CNN_layer4 = BatchNormalization()(deblur_CNN_layer4)
deblur_CNN_layer4 = Activation('relu')(deblur_CNN_layer4)

deblur_CNN_layer5 = Conv2D(filters=128, kernel_size=1, strides = 1, padding='same')(deblur_CNN_layer4)
deblur_CNN_layer5 = BatchNormalization()(deblur_CNN_layer5)
deblur_CNN_layer5 = Activation('relu')(deblur_CNN_layer5)

deblur_CNN_layer6 = Conv2D(filters=128, kernel_size=3, strides = 1, padding='same')(deblur_CNN_layer5)
deblur_CNN_layer6 = BatchNormalization()(deblur_CNN_layer6)
deblur_CNN_layer6 = Activation('relu')(deblur_CNN_layer6)

deblur_CNN_layer7 = Conv2D(filters=512, kernel_size=1, strides = 1, padding='same')(deblur_CNN_layer6)
deblur_CNN_layer7 = BatchNormalization()(deblur_CNN_layer7)
deblur_CNN_layer7 = Activation('relu')(deblur_CNN_layer7)

deblur_CNN_layer8 = Conv2D(filters=128, kernel_size=5, strides = 1, padding='same')(deblur_CNN_layer7)
deblur_CNN_layer8 = BatchNormalization()(deblur_CNN_layer8)
deblur_CNN_layer8 = Activation('relu')(deblur_CNN_layer8)

deblur_CNN_layer9 = Conv2D(filters=128, kernel_size=5, strides = 1, padding='same')(deblur_CNN_layer8)
deblur_CNN_layer9 = BatchNormalization()(deblur_CNN_layer9)
deblur_CNN_layer9 = Activation('relu')(deblur_CNN_layer9)

deblur_CNN_layer10 = Conv2D(filters=128, kernel_size=3, strides = 1, padding='same')(deblur_CNN_layer9)
deblur_CNN_layer10 = BatchNormalization()(deblur_CNN_layer10)
deblur_CNN_layer10 = Activation('relu')(deblur_CNN_layer10)

deblur_CNN_layer11 = Conv2D(filters=128, kernel_size=5, strides = 1, padding='same')(deblur_CNN_layer10)
deblur_CNN_layer11 = BatchNormalization()(deblur_CNN_layer11)
deblur_CNN_layer11 = Activation('relu')(deblur_CNN_layer11)

deblur_CNN_layer12 = Conv2D(filters=128, kernel_size=5, strides = 1, padding='same')(deblur_CNN_layer11)
deblur_CNN_layer12 = BatchNormalization()(deblur_CNN_layer12)
deblur_CNN_layer12 = Activation('relu')(deblur_CNN_layer12)

deblur_CNN_layer13 = Conv2D(filters=256, kernel_size=1, strides = 1, padding='same')(deblur_CNN_layer12)
deblur_CNN_layer13 = BatchNormalization()(deblur_CNN_layer13)
deblur_CNN_layer13 = Activation('relu')(deblur_CNN_layer13)

deblur_CNN_layer14 = Conv2D(filters=64, kernel_size=7, strides = 1, padding='same')(deblur_CNN_layer13)
deblur_CNN_layer14 = BatchNormalization()(deblur_CNN_layer14)
deblur_CNN_layer14 = Activation('relu')(deblur_CNN_layer14)

deblur_CNN_output = Conv2D(filters=3, kernel_size=7, strides = 1, padding='same', activation='relu')(deblur_CNN_layer14)

deblur_CNN = Model(inputs= deblur_CNN_input, outputs=deblur_CNN_output )

deblur_CNN.summary()

deblur_CNN.compile(optimizer= 'adam', loss= 'mean_squared_error', metrics=['acc', 'mse'])
#Train



hdf5_file = "D:\\AI code\\deeplearning\\Final_CNN_Code\\motion deblurring\\weight\\deblur_cnn_weights.h5"

if os.path.exists(hdf5_file):
    # 기존에 학습된 모델 불러들이기
    deblur_CNN.load_weights(hdf5_file)
    
    Deblurred = deblur_CNN.predict(x_val)
    Deblurred = np.clip(Deblurred, 0, 255)
    f, ax = plt.subplots(6,10, figsize=(20,15))
    gray_ori = []
    gray_motion = []
    gray_deblurr = []
    
    for i in range(10):
        ax[0,i].imshow(y_val[i]);  ax[0,i].axis('Off'); ax[0,i].set_title('Original', size=15)
        ax[1,i].imshow(x_val[i]);  ax[1,i].axis('Off'); ax[1,i].set_title('Motion blurred', size=15)
        ax[2,i].imshow(Deblurred[i]);  ax[2,i].axis('Off'); ax[2,i].set_title('Deblurred', size=15)
        
        gray_ori.append(cv2.cvtColor(y_val[i] * 255, cv2.COLOR_BGR2GRAY))
        
        ax[3,i].imshow(gray_ori[i], cmap = cm.gray, vmin = 0, vmax = 255);  ax[3,i].axis('Off'); ax[3,i].set_title('Original', size=15)
        
        gray_motion.append(cv2.cvtColor(x_val[i] * 255, cv2.COLOR_BGR2GRAY))
        
        ax[4,i].imshow(gray_motion[i], cmap = cm.gray, vmin = 0, vmax = 255);  ax[4,i].axis('Off'); ax[4,i].set_title('Motion blurred', size=15)
        
        gray_deblurr.append(cv2.cvtColor(Deblurred[i] * 255, cv2.COLOR_BGR2GRAY))
        
        ax[5,i].imshow(gray_deblurr[i], cmap = cm.gray, vmin = 0, vmax = 255);  ax[5,i].axis('Off'); ax[5,i].set_title('Deblurred', size=15)
    plt.show()
    


else:
    # 학습한 모델이 없으면 파일로 저장
    history = deblur_CNN.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50, batch_size=30, callbacks=[
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=1, mode='auto', min_lr=1e-05)])

    deblur_CNN.save_weights(hdf5_file)

#Evaluation

    fig, ax = plt.subplots(2, 2, figsize=(10, 7))

    ax[0, 0].set_title('loss')
    ax[0, 0].plot(history.history['loss'], 'r')
    ax[0, 1].set_title('acc')
    ax[0, 1].plot(history.history['acc'], 'b')

    ax[1, 0].set_title('val_loss')
    ax[1, 0].plot(history.history['val_loss'], 'r--')
    ax[1, 1].set_title('val_acc')
    ax[1, 1].plot(history.history['val_acc'], 'b--')

"""
    preds = deblur_CNN.predict(x_val)

    fig, ax = plt.subplots(len(x_val), 3, figsize=(10, 100))

    #for i, pred in enumerate(preds):
     #   ax[i, 0].imshow(x_val[i].squeeze(), cmap='gray')
      #  ax[i, 1].imshow(y_val[i].squeeze(), cmap='gray')
       # ax[i, 2].imshow(pred.squeeze(), cmap='gray')
    
    for i, pred in enumerate(preds):
        ax[i, 0].imshow(x_val[i].squeeze(), cmap='gray')
        ax[i, 1].imshow(y_val[i].squeeze(), cmap='gray')
        ax[i, 2].imshow(preds[i].squeeze(), cmap='gray')
"""