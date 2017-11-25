# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 07:49:08 2017

@author: THANHHAI
"""
# Modified the original U-Net model with input (240, 240, 3)
from __future__ import print_function
# import os
# force tensorflow to use CPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
# from skimage import data, util
# from skimage.measure import label, regionprops
# from skimage import io
# from skimage.transform import resize
# import SimpleITK as sitk
from matplotlib import pyplot as plt
# import subprocess
# import random
# import progressbar
# from glob import glob
import gc


import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dropout, Activation, Reshape
from keras.layers import Input, Conv2DTranspose
from keras.layers import Conv2D, MaxPooling2D, concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import PReLU, LeakyReLU
# from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

from BRATs_data_unet_2 import load_train_data, load_val_data

# TF dimension ordering in this code
# K.set_image_data_format('channels_last')  

# an argument to your Session's config arguments, it might help a little, 
# though it won't release memory, it just allows growth at the cost of some memory efficiency
K.clear_session()
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.90
sess = tf.Session(config=config)
K.set_session(sess)
# with tf.Session(config = config) as s:

# ==========================================================================
smooth = 1.
nclasses = 5 # no of classes, if the output layer is softmax
# nclasses = 1 # if the output layer is sigmoid
img_rows = 240
img_cols = 240

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true.astype('float32'))
    y_pred_f = K.flatten(y_pred.astype('float32'))
    intersection = K.sum(y_true_f * y_pred_f)
    
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def cnnBRATsInit_unet():  
    # the output layer is softmax, loss should be categorical_crossentropy
    # Downsampling 2^4 = 16 ==> 240/16 = 15
    
    inputs = Input((img_rows, img_cols, 3))        
    # inputs = Input((img_rows, img_cols, 4))
    
    # Normalize all data before extracting features
    input_nor = BatchNormalization()(inputs)
    # Block 1
    conv1 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(input_nor)
    # conv1 = BatchNormalization()(conv1)
    # conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(conv1)
    conv1 = BatchNormalization()(conv1)
    # conv1 = Dropout(0.2)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)    
    
    # Block 2
    conv2 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(pool1) 
    # conv2 = BatchNormalization()(conv2)
    # conv2 = Dropout(0.5)(conv2)
    conv2 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(conv2)
    conv2 = BatchNormalization()(conv2)
    # conv2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # pool2 = Dropout(0.5)(pool2)
    
    # this MaxPooling2D has the same result with the above line.
    # pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv2)
    
    # W_new = (W - F + 2P)/S + 1
    # W_new = (120 - 3 + 2*1)/2 + 1 = 60.5 --> 60 (round down)
    # P = 1, because the padding parameter is 'same', the size of W has no change
    # before divide by Stride parameter
    
    
    # Block 3
    conv3 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', 
                   kernel_regularizer=l2(0.01))(pool2)  
    # conv3 = BatchNormalization()(conv3)
    # conv3 = Dropout(0.5)(conv3)
    conv3 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', 
                   kernel_regularizer=l2(0.01))(conv3)
    conv3 = BatchNormalization()(conv3)
    # conv3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # pool3 = Dropout(0.5)(pool3)
    # dropout1 = Dropout(0.5)(pool3)
    
    # Block 4
    conv4 = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', 
                   kernel_regularizer=l2(0.01))(pool3)
    # conv4 = BatchNormalization()(conv4)
    # conv4 = Dropout(0.5)(conv4)
    conv4 = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', 
                   kernel_regularizer=l2(0.01))(conv4)
    conv4 = BatchNormalization()(conv4)
    # conv4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # pool4 = Dropout(0.5)(pool4)
    # dropout2 = Dropout(0.5)(pool4)
    
    # Block 5
    conv5 = Conv2D(1024, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(pool4)
    # conv5 = BatchNormalization()(conv5)
    # conv5 = Dropout(0.5)(conv5)
    conv5 = Conv2D(1024, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(conv5)
    conv5 = BatchNormalization()(conv5)
    # pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(conv5)
    # conv5 = Dropout(0.5)(conv5)
    
    # Block 6
    # axis = -1 same as axis = last dim order, in this case axis = 3
    up6 = concatenate([Conv2DTranspose(512, kernel_size=(2, 2), strides=(2, 2), 
                                       padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(up6) 
    # conv6 = BatchNormalization()(conv6)
    # conv6 = Dropout(0.5)(conv6)
    conv6 = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(conv6)
    conv6 = BatchNormalization()(conv6)
    # conv6 = Dropout(0.5)(conv6)
    
    # Block 7
    up7 = concatenate([Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2), 
                                       padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(up7) 
    # conv7 = BatchNormalization()(conv7)
    # conv7 = Dropout(0.5)(conv7)
    conv7 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(conv7)
    conv7 = BatchNormalization()(conv7)
    # conv7 = Dropout(0.5)(conv7)
    
    # Block 8
    up8 = concatenate([Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), 
                                       padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(up8)
    # conv8 = BatchNormalization()(conv8)
    # conv8 = Dropout(0.5)(conv8)
    conv8 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(conv8)
    conv8 = BatchNormalization()(conv8)
    # conv8 = Dropout(0.5)(conv8)
    
    # Block 9
    up9 = concatenate([Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), 
                                       padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(up9) 
    # conv9 = BatchNormalization()(conv9)
    # conv9 = Dropout(0.5)(conv9)
    conv9 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(conv9)
    conv9 = BatchNormalization()(conv9)
    # conv9 = Dropout(0.5)(conv9)
    
    # Block 10
    conv10 = Conv2D(nclasses, kernel_size=(1, 1), padding='same')(conv9)
    
    # Block 10
    # curr_channels = nclasses
    _, curr_width, curr_height, curr_channels = conv10._keras_shape
    out_conv10 = Reshape((curr_width * curr_height, curr_channels))(conv10)
    act_soft = Activation('softmax')(out_conv10)
    
    model = Model(inputs=[inputs], outputs=[act_soft])


    sgd = SGD(lr=0.0001, decay=0.01, momentum=0.9, nesterov=True)    
    # sgd = SGD(lr=0.0001, decay=1, momentum=0.9, nesterov=True)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy']) 
    
    # adam = Adam(lr=0.0001)
    # model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy']) 
    
    # model.compile(optimizer='sgd', loss=dice_coef_loss, metrics=[dice_coef])
    
    # model.summary()
    return model

# inspired by "Get acquainted with U-NET architecture + some Keras shortcuts"
def cnnBRATsInit_unet_2(): 
    # the output layer is softmax, loss should be categorical_crossentropy
    # Downsampling 2^4 = 16 ==> 240/16 = 15
    
    inputs = Input((img_rows, img_cols, 3))        
    # inputs = Input((img_rows, img_cols, 4))
    
    # Block 1
    conv1 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(inputs)    
    conv1 = BatchNormalization()(conv1)
    # conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    # conv1 = Dropout(0.2)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)    
    
    # Block 2
    conv2 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(pool1) 
    conv2 = BatchNormalization()(conv2)
    # conv2 = Dropout(0.5)(conv2)
    conv2 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    # conv2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # pool2 = Dropout(0.5)(pool2)
    
    # this MaxPooling2D has the same result with the above line.
    # pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv2)
    
    # W_new = (W - F + 2P)/S + 1
    # W_new = (120 - 3 + 2*1)/2 + 1 = 60.5 --> 60 (round down)
    # P = 1, because the padding parameter is 'same', the size of W has no change
    # before divide by Stride parameter
    
    
    # Block 3
    conv3 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(pool2)  
    conv3 = BatchNormalization()(conv3)
    # conv3 = Dropout(0.5)(conv3)
    conv3 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    # conv3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # pool3 = Dropout(0.5)(pool3)
    # dropout1 = Dropout(0.5)(pool3)
    
    # Block 4
    conv4 = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    # conv4 = Dropout(0.5)(conv4)
    conv4 = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    # conv4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # pool4 = Dropout(0.5)(pool4)
    # dropout2 = Dropout(0.5)(pool4)
    
    # Block 5
    conv5 = Conv2D(1024, kernel_size=(3, 3), activation='relu', padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    # conv5 = Dropout(0.5)(conv5)
    conv5 = Conv2D(1024, kernel_size=(3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    # pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(conv5)
    # conv5 = Dropout(0.5)(conv5)
    
    # Block 6
    # axis = -1 same as axis = last dim order, in this case axis = 3
    up6 = concatenate([Conv2DTranspose(512, kernel_size=(2, 2), strides=(2, 2), 
                                       padding='same')(conv5), conv4], axis=3)
    drop1 = Dropout(0.5)(up6)
    conv6 = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same')(drop1) 
    # conv6 = BatchNormalization()(conv6)
    # conv6 = Dropout(0.5)(conv6)
    conv6 = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same')(conv6)
    # conv6 = BatchNormalization()(conv6)
    # conv6 = Dropout(0.5)(conv6)
    
    # Block 7
    up7 = concatenate([Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2), 
                                       padding='same')(conv6), conv3], axis=3)
    drop2 = Dropout(0.5)(up7)
    conv7 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(drop2) 
    # conv7 = BatchNormalization()(conv7)
    # conv7 = Dropout(0.5)(conv7)
    conv7 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(conv7)
    # conv7 = BatchNormalization()(conv7)
    # conv7 = Dropout(0.5)(conv7)
    
    # Block 8
    up8 = concatenate([Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), 
                                       padding='same')(conv7), conv2], axis=3)
    drop3 = Dropout(0.5)(up8)
    conv8 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(drop3)
    # conv8 = BatchNormalization()(conv8)
    # conv8 = Dropout(0.5)(conv8)
    conv8 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(conv8)
    # conv8 = BatchNormalization()(conv8)
    # conv8 = Dropout(0.5)(conv8)
    
    # Block 9
    up9 = concatenate([Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), 
                                       padding='same')(conv8), conv1], axis=3)
    drop4 = Dropout(0.5)(up9)
    conv9 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(drop4) 
    # conv9 = BatchNormalization()(conv9)
    # conv9 = Dropout(0.5)(conv9)
    conv9 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(conv9)
    # conv9 = BatchNormalization()(conv9)
    # conv9 = Dropout(0.5)(conv9)
    
    # Block 10
    conv10 = Conv2D(nclasses, kernel_size=(1, 1), padding='same')(conv9)
    
    # Block 10
    # curr_channels = nclasses
    _, curr_width, curr_height, curr_channels = conv10._keras_shape
    out_conv10 = Reshape((curr_width * curr_height, curr_channels))(conv10)
    act_soft = Activation('softmax')(out_conv10)
    
    model = Model(inputs=[inputs], outputs=[act_soft])


    sgd = SGD(lr=0.0001, decay=0.01, momentum=0.9, nesterov=True)    
    # sgd = SGD(lr=0.0001, decay=1, momentum=0.9, nesterov=True)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy']) 
    
    # adam = Adam(lr=0.0001)
    # model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy']) 
    
    # model.compile(optimizer='sgd', loss=dice_coef_loss, metrics=[dice_coef])
    
    # model.summary()
    return model

# inspired by "Kaggle Carvana Image Masking Challenge"
def cnnBRATsInit_unet_3(): 
    # the output layer is softmax, loss should be categorical_crossentropy
    # Downsampling 2^4 = 16 ==> 240/16 = 15
    
    inputs = Input((img_rows, img_cols, 3))        
    # inputs = Input((img_rows, img_cols, 4))
    # Normalize all data before extracting features
    input_nor = BatchNormalization()(inputs)
    
    # Block 1
    conv1 = Conv2D(64, kernel_size=(3, 3), padding='same')(input_nor)    
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(64, kernel_size=(3, 3), padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)    
    
    # Block 2
    conv2 = Conv2D(128, kernel_size=(3, 3), padding='same')(pool1) 
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(128, kernel_size=(3, 3), padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
    # this MaxPooling2D has the same result with the above line.
    # pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv2)
    
    # W_new = (W - F + 2P)/S + 1
    # W_new = (120 - 3 + 2*1)/2 + 1 = 60.5 --> 60 (round down)
    # P = 1, because the padding parameter is 'same', the size of W has no change
    # before divide by Stride parameter
        
    # Block 3
    conv3 = Conv2D(256, kernel_size=(3, 3), padding='same')(pool2)  
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(256, kernel_size=(3, 3), padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
    # Block 4
    conv4 = Conv2D(512, kernel_size=(3, 3), padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(512, kernel_size=(3, 3), padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Block 5
    conv5 = Conv2D(1024, kernel_size=(3, 3), padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(1024, kernel_size=(3, 3), padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    
    # Block 6
    # axis = -1 same as axis = last dim order, in this case axis = 3
    up6 = concatenate([Conv2DTranspose(512, kernel_size=(2, 2), strides=(2, 2), 
                                       padding='same')(conv5), conv4], axis=3)    
    conv6 = Conv2D(512, kernel_size=(3, 3), padding='same')(up6) 
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(512, kernel_size=(3, 3), padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    
    # Block 7
    up7 = concatenate([Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2), 
                                       padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(256, kernel_size=(3, 3), padding='same')(up7) 
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(256, kernel_size=(3, 3), padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    
    # Block 8
    up8 = concatenate([Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), 
                                       padding='same')(conv7), conv2], axis=3)    
    conv8 = Conv2D(128, kernel_size=(3, 3), padding='same')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(128, kernel_size=(3, 3), padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    
    # Block 9
    up9 = concatenate([Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), 
                                       padding='same')(conv8), conv1], axis=3)    
    conv9 = Conv2D(64, kernel_size=(3, 3), padding='same')(up9) 
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(64, kernel_size=(3, 3), padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    
    # Block 10
    conv10 = Conv2D(nclasses, kernel_size=(1, 1), padding='same')(conv9)
    
    # Block 10
    # curr_channels = nclasses
    _, curr_width, curr_height, curr_channels = conv10._keras_shape
    out_conv10 = Reshape((curr_width * curr_height, curr_channels))(conv10)
    act_soft = Activation('softmax')(out_conv10)
    
    model = Model(inputs=[inputs], outputs=[act_soft])


    sgd = SGD(lr=0.0001, decay=0.01, momentum=0.9, nesterov=True)    
    # sgd = SGD(lr=0.0001, decay=1, momentum=0.9, nesterov=True)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy']) 
    
    # adam = Adam(lr=0.0001)
    # model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy']) 
    
    # model.compile(optimizer='sgd', loss=dice_coef_loss, metrics=[dice_coef])
    
    # model.summary()
    return model

def cnnBRATsInit_unet_4():  
    # using PReLU activation
    
    # the output layer is softmax, loss should be categorical_crossentropy
    # Downsampling 2^4 = 16 ==> 240/16 = 15
    
    inputs = Input((img_rows, img_cols, 3))        
    # inputs = Input((img_rows, img_cols, 4))
    # Normalize all data before extracting features
    input_nor = BatchNormalization()(inputs)
    
    # Block 1
    conv1 = Conv2D(64, kernel_size=(3, 3), padding='same')(input_nor)
    # conv1 = BatchNormalization()(conv1)
    conv1 = PReLU()(conv1)
    # conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, kernel_size=(3, 3), padding='same')(conv1)
    # conv1 = BatchNormalization()(conv1)
    conv1 = PReLU()(conv1)    
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)    
    
    # Block 2
    conv2 = Conv2D(128, kernel_size=(3, 3), padding='same')(pool1) 
    # conv2 = BatchNormalization()(conv2)
    conv2 = PReLU()(conv2)
    # conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, kernel_size=(3, 3), padding='same')(conv2)
    # conv2 = BatchNormalization()(conv2)
    conv2 = PReLU()(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # pool2 = Dropout(0.5)(pool2)
    
    # this MaxPooling2D has the same result with the above line.
    # pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv2)
    
    # W_new = (W - F + 2P)/S + 1
    # W_new = (120 - 3 + 2*1)/2 + 1 = 60.5 --> 60 (round down)
    # P = 1, because the padding parameter is 'same', the size of W has no change
    # before divide by Stride parameter
    
    
    # Block 3
    conv3 = Conv2D(256, kernel_size=(3, 3), padding='same')(pool2)
    # conv3 = BatchNormalization()(conv3)
    conv3 = PReLU()(conv3)    
    # conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, kernel_size=(3, 3), padding='same')(conv3)
    # conv3 = BatchNormalization()(conv3)
    conv3 = PReLU()(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # pool3 = Dropout(0.5)(pool3)
    # dropout1 = Dropout(0.5)(pool3)
    
    # Block 4
    conv4 = Conv2D(512, kernel_size=(3, 3), padding='same')(pool3)
    # conv4 = BatchNormalization()(conv4)
    conv4 = PReLU()(conv4)
    # conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, kernel_size=(3, 3), padding='same')(conv4)
    # conv4 = BatchNormalization()(conv4)
    conv4 = PReLU()(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # pool4 = Dropout(0.5)(pool4)
    # dropout2 = Dropout(0.5)(pool4)
    
    # Block 5
    conv5 = Conv2D(1024, kernel_size=(3, 3), padding='same')(pool4)
    # conv5 = BatchNormalization()(conv5)
    conv5 = PReLU()(conv5)    
    # conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1024, kernel_size=(3, 3), padding='same')(conv5)
    # conv5 = BatchNormalization()(conv5)
    conv5 = PReLU()(conv5)
    conv5 = BatchNormalization()(conv5)
    
    # Block 6
    # axis = -1 same as axis = last dim order, in this case axis = 3
    up6 = concatenate([Conv2DTranspose(512, kernel_size=(2, 2), strides=(2, 2), 
                                       padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(512, kernel_size=(3, 3), padding='same')(up6) 
    # conv6 = BatchNormalization()(conv6)
    conv6 = PReLU()(conv6)
    # conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(512, kernel_size=(3, 3), padding='same')(conv6)
    # conv6 = BatchNormalization()(conv6)
    conv6 = PReLU()(conv6)    
    conv6 = BatchNormalization()(conv6)
    
    # Block 7
    up7 = concatenate([Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2), 
                                       padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(256, kernel_size=(3, 3), padding='same')(up7) 
    # conv7 = BatchNormalization()(conv7)
    conv7 = PReLU()(conv7)    
    # conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(256, kernel_size=(3, 3), padding='same')(conv7)
    # conv7 = BatchNormalization()(conv7)
    conv7 = PReLU()(conv7)    
    conv7 = BatchNormalization()(conv7)
    
    # Block 8
    up8 = concatenate([Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), 
                                       padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(128, kernel_size=(3, 3), padding='same')(up8)
    # conv8 = BatchNormalization()(conv8)
    conv8 = PReLU()(conv8)    
    # conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(128, kernel_size=(3, 3), padding='same')(conv8)
    # conv8 = BatchNormalization()(conv8)
    conv8 = PReLU()(conv8)    
    conv8 = BatchNormalization()(conv8)
    
    # Block 9
    up9 = concatenate([Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), 
                                       padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(64, kernel_size=(3, 3), padding='same')(up9) 
    # conv9 = BatchNormalization()(conv9)
    conv9 = PReLU()(conv9)   
    # conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(64, kernel_size=(3, 3), padding='same')(conv9)
    # conv9 = BatchNormalization()(conv9)
    conv9 = PReLU()(conv9)
    conv9 = BatchNormalization()(conv9)
    
    # Block 10
    conv10 = Conv2D(nclasses, kernel_size=(1, 1), padding='same')(conv9)
    
    # Block 10
    # curr_channels = nclasses
    _, curr_width, curr_height, curr_channels = conv10._keras_shape
    out_conv10 = Reshape((curr_width * curr_height, curr_channels))(conv10)
    act_soft = Activation('softmax')(out_conv10)
    
    model = Model(inputs=[inputs], outputs=[act_soft])

    sgd = SGD(lr=0.0001, decay=0.01, momentum=0.9, nesterov=True)
    # sgd = SGD(lr=0.0001, decay=1, momentum=0.9, nesterov=True)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy']) 
    
    # model.compile(optimizer='sgd', loss=dice_coef_loss, metrics=[dice_coef])
    
    return model

def cnnBRATsInit_unet_5():  
    # using LeakyReLU activation
    
    # the output layer is softmax, loss should be categorical_crossentropy
    # Downsampling 2^4 = 16 ==> 240/16 = 15
    
    inputs = Input((img_rows, img_cols, 3))        
    # inputs = Input((img_rows, img_cols, 4))
    # Normalize all data before extracting features
    input_nor = BatchNormalization()(inputs)
    
    # Block 1
    conv1 = Conv2D(64, kernel_size=(3, 3), padding='same')(input_nor)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU()(conv1)    
    conv1 = Conv2D(64, kernel_size=(3, 3), padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU()(conv1)    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)    
    
    # Block 2
    conv2 = Conv2D(128, kernel_size=(3, 3), padding='same')(pool1) 
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU()(conv2)    
    conv2 = Conv2D(128, kernel_size=(3, 3), padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU()(conv2)    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # pool2 = Dropout(0.5)(pool2)
    
    # this MaxPooling2D has the same result with the above line.
    # pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv2)
    
    # W_new = (W - F + 2P)/S + 1
    # W_new = (120 - 3 + 2*1)/2 + 1 = 60.5 --> 60 (round down)
    # P = 1, because the padding parameter is 'same', the size of W has no change
    # before divide by Stride parameter
    
    
    # Block 3
    conv3 = Conv2D(256, kernel_size=(3, 3), padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU()(conv3)    
    conv3 = Conv2D(256, kernel_size=(3, 3), padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU()(conv3)   
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # pool3 = Dropout(0.5)(pool3)
    # dropout1 = Dropout(0.5)(pool3)
    
    # Block 4
    conv4 = Conv2D(512, kernel_size=(3, 3), padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU()(conv4)    
    conv4 = Conv2D(512, kernel_size=(3, 3), padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU()(conv4)    
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # pool4 = Dropout(0.5)(pool4)
    # dropout2 = Dropout(0.5)(pool4)
    
    # Block 5
    conv5 = Conv2D(1024, kernel_size=(3, 3), padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU()(conv5)    
    conv5 = Conv2D(1024, kernel_size=(3, 3), padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU()(conv5)   
    
    # Block 6
    # axis = -1 same as axis = last dim order, in this case axis = 3
    up6 = concatenate([Conv2DTranspose(512, kernel_size=(2, 2), strides=(2, 2), 
                                       padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(512, kernel_size=(3, 3), padding='same')(up6) 
    conv6 = BatchNormalization()(conv6)
    conv6 = LeakyReLU()(conv6)    
    conv6 = Conv2D(512, kernel_size=(3, 3), padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = LeakyReLU()(conv6)    
    
    # Block 7
    up7 = concatenate([Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2), 
                                       padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(256, kernel_size=(3, 3), padding='same')(up7) 
    conv7 = BatchNormalization()(conv7)
    conv7 = LeakyReLU()(conv7)    
    conv7 = Conv2D(256, kernel_size=(3, 3), padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = LeakyReLU()(conv7)    
    
    # Block 8
    up8 = concatenate([Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), 
                                       padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(128, kernel_size=(3, 3), padding='same')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = LeakyReLU()(conv8)    
    conv8 = Conv2D(128, kernel_size=(3, 3), padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = LeakyReLU()(conv8)    
    
    # Block 9
    up9 = concatenate([Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), 
                                       padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(64, kernel_size=(3, 3), padding='same')(up9) 
    conv9 = BatchNormalization()(conv9)
    conv9 = LeakyReLU()(conv9)    
    conv9 = Conv2D(64, kernel_size=(3, 3), padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = LeakyReLU()(conv9)   
    
    # Block 10
    conv10 = Conv2D(nclasses, kernel_size=(1, 1), padding='same')(conv9)
    
    # Block 10
    # curr_channels = nclasses
    _, curr_width, curr_height, curr_channels = conv10._keras_shape
    out_conv10 = Reshape((curr_width * curr_height, curr_channels))(conv10)
    act_soft = Activation('softmax')(out_conv10)
    
    model = Model(inputs=[inputs], outputs=[act_soft])


    sgd = SGD(lr=0.0001, decay=0.01, momentum=0.9, nesterov=True)    
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy']) 
    
    # model.compile(optimizer='sgd', loss=dice_coef_loss, metrics=[dice_coef])    
    return model
    
def preprocessing_data(imgs_train_HG, imgs_label_HG, imgs_train_LG, imgs_label_LG):
    imgs_train = np.concatenate((imgs_train_HG, imgs_train_LG), axis=0)
    imgs_label = np.concatenate((imgs_label_HG, imgs_label_LG), axis=0)
    return imgs_train, imgs_label

def save_trained_model(model):
    # apply the histogram normalization method in pre-processing step    
    # serialize model to JSON
    model_json = model.to_json()
    with open("cnn_BRATs_unet_HN.json", "w") as json_file:    
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("cnn_BRATs_unet_HN.h5")   
    print("Saved model to disk")
    
def save_trained_model_IN(model): 
    # apply the intensity normalization method in pre-processing step
    # serialize model to JSON
    model_json = model.to_json()    
    with open("cnn_BRATs_unet_IN.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5    
    model.save_weights("cnn_BRATs_unet_IN.h5")
    print("Saved model to disk")

def save_trained_agu_model(model):   
    # serialize model to JSON
    model_json = model.to_json()
    with open("cnn_BRATs_unet_agu.json", "w") as json_file:    
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("cnn_BRATs_unet_agu.h5")    
    print("Saved model to disk")
    
def convert_data_toimage(imgs_pred):
    if imgs_pred.ndim == 3:
        nimgs, npixels, nclasses = imgs_pred.shape
        img_rows = np.sqrt(npixels).astype('int32') 
        img_cols = img_rows
        labels = True
    elif imgs_pred.ndim == 4:
        nimgs, img_rows, img_cols, _ = imgs_pred.shape
        labels = False
        
    for n in range(nimgs):
        # print(imgs_pred[n])
        if labels:
            imgs_temp = np.argmax(imgs_pred[n], axis=-1)
            # print(imgs_temp.shape)
            imgs_temp = np.reshape(imgs_temp, (img_rows, img_cols))
            # imgs_temp = imgs_temp == 4
            # print(imgs_temp.shape)            
            # c = np.count_nonzero(imgs_temp)
            # print(c)
        else:
            imgs_temp = imgs_pred[n]
            imgs_temp = np.reshape(imgs_temp, (img_rows, img_cols))
        
        imgs_temp = imgs_temp[np.newaxis, ...]
        if n == 0:
           imgs_result = imgs_temp 
        else:
           imgs_result = np.concatenate((imgs_result, imgs_temp), axis=0)           
        
    return imgs_result
    
def show_img(imgs_pred, imgs_label):
    for n in range(imgs_pred.shape[0]):
        print('Slice: %i' %n)
        img_pred = imgs_pred[n].astype('float32')
        img_label = imgs_label[n].astype('float32')               
               
        show_2Dimg(img_pred, img_label)
        
def show_2Dimg(img_1, img_2):
    fig, axes = plt.subplots(ncols=2)
    ax = axes.ravel()
    ax[0].imshow(img_1, cmap=plt.cm.gray)
    ax[1].imshow(img_2, cmap=plt.cm.gray)
    plt.show()

def train_network():
    print('Loading and preprocessing training data...')
#==============================================================================
#     imgs_train_HG, imgs_label_HG = load_train_data('HG')
#     imgs_train_LG, imgs_label_LG = load_train_data('LG')
#     print('HG shape', imgs_train_HG.shape, imgs_label_HG.shape)
#     print('LG shape', imgs_train_LG.shape, imgs_label_LG.shape)
#           
#     imgs_train, imgs_label_train = preprocessing_data(imgs_train_HG, imgs_label_HG,
#                                                imgs_train_LG, imgs_label_LG)
#==============================================================================
    
    imgs_train, imgs_label_train = load_train_data('HG')
    # imgs_train, imgs_label_train = load_train_data('Full_HG')
    print('Imgs train shape', imgs_train.shape)  
    print('Imgs label shape', imgs_label_train.shape)
    
    print('Calculating mean and std of training data...')
    imgs_train = imgs_train.astype('float32') 
    imgs_train /= 255. 
    
# =============================================================================
#     mean0 = np.mean(imgs_train[:, :, :, 0])  # mean of flair data
#     std0 = np.std(imgs_train[:, :, :, 0]) 
#     # print('mean0 = %f, std0 = %f' %(mean0, std0)) 
#     mean1 = np.mean(imgs_train[:, :, :, 1])  # mean of t1c data
#     std1 = np.std(imgs_train[:, :, :, 1]) 
#     # print('mean1 = %f, std1 = %f' %(mean1, std1))
#     mean2 = np.mean(imgs_train[:, :, :, 2])  # mean of t2 data
#     std2 = np.std(imgs_train[:, :, :, 2]) 
#     # print('mean2 = %f, std2 = %f' %(mean2, std2))  
#     
#     imgs_train[:, :, :, 0] -= mean0
#     imgs_train[:, :, :, 0] /= std0
#     imgs_train[:, :, :, 1] -= mean1
#     imgs_train[:, :, :, 1] /= std1
#     imgs_train[:, :, :, 2] -= mean2
#     imgs_train[:, :, :, 2] /= std2
# =============================================================================
    
# =============================================================================
#     mean = np.mean(imgs_train)  # mean for data centering
#     std = np.std(imgs_train)  # std for data normalization
#     # print('mean = %f, std = %f' %(mean, std))    
#     # np.save('imgs_train_mean_std.npy', [mean, std])
#     
#     imgs_train -= mean
#     imgs_train /= std    
# =============================================================================
    
    minv = np.min(imgs_train)  # mean for data centering
    maxv = np.max(imgs_train)  # std for data normalization
    print('min = %f, max = %f' %(minv, maxv))  
           
    print('Creating and compiling model...')    
    # model = cnnBRATsInit_unet()
    # model = cnnBRATsInit_unet_2() # inspired by "Get acquainted with U-NET architecture"
    # model = cnnBRATsInit_unet_3() # inspired by "Kaggle Carvana Image Masking Challenge"
    model = cnnBRATsInit_unet_4() # using PReLU activation
    # model = cnnBRATsInit_unet_5() # using LeakyReLU activation
    model.summary()
    
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
    
    print('Fitting model...')
    batch_size = 8
    history = model.fit(imgs_train, imgs_label_train, batch_size=batch_size, epochs=50, 
                        verbose=1, shuffle=True, validation_split=0.05,
                        callbacks=[model_checkpoint])
    
    save_trained_model(model)
    # save_trained_model_IN(model)
    
    # release memory in GPU and RAM
    del history
    del model
    for i in range(30):
        gc.collect()
    
    # print('Evaluating model...')
    # scores = model.evaluate(imgs_train, imgs_label_train, batch_size=4, verbose=0)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
def train_network_aug():
    # Apply a set of data augmentation methods
    print('Loading training data...')
    imgs_train, imgs_label_train = load_train_data('HG')
    print('Imgs train shape', imgs_train.shape)
    print('Imgs label shape', imgs_label_train.shape)
    
    # print('Loading validation data...')
    imgs_val, imgs_label_val = load_val_data()
    print('Imgs validation shape', imgs_val.shape)
    print('Imgs validation label shape', imgs_label_val.shape)
             
    print('Augmenting the training and validation data...')
    imgs_train = imgs_train.astype('float32') 
    imgs_val = imgs_val.astype('float32')
    imgs_train /= 255. 
    imgs_val /= 255.
    # imgs_train_1 = imgs_train[:, :, :, 0]
    # imgs_train_1 = imgs_train_1[..., np.newaxis]
    # print(imgs_train_1.shape)
    
    # define data preparation
    batch_size = 12
    datagen = ImageDataGenerator(featurewise_center=True,
                                 featurewise_std_normalization=True,
                                 rotation_range=20, # rescale=1./255,
                                 width_shift_range=0.1, height_shift_range=0.1,
                                 horizontal_flip=True, vertical_flip=True,                                 
                                 shear_range=0.2, fill_mode='nearest')
#==============================================================================
#     datagen = ImageDataGenerator(rotation_range=20, # rescale=1./255,
#                                  width_shift_range=0.1, height_shift_range=0.1,
#                                  horizontal_flip=True, vertical_flip=True,                                 
#                                  shear_range=0.2, fill_mode='nearest')
#==============================================================================
    # fit parameters from data
    datagen.fit(imgs_train)     
    # datagen.fit(imgs_train_1)
    train_generator = datagen.flow(imgs_train, imgs_label_train, batch_size=batch_size)
    
# =============================================================================
#     # Configure batch size and retrieve one batch of images
#     ni = 0
#     for X_batch, y_batch in datagen.flow(imgs_train_1, imgs_label_train, batch_size=batch_size):
#         ni += 1
#         # Show 9 images        
#         for i in range(0, 2):
#             print(X_batch[i].shape)
#             plt.subplot(330 + 1 + i)
#             plt.imshow(X_batch[i, :, :, 0], cmap=plt.cm.gray)
#         # show the plot
#         plt.show()
#         if ni > 4:
#             break
# =============================================================================

        
    # this is the augmentation configuration we will use for testing:
    val_datagen = ImageDataGenerator(featurewise_center=True,
                                     featurewise_std_normalization=True)
                                     # rescale=1./255)
    # val_datagen = ImageDataGenerator(rescale=1./255)
    
    # fit parameters from data
    val_datagen.fit(imgs_val) 
    # val_datagen.fit(imgs_train)   
    val_generator = val_datagen.flow(imgs_val, imgs_label_val, batch_size=batch_size)
    # datagen.standardize(imgs_val)
    
    print('Creating and compiling model...')    
    model = cnnBRATsInit_unet()
    model.summary()
    
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
    
    print('Fitting model with the data augmentation ...')    
    history = model.fit_generator(train_generator,                        
                                  steps_per_epoch=(imgs_train.shape[0] // batch_size) + 1,                        
                                  epochs=20,                        
                                  verbose=1,
                                  callbacks=[model_checkpoint],
                                  validation_data=val_generator,
                                  validation_steps=(imgs_val.shape[0] // batch_size) + 1)
    
    print('Predicting model with validation data...')
    prop = model.predict_generator(val_generator, steps=(imgs_val.shape[0] // batch_size) + 1)    
    # prop = model.predict(imgs_val, batch_size=batch_size, verbose=0)
    print(prop.shape)
    
    imgs_test_pred = convert_data_toimage(prop)
    print(imgs_test_pred.shape)
    imgs_label_test = convert_data_toimage(imgs_label_val)
    print(imgs_label_test.shape)
    show_img(imgs_test_pred, imgs_label_test)
    
    # save_trained_agu_model(model)   
    
    # release memory in GPU and RAM
    del history
    del model
    for i in range(15):
        gc.collect()
        
if __name__ == '__main__':
    train_network()
    # if the training grayscale data has channels greater than 1, it cannot 
    # be augmented using ImageDataGenerator
    # train_network_aug()   
