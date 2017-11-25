# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 15:03:07 2017

@author: THANHHAI
"""
# Modified the original U-Net model with input (240, 240, 3)
from __future__ import print_function
import os
# force tensorflow to use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from skimage import data, util
# from skimage.measure import label, regionprops
from skimage import io, color
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_holes
# from skimage.transform import resize
import SimpleITK as sitk
from matplotlib import pyplot as plt
# import subprocess
# import random
# import progressbar
# from glob import glob
import h5py
import gc

# ----------------------------------------------------------------------------
import tensorflow as tf
# import keras
from keras.models import Sequential, Model, load_model,  model_from_json
# from keras.layers import Dense, Dropout, Activation, Flatten, Merge, Reshape
# from keras.layers import Input, MaxoutDense
# from keras.layers import Conv2D, MaxPooling2D, AtrousConv2D
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
# from keras.callbacks import ModelCheckpoint
from keras import backend as K

# TF dimension ordering in this code
# K.set_image_data_format('channels_last')

# an argument to your Session's config arguments, it might help a little, 
# though it won't release memory, it just allows growth at the cost of some memory efficiency
K.clear_session()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'
# with tf.Session(config = config) as s:
sess = tf.Session(config = config)
K.set_session(sess)

# ----------------------------------------------------------------------------
from BRATs_data_unet_2 import load_test_data, load_test_data_noGT
# from BRATs_unet_2 import preprocessing

# ----------------------------------------------------------------------------
# np.set_printoptions(threshold=np.inf) # help to print full array value in numpy
smooth = 1.
# img_rows = 64
# img_cols = 64

def dice_score(y_pred, y_true):
    y_true_f = y_true.flatten() # K.flatten(y_true.astype('float32'))
    y_pred_f = y_pred.flatten() # K.flatten(y_pred.astype('float32'))
    intersection = np.count_nonzero(y_true_f * y_pred_f)
    
    return (2. * intersection) / (np.count_nonzero(y_true_f) + np.count_nonzero(y_pred_f))

def dice_score_full(y_pred, y_true):
    # dice coef of entire tumor
    y_true_f = y_true.flatten() 
    y_pred_f = y_pred.flatten() 
    intersection = np.count_nonzero(y_true_f * y_pred_f)
    if intersection > 0:
        whole_tumor = (2. * intersection) / (np.count_nonzero(y_true_f) + np.count_nonzero(y_pred_f))
    else:
        whole_tumor = 0
    
    # dice coef of enhancing tumor
    enhan_gt = np.argwhere(y_true == 4)
    gt_a, seg_a = [], [] # classification of
    for i in enhan_gt:
        gt_a.append(y_true[i[0]][i[1]])
        seg_a.append(y_pred[i[0]][i[1]])
    gta = np.array(gt_a)
    sega = np.array(seg_a)
    if len(enhan_gt) > 0:
        enhan_tumor = float(len(np.argwhere(gta == sega))) / float(len(enhan_gt))
    else:
        enhan_tumor = 0
    
    # dice coef core tumor
    noenhan_gt = np.argwhere(y_true == 3)
    necrosis_gt = np.argwhere(y_true == 1)
    live_tumor_gt = np.append(enhan_gt, noenhan_gt, axis = 0)
    core_gt = np.append(live_tumor_gt, necrosis_gt, axis = 0)
    gt_core, seg_core = [], []
    for i in core_gt:
        gt_core.append(y_true[i[0]][i[1]])
        seg_core.append(y_pred[i[0]][i[1]])
    gtcore, segcore = np.array(gt_core), np.array(seg_core)
    if len(core_gt) > 0:
        core_tumor = float(len(np.argwhere(gtcore == segcore))) / float(len(core_gt))
    else:
        core_tumor = 0
    
    return whole_tumor, enhan_tumor, core_tumor

def load_trained_model():
    # apply the histogram normalization method in pre-processing step:   
    # load json and create model    
    json_file = open('cnn_BRATs_unet_HN.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model    
    loaded_model.load_weights("cnn_BRATs_unet_HN.h5")
    print("Loaded model from disk")
    
    return loaded_model

def load_trained_model_IN(): 
    # apply the intensity normalization method in pre-processing step
    # load json and create model
    json_file = open('cnn_BRATs_unet_IN.json', 'r')    
    loaded_model_json = json_file.read()
    json_file.close()
    
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("cnn_BRATs_unet_IN.h5")    
    print("Loaded model from disk")
    
    return loaded_model

def load_trained_agu_model():   
    # load json and create model    
    json_file = open('cnn_BRATs_unet_agu.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model    
    loaded_model.load_weights("cnn_BRATs_unet_agu.h5")
    print("Loaded model from disk")
    
    return loaded_model

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
        
    return imgs_result.astype('int16')

def show_img(imgs_pred, imgs_label):
    for n in range(imgs_pred.shape[0]):
        print('Slice: %i' %n)
        img_pred = imgs_pred[n].astype('float32')
        img_label = imgs_label[n].astype('float32')
        
        # dice = dice_score(imgs_pred[n], imgs_label[n])
        # print("Dice score: %.3f" %dice)
        whole_tumor, enhan_tumor, core_tumor = dice_score_full(img_pred, img_label)
        print("Whole tumor: %.3f, Enhancing tumor: %.3f, Core: %.3f" % (whole_tumor, enhan_tumor, core_tumor))
        
        
        img_1 = cvt2color_img(img_pred)
        img_2 = cvt2color_img(img_label)
        show_2Dimg(img_1, img_2)
        
        # show_2Dimg(imgs_pred[n], imgs_label[n])

def show_img_noGT(imgs_pred, imgs_orig):
    for n in range(imgs_pred.shape[0]):
        print('Slice: %i' %n)
        img_pred = imgs_pred[n].astype('float32')
        img_orig = imgs_orig[n].astype('float32')
                
        img_1 = cvt2color_img(img_pred)
        img_2 = img_orig
        show_2Dimg(img_1, img_2)
                
def show_2Dimg(img_1, img_2):
    fig, axes = plt.subplots(ncols=2)
    ax = axes.ravel()
    ax[0].imshow(img_1, cmap=plt.cm.gray)
    ax[1].imshow(img_2, cmap=plt.cm.gray)
    plt.show()

def show_img_3(imgs_seg1, imgs_seg2, imgs_orig):
    for n in range(imgs_seg1.shape[0]):
        print('Slice: %i' %n)
        img_seg1 = imgs_seg1[n].astype('float32')
        img_seg2 = imgs_seg2[n].astype('float32')
        img_orig = imgs_orig[n].astype('float32')
                
        img_1 = cvt2color_img(img_seg1)
        img_2 = cvt2color_img(img_seg2)
        # img_3 = img_orig
        img_3 = cvt2color_img(img_orig)
        show_2Dimg_3(img_1, img_2, img_3)

def show_img_32(imgs_seg1, imgs_seg2, imgs_label):
    for n in range(imgs_seg1.shape[0]):
        print('Slice: %i' %n)
        img_seg1 = imgs_seg1[n].astype('float32')
        img_seg2 = imgs_seg2[n].astype('float32')
        img_label = imgs_label[n].astype('float32')
                
        img_1 = cvt2color_img(img_seg1)
        img_2 = cvt2color_img(img_seg2)        
        img_3 = cvt2color_img(img_label)
        show_2Dimg_3(img_1, img_2, img_3)
                
def show_2Dimg_3(img_1, img_2, img_3):
    fig, axes = plt.subplots(ncols=3)
    ax = axes.ravel()
    ax[0].imshow(img_1, cmap=plt.cm.gray)
    ax[1].imshow(img_2, cmap=plt.cm.gray)
    ax[2].imshow(img_3, cmap=plt.cm.gray)
    plt.show()
        
def cvt2color_img(img_src):
    ones = np.argwhere(img_src == 1) # class 1/necrosis
    twos = np.argwhere(img_src == 2) # class 2/edema
    threes = np.argwhere(img_src == 3) # class 3/non-enhancing tumor
    fours = np.argwhere(img_src == 4) # class 4/enhancing tumor
    
    img_dst = color.gray2rgb(img_src)
    red_multiplier = [1, 0.2, 0.2] # class 1/necrosis    
    green_multiplier = [0.35, 0.75, 0.25] # class 2/edema
    blue_multiplier = [0, 0.25, 0.9] # class 3/non-enhancing tumor
    yellow_multiplier = [1, 1, 0.25] # class 4/enhancing tumor
# =============================================================================
#     img_dst[img_src == 1] = red_multiplier
#     img_dst[img_src == 2] = green_multiplier
#     img_dst[img_src == 3] = blue_multiplier
#     img_dst[img_src == 4] = yellow_multiplier
# =============================================================================
    
    # change colors of segmented classes
    for i in range(len(ones)):
        img_dst[ones[i][0]][ones[i][1]] = red_multiplier        
    for i in range(len(twos)):
        img_dst[twos[i][0]][twos[i][1]] = green_multiplier        
    for i in range(len(threes)):
        img_dst[threes[i][0]][threes[i][1]] = blue_multiplier
    for i in range(len(fours)):
        img_dst[fours[i][0]][fours[i][1]] = yellow_multiplier
        
    return img_dst
        
def save_result_h5py(nda):
    h5f = h5py.File('data.h5', 'w')
    h5f.create_dataset('data_1', data=nda)
    h5f.close()
    print("Saving data to disk done.")
    
def post_processing(imgs_src, imgs_pred):
    imgs_flair = imgs_src[:, :, :, 0]
    imgs_t1c = imgs_src[:, :, :, 1]
    imgs_t2 = imgs_src[:, :, :, 2]
    imgs_seg = imgs_pred
    
    # Step 1
    imgs_thresh = imgs_seg > 0
    
    # Labelling for 3D images
    # For a 3D image a connectivity = 1 is just the 6-neighborhood (or voxels that share an face), 
    # connectivity = 2 is then voxels that share an edge 
    # and 3 is voxels that share a vertex
    imgs_label = label(imgs_thresh, connectivity=3) 
        
    shape_flair = regionprops(imgs_label, imgs_flair)    
    # print('List of region properties for', len(shape_flair), 'regions')
            
    # sum_mean_intensity_flair = 0
    # for n in range(len(shape_flair)):        
        # sum_mean_intensity_flair += shape_flair[n].mean_intensity
        # print(shape_flair[n].mean_intensity)
    # print(sum_mean_intensity_flair)
    # Mean_flair = 0.8 * sum_mean_intensity_flair
    # print(Mean_flair)
    
    shape_t2 = regionprops(imgs_label, imgs_t2)    
    # print('List of region properties for', len(shape_t2), 'regions')
    # sum_mean_intensity_t2 = 0
    # for n in range(len(shape_t2)):
        # sum_mean_intensity_t2 += shape_t2[n].mean_intensity
        # print(shape_t2[n].mean_intensity)
    # print(sum_mean_intensity_t2)
    # Mean_t2 = 0.9 * sum_mean_intensity_t2
    # print(Mean_t2)
        
    for n in range(len(shape_flair)):
        if shape_flair[n].mean_intensity > 150 and shape_t2[n].mean_intensity > 150:
            voxel_pos = imgs_label == shape_flair[n].label
            imgs_seg[voxel_pos] = 0
    
    # Step 2
    # file = open('regions.txt', 'w') 
    imgs_thresh_2 = imgs_seg > 0 
    imgs_label_2 = label(imgs_thresh_2, connectivity=3) 
    shape_analysis = regionprops(imgs_label_2, imgs_flair)
    # print('List of region properties for', len(shape_analysis), 'regions')
    # file.write(str(len(shape_analysis)) + '\n')
    for n in range(len(shape_analysis)):
        label_pos = np.argwhere(imgs_label_2 == shape_analysis[n].label)        
        for i in range(len(label_pos)):
            V_flair = imgs_flair[label_pos[i][0]][label_pos[i][1]][label_pos[i][2]]
            V_t1c = imgs_t1c[label_pos[i][0]][label_pos[i][1]][label_pos[i][2]]
            V_t2 = imgs_t2[label_pos[i][0]][label_pos[i][1]][label_pos[i][2]]
            V_seg = imgs_seg[label_pos[i][0]][label_pos[i][1]][label_pos[i][2]]
            # print(V_flair, V_t1c, V_t2)
            # file.write(str(V_flair))
            # file.write(' ' + str(V_t1c))
            # file.write(' ' + str(V_t2) + '\n')
            
            # if V_flair < Mean_flair and V_t1c < 70 and V_t2 < Mean_t2 and V_seg < 4:
            if V_flair < 130 and V_t1c < 60 and V_t2 < 100 and V_seg < 4:
                imgs_seg[label_pos[i][0]][label_pos[i][1]][label_pos[i][2]] = 0
    
    # file.close()
                
    # Step 3    
    imgs_thresh_3 = imgs_seg > 0 
    imgs_label_3 = label(imgs_thresh_3, connectivity=3) 
    shape_analysis_2 = regionprops(imgs_label_3)
    # print('List of region properties for', len(shape_analysis_2), 'regions')
    
    maxArea = 0    
    for n in range(len(shape_analysis_2)):
        if maxArea < shape_analysis_2[n].area:
            maxArea = shape_analysis_2[n].area
    # print (maxArea)
    for n in range(len(shape_analysis_2)):
        maxArea_n = shape_analysis_2[n].area
        if (maxArea_n / maxArea) < 0.1:
            voxel_pos = imgs_label_3 == shape_analysis_2[n].label
            imgs_seg[voxel_pos] = 0
    
    # Step 4   
    # Fill the holes with 1, necrosis
    # Apply to Step 2, replace 0 with 1   
    
    # Step 5
    imgs_thresh_4 = imgs_seg > 0 
    imgs_label_4 = label(imgs_thresh_4, connectivity=3) 
    shape_analysis_3 = regionprops(imgs_label_4)
    # print('List of region properties for', len(shape_analysis_3), 'regions')
    
    for n in range(len(shape_analysis_3)):
        label_pos = np.argwhere(imgs_label_4 == shape_analysis_3[n].label)
        for i in range(len(label_pos)):
            V_t1c = imgs_t1c[label_pos[i][0]][label_pos[i][1]][label_pos[i][2]]
            V_seg = imgs_seg[label_pos[i][0]][label_pos[i][1]][label_pos[i][2]]
            
            if V_t1c < 60 and V_seg == 4:
                imgs_seg[label_pos[i][0]][label_pos[i][1]][label_pos[i][2]] = 1
                
    # Step 6
    whole_tumor_pos = np.argwhere(imgs_seg > 0)
    total_tumor = whole_tumor_pos.sum()
    enh_core_pos = np.argwhere(imgs_seg == 4)
    total_enh = enh_core_pos.sum()
        
    if (total_enh / total_tumor) < 0.05:
        label_pos = np.argwhere(imgs_seg == 2)
        for i in range(len(label_pos)):
            V_t1c = imgs_t1c[label_pos[i][0]][label_pos[i][1]][label_pos[i][2]]            
            if V_t1c < 45:
                imgs_seg[label_pos[i][0]][label_pos[i][1]][label_pos[i][2]] = 3    
    
    return imgs_seg

def test_network():
    # print('Loading mean and std of training data...')
    # mean, std = np.load('imgs_train_mean_std.npy')
    # print('mean = %f, std = %f' %(mean, std))
    
    print('Loading and preprocessing test data...')
    imgs_test, imgs_label_test = load_test_data()
    print('Test data shape: ', imgs_test.shape)
    print('Label data shape: ', imgs_label_test.shape)
    imgs_test_ref = imgs_test
                
    # imgs_test = preprocessing(imgs_test)
    imgs_test = imgs_test.astype('float32') 
    imgs_test /= 255.    
    
# =============================================================================
#     mean0 = np.mean(imgs_test[:, :, :, 0])  # mean of flair data
#     std0 = np.std(imgs_test[:, :, :, 0]) # std of flair data
#     # print('mean0 = %f, std0 = %f' %(mean0, std0)) 
#     mean1 = np.mean(imgs_test[:, :, :, 1])  # mean of t1c data
#     std1 = np.std(imgs_test[:, :, :, 1]) # std of t1c data
#     # print('mean1 = %f, std1 = %f' %(mean1, std1))
#     mean2 = np.mean(imgs_test[:, :, :, 2])  # mean of t2 data
#     std2 = np.std(imgs_test[:, :, :, 2]) # std of t2 data
#     # print('mean2 = %f, std2 = %f' %(mean2, std2))
#         
#     imgs_test[:, :, :, 0] -= mean0
#     imgs_test[:, :, :, 0] /= std0
#     imgs_test[:, :, :, 1] -= mean1
#     imgs_test[:, :, :, 1] /= std1
#     imgs_test[:, :, :, 2] -= mean2
#     imgs_test[:, :, :, 2] /= std2
# =============================================================================
    
# =============================================================================
#     mean = np.mean(imgs_test)  # mean for data centering
#     std = np.std(imgs_test)  # std for data normalization
#     # print('mean = %f, std = %f' %(mean, std))    
#     
#     imgs_test -= mean
#     imgs_test /= std
# =============================================================================
        
    minv = np.min(imgs_test)  # min val. of training data
    maxv = np.max(imgs_test)  # max val. of training data
    print('min = %f, max = %f' %(minv, maxv))  
    
    print('Loading the trained model...')
    model = load_trained_model()    
    # model = load_trained_model_IN()
    # model.summary()
    
    # evaluate loaded model on test data
    sgd = SGD(lr=0.0001, decay=0.01, momentum=0.9, nesterov=True)
    # sgd = SGD(lr=0.0001, decay=1, momentum=0.9)
    # model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy']) 
    
    # model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy']) 
        
    batch_size = 8
    
    # print('Evaluating model with test data...')
    # score = model.evaluate(imgs_test, imgs_label_test, batch_size=batch_size, verbose=0)    
    # print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))   
    
    print('Predicting model with test data...')    
    prop = model.predict(imgs_test, batch_size=batch_size, verbose=0)
    # prop = model.predict_generator(test_generator, steps=imgs_test.shape[0] // batch_size,
                                   # verbose=0)
    # print(prop.shape)
    
    imgs_test_pred = convert_data_toimage(prop)
    print(imgs_test_pred.shape)
    # maxv = np.max(imgs_test_pred)
    # print(maxv)
    imgs_label_test = convert_data_toimage(imgs_label_test)
    print(imgs_label_test.shape)
    
    # imgs_test[:, :, :, 0] *= std0
    # imgs_test[:, :, :, 0] += mean0
    # imgs_test[:, :, :, 1] *= std1
    # imgs_test[:, :, :, 1] += mean1
    # imgs_test[:, :, :, 2] *= std2
    # imgs_test[:, :, :, 2] += mean2
    # imgs_test *= 255.    
    # minv = np.min(imgs_test_ref)  # mean for data centering
    # maxv = np.max(imgs_test_ref)  # std for data normalization
    # print('min = %f, max = %f' %(minv, maxv))
    print('Postprocessing ...')    
    imgs_pred_post = post_processing(imgs_test_ref.astype('float32'), 
                                     imgs_test_pred.astype('float32'))
    
# =============================================================================
#     # convert a numpy array to a SimpleITK Image for saving
#     print('Saving the predicted result to mha file ...')
#     # imgs_sitk = sitk.GetImageFromArray(imgs_test_pred.astype('int16')) 
#     imgs_sitk = sitk.GetImageFromArray(imgs_pred_post.astype('int16'))
#     # print(imgs_sitk.GetSize())
#     # sitk.WriteImage(imgs_sitk, 'D:\mhafiles\Eval\VSD.Seg_HG_113.35537.mha')
#     sitk.WriteImage(imgs_sitk, 'D:\mhafiles\Eval\VSD.Seg_HG_IN_113.35537.mha')
# =============================================================================
    
    # show_img(imgs_test_pred, imgs_label_test)
    # show_img(imgs_pred_post, imgs_label_test)
    show_img_32(imgs_pred_post, imgs_test_pred, imgs_label_test)
    
    del model
    for i in range(30):
        gc.collect()

def test_network_aug():
    batch_size = 2   
    
    print('Loading and preprocessing test data...')
    imgs_test, imgs_label_test = load_test_data()
    print('Imgs test shape', imgs_test.shape)
    print('Imgs test label shape', imgs_label_test.shape)
                
    # imgs_test = preprocessing(imgs_test)
    imgs_test = imgs_test.astype('float32') 
    # imgs_test /= 255.
    
    # this is the augmentation configuration we will use for testing:    
#==============================================================================
#     test_datagen = ImageDataGenerator(featurewise_center=True,
#                                       featurewise_std_normalization=True,
#                                       rescale=1./255, rotation_range=20,
#                                       width_shift_range=0.1, height_shift_range=0.1,
#                                       horizontal_flip=True, vertical_flip=True,
#                                       zoom_range=[0.1, 0.1],
#                                       shear_range=0.2, fill_mode='nearest')
#==============================================================================
    test_datagen = ImageDataGenerator(featurewise_center=True,
                                      featurewise_std_normalization=True,
                                      rescale=1./255)    

    # fit parameters from data
    test_datagen.fit(imgs_test)     
    test_generator = test_datagen.flow(imgs_test, imgs_label_test, batch_size=batch_size)  
    
    print('Loading the trained model...')
    model = load_trained_agu_model()
    # model.summary()
    
    # evaluate loaded model on test data
    sgd = SGD(lr=0.0001, decay=0.01, momentum=0.9, nesterov=True)
    # sgd = SGD(lr=0.0001, decay=1, momentum=0.9)    
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy']) 
    
    # print('Evaluating model with test data...')
    # score = model.evaluate_generator(test_generator, steps=(imgs_test.shape[0] // batch_size) + 1)       
    # print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))    
    
    print('Predicting model with test data...')
    prop = model.predict_generator(test_generator, steps=(imgs_test.shape[0] // batch_size) + 1)
    print(prop.shape)
    
    # save_result_h5py(prop)
        
    imgs_test_pred = convert_data_toimage(prop)
    print(imgs_test_pred.shape)
    # maxv = np.max(imgs_test_pred)
    # print(maxv)
    imgs_label_test = convert_data_toimage(imgs_label_test)
    print(imgs_label_test.shape)
    
    # convert a numpy array to a SimpleITK Image for saving
    # print('Saving data ...')
    # imgs_sitk = sitk.GetImageFromArray(imgs_test_pred.astype('int16')) 
    # print(imgs_sitk.GetSize())
    # sitk.WriteImage(imgs_sitk, 'D:\mhafiles\Eval\VSD.Seg_HG_0006.54542.mha')
    # sitk.WriteImage(imgs_sitk, 'D:\mhafiles\Eval\VSD.Seg_HG_105.35533.mha')
        
    # show_img(imgs_test_pred, imgs_label_test)
        
def test_network_noGT():
    batch_size = 2    
    
    print('Loading and preprocessing test data ...')
    imgs_test = load_test_data_noGT()
    imgs_test_2 = load_test_data_noGT()
    print(imgs_test.shape)
                
    # imgs_test = preprocessing(imgs_test)
    imgs_test = imgs_test.astype('float32') 
    imgs_test /= 255.
    
    mean = np.mean(imgs_test)  # mean for data centering
    std = np.std(imgs_test)  # std for data normalization
    # print('mean = %f, std = %f' %(mean, std))    
    
    imgs_test -= mean
    imgs_test /= std
        
    minv = np.min(imgs_test)  # mean for data centering
    maxv = np.max(imgs_test)  # std for data normalization
    print('min = %f, max = %f' %(minv, maxv)) 
            
    print('Loading the trained model...')
    # model = load_trained_model()    
    model = load_trained_model_IN() 
    # model.summary()
    
    # evaluate loaded model on test data
    sgd = SGD(lr=0.0001, decay=0.01, momentum=0.9, nesterov=True)    
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy']) 
        
    print('Predicting model with test data...')
    prop = model.predict(imgs_test, batch_size=batch_size, verbose=0)
    print(prop.shape)    
    
    print('Convert the results for saving data ...')
    imgs_test_pred = convert_data_toimage(prop)
    print(imgs_test_pred.shape)
    
    imgs_pred_post = post_processing(imgs_test_2.astype('float32'), 
                                     imgs_test_pred.astype('float32'))    
        
    # convert a numpy array to a SimpleITK Image for saving
    # print('Saving data to mha file for evaluating online ...')
    # imgs_sitk = sitk.GetImageFromArray(imgs_test_pred.astype('int16')) 
    # print(imgs_sitk.GetSize())
    # sitk.WriteImage(imgs_sitk, 'D:\mhafiles\Eval\VSD.Seg_H_LG_x116.54281.mha')
    
    
    # copy a FLAIR MRI from imgs_test to show
    imgs_flair = imgs_test[:, :, :, 0] 
    # imgs_t1c = imgs_test[:, :, :, 1]
    # imgs_t2 = imgs_test[:, :, :, 2]
    # show_img_noGT(imgs_test_pred, imgs_flair)
    # show_img_noGT(imgs_pred_post, imgs_flair)
    show_img_3(imgs_pred_post, imgs_test_pred, imgs_flair)
        
if __name__ == '__main__':
    test_network()
    # test_network_aug()
    # test_network_noGT()
