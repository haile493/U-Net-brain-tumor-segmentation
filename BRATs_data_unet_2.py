# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:11:51 2017

@author: THANHHAI
"""
# Modified the original U-Net model with input (240, 240, 3)
from __future__ import print_function
import numpy as np
# from skimage import data, util
from skimage.measure import label, regionprops
# from skimage import io
# from skimage.transform import resize
from skimage.filters import sobel
import SimpleITK as sitk
# from matplotlib import pyplot as plt
# import subprocess
# import random
# import progressbar
from glob import glob
# import os
import re
import gc
from NyulNormalizer import NyulNormalizer

import tensorflow as tf
import tensorlayer as tl
from keras.utils import to_categorical
from keras import backend as K

# ----------------------------------------------------------------------------
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

# np.set_printoptions(threshold=np.inf) # help to print full array value in numpy
nclasses = 5

def convert(str):
    return int("".join(re.findall("\d*", str)))

def hist_norm(imgs, BWM=0., GWM=255.):
    # BWM = 0; % a minimum desired level
    # GWM = 255; % a maximum desired level
    
    # slices, row, col
    nslices, insz_h, insz_w = imgs.shape[0], imgs.shape[1], imgs.shape[2]	 
    # print(nslices, insz_h, insz_w)
    converted_data = np.reshape(imgs, (1, nslices*insz_h*insz_w))
    converted_data = converted_data.astype('float32')
    gmin = np.min(converted_data); # a minimum level of original 3D MRI data
    gmax = np.max(converted_data); # a maximum level of original 3D MRI data
    # print (gmax)
    
    # Normalize between BWM and GWM
    converted_data = (GWM - BWM) * (converted_data - gmin) / (gmax - gmin) + BWM
           
    imgs_norm = np.reshape(converted_data, (nslices, insz_h, insz_w))
    return imgs_norm

# A deep learning model integrating FCNNs and CRFs for brain tumor seg
# Xiaomei Zhao, et. al.
def intensity_norm(imgs, sigma0=0., gray_val0=0., first_scan=True):
    # slices, row, col
    nslices, insz_h, insz_w = imgs.shape[0], imgs.shape[1], imgs.shape[2]
    converted_data = np.reshape(imgs, (1, nslices*insz_h*insz_w))
    converted_data = converted_data.astype('float32')
    
    BWM = 0.    # a minimum desired level
    GWM = 255.  # a maximum desired level
    gmin = np.min(converted_data) # a minimum level of original 3D MRI data
    gmax = np.max(converted_data) # a maximum level of original 3D MRI data
    # print (gmax)
    
    # Normalize between BWM and GWM
    converted_data = (GWM - BWM) * (converted_data - gmin) / (gmax - gmin) + BWM
            
    hist, _ = np.histogram(converted_data, bins=256)
    hist[0] = 0
    gray_val = np.argmax(hist) # gray level of highest histogram bin
    
    no_voxels = converted_data > 0 # find positions of the value greater than 0 in data
    N = no_voxels.sum() # total number of pixels is greater than 0
    converted_data[no_voxels] -= gray_val
    sum_val = np.square(converted_data[no_voxels]).sum()
    sigma = np.sqrt(sum_val/N)
    converted_data[no_voxels] /= sigma
    
    # if not first_scan:
    if first_scan: 
        converted_data[no_voxels] *= sigma
        converted_data[no_voxels] += gray_val
    else:
        converted_data[no_voxels] *= sigma0
        converted_data[no_voxels] += gray_val0
        
    no_data1 = converted_data < 0.
    converted_data[no_data1] = 0.
    no_data2 = converted_data > 255.
    converted_data[no_data2] = 255.
    
    # print(sigma)
    
    imgs_normed = np.reshape(converted_data, (nslices, insz_h, insz_w))
    
    return imgs_normed, sigma, gray_val

def Nyul_train(filename, MRI_type=0):
    nyul = NyulNormalizer()
    nyul.train(filename)
    if MRI_type == 0:
        nyul.saveTrainedModel('D:/mhafiles/model/nyulModel_Flair')
    elif MRI_type == 1:
        nyul.saveTrainedModel('D:/mhafiles/model/nyulModel_T1c')
    else:
        nyul.saveTrainedModel('D:/mhafiles/model/nyulModel_T2')
    
def Nyul_transform(imgs_src, MRI_type):
    nyul = NyulNormalizer()
    if MRI_type == 0:
        nyul.loadTrainedModel('D:/mhafiles/model/nyulModel_Flair.npz')
    elif MRI_type == 1:
        nyul.loadTrainedModel('D:/mhafiles/model/nyulModel_T1c.npz')
    else:
        nyul.loadTrainedModel('D:/mhafiles/model/nyulModel_T2.npz')
    
    imgs_normed = nyul.transform(imgs_src)
    
    return imgs_normed

# copied from https://github.com/zsdonghao/u-net-brain-tumor
def distort_imgs(data):
    """ data augmentation """
    x1, x2, x3, y = data
    
    # previous without this, hard-dice=83.7
    x1, x2, x3, y = tl.prepro.flip_axis_multi([x1, x2, x3, y], 
                                              axis=0, is_random=True) # up down
    x1, x2, x3, y = tl.prepro.flip_axis_multi([x1, x2, x3, y], 
                                              axis=1, is_random=True) # left right
    x1, x2, x3, y = tl.prepro.elastic_transform_multi([x1, x2, x3, y], 
                                                      alpha=720, sigma=24, is_random=True)
    x1, x2, x3, y = tl.prepro.rotation_multi([x1, x2, x3, y], rg=20, 
                                             is_random=True, fill_mode='constant') # nearest, constant
    x1, x2, x3, y = tl.prepro.shift_multi([x1, x2, x3, y], wrg=0.10, 
                                          hrg=0.10, is_random=True, fill_mode='constant')
    x1, x2, x3, y = tl.prepro.shear_multi([x1, x2, x3, y], 0.05, 
                                          is_random=True, fill_mode='constant')
    x1, x2, x3, y = tl.prepro.zoom_multi([x1, x2, x3, y], zoom_range=[0.9, 1.1], 
                                         is_random=True, fill_mode='constant')
    
    return x1, x2, x3, y

def read_scans(file_path1, data_tr_test=False):
    # nfiles = len(file_path1)
    scan_idx = 0
    for name in file_path1:
        # print ('\t', name)
        file_scan = sitk.ReadImage(name) # (240, 240, 155) = (rows, cols, slices)
        # if scan_idx == 99:
            # print ('\t', name)
        nda = sitk.GetArrayFromImage(file_scan) # convert to numpy array, (155, 240, 240)
        
        # print(nda.shape)
        if data_tr_test:
            nda = hist_norm(nda, 0., 255.)
            # print(nda.shape)
        
        if scan_idx == 0:
            nda_sum = nda            
            # print(gt_sum.shape)
        else:
            # nda_sum = np.append(nda_sum, nda, axis=0)
            nda_sum = np.concatenate((nda_sum, nda), axis=0) # faster
            # print(nda_sum.shape)
        
        # scan_idx += 1
        if scan_idx < 99: # for BRATS 2015
            scan_idx += 1
        else:
            break
    
    # print(nda_sum.shape)
    return nda_sum

def read_scans_Nyul(file_path1, MRI_type=0):
    # nfiles = len(file_path1)
    scan_idx = 0
    for name in file_path1:
        # print ('\t', name)
        file_scan = sitk.ReadImage(name) # (240, 240, 155) = (rows, cols, slices)
        # print(flairs_scan.GetSize())
        nda = Nyul_transform(file_scan, MRI_type) # convert to numpy array, (155, 240, 240)
        # print(nda.shape)
        nda = hist_norm(nda, 0., 255.)
        # print(nda.shape)
        
        if scan_idx == 0:
            nda_sum = nda            
            # print(gt_sum.shape)
        else:
            # nda_sum = np.append(nda_sum, nda, axis=0)
            nda_sum = np.concatenate((nda_sum, nda), axis=0) # faster
            # print(nda_sum.shape)
        
        scan_idx += 1
    
    # print(nda_sum.shape)
    return nda_sum

def read_scans_Nyul_IN(file_path1, MRI_type=0):
    if MRI_type == 0:
        sigma, graylevel = np.load('imgs_flair_sig_gray_Nyul.npy')
        print("sigma: %.3f, gray level: %i" % (sigma, graylevel))
    elif MRI_type == 1:
        sigma, graylevel = np.load('imgs_t1c_sig_gray_Nyul.npy')
        print("sigma: %.3f, gray level: %i" % (sigma, graylevel))
    else:
        sigma, graylevel = np.load('imgs_t2_sig_gray_Nyul.npy')
        print("sigma: %.3f, gray level: %i" % (sigma, graylevel))
        
    scan_idx = 0
    for name in file_path1:
        # print ('\t', name)
        file_scan = sitk.ReadImage(name) # (240, 240, 155) = (rows, cols, slices)
        # print(flairs_scan.GetSize())
        nda = Nyul_transform(file_scan, MRI_type) # convert to numpy array, (155, 240, 240)
        # print(nda.shape)
        nda, _, _ = intensity_norm(nda, sigma, graylevel, False)
        # print(nda.shape)
        
        if scan_idx == 0:
            nda_sum = nda            
            # print(gt_sum.shape)
        else:
            # nda_sum = np.append(nda_sum, nda, axis=0)
            nda_sum = np.concatenate((nda_sum, nda), axis=0) # faster
            # print(nda_sum.shape)
        
        scan_idx += 1
    
    # print(nda_sum.shape)
    return nda_sum

# apply the intensity normalization method
def read_scans_IN(file_path1, MRI_type=0):
    # nfiles = len(file_path1)
    if MRI_type == 0:
        # sigma, graylevel = np.load('imgs_flair1_sig_gray_NoN4.npy')
        sigma, graylevel = np.load('imgs_flair_sig_gray_N4.npy')
        print("sigma: %.3f, gray level: %i" % (sigma, graylevel))
    elif MRI_type == 1:
        # sigma, graylevel = np.load('imgs_t1c_sig_gray_NoN4.npy')
        sigma, graylevel = np.load('imgs_t1c_sig_gray_N4.npy')
        print("sigma: %.3f, gray level: %i" % (sigma, graylevel))
    else:
        # sigma, graylevel = np.load('imgs_t2_sig_gray_NoN4.npy')
        sigma, graylevel = np.load('imgs_t2_sig_gray_N4.npy')
        print("sigma: %.3f, gray level: %i" % (sigma, graylevel))
        
    scan_idx = 0
    for name in file_path1:
        # print ('\t', name)
        file_scan = sitk.ReadImage(name) # (240, 240, 155) = (rows, cols, slices)
        # print(flairs_scan.GetSize())
        nda = sitk.GetArrayFromImage(file_scan) # convert to numpy array, (155, 240, 240)        
        nda, _, _ = intensity_norm(nda, sigma, graylevel, False)
        # print(nda.shape)
        
        if scan_idx == 0:
            nda_sum = nda            
            # print(gt_sum.shape)
        else:            
            nda_sum = np.concatenate((nda_sum, nda), axis=0) # faster
            # print(nda_sum.shape)
        
        scan_idx += 1
    
    # print(nda_sum.shape)
    return nda_sum

# def resize_data(imgs_train1, imgs_train2, imgs_train3, imgs_train4, imgs_label):
def resize_data(imgs_train1, imgs_train2, imgs_train3, imgs_label):
    # prepare data for CNNs with the softmax activation
    nslices = 0
    for n in range(imgs_train1.shape[0]):
        label_temp = imgs_label[n] # imgs_label[n][:][:]
        edges = sobel(label_temp)
        # print(label_temp.shape)
        c = np.count_nonzero(edges)
        # print(c)   
        if c > 149:
            train_resz1 = imgs_train1[n] # keep the original size of data
            train_resz2 = imgs_train2[n]
            train_resz3 = imgs_train3[n]
            # train_resz4 = imgs_train4[n]
            train_resz1 = train_resz1[..., np.newaxis]
            train_resz2 = train_resz2[..., np.newaxis]
            train_resz3 = train_resz3[..., np.newaxis]
            # train_resz4 = train_resz4[..., np.newaxis]
            
            train_sum = np.concatenate((train_resz1, train_resz2, train_resz3), axis=-1)
            # train_sum = np.concatenate((train_sum, train_resz3), axis=-1) # 240, 240, 3
            # train_sum = np.concatenate((train_sum, train_resz4), axis=-1) # 240, 240, 4
            
            train_sum = train_sum[np.newaxis, ...] # 1, 240, 240, 3
                        
            label_resz = label_temp
            label_resz2 = np.reshape(label_resz, 240*240).astype('int32')
            label_resz2 = to_categorical(label_resz2, nclasses)
                       
            label_resz2 = label_resz2[np.newaxis, ...] # 1, 240*240, nclasses
            if nslices == 0:
                # flair_sum = np.asarray([flair_resz]) same as np.reshape(label_resz, (1, 64, 64))
                # gt_sum = np.asarray([gt_resz])
                data_sum = train_sum
                label_sum = label_resz2
            else:                
                data_sum = np.concatenate((data_sum, train_sum), axis=0) # faster                
                label_sum = np.concatenate((label_sum, label_resz2), axis=0)
            
            nslices += 1
    # print(train_sum.shape)
    return data_sum, label_sum

# def resize_data(imgs_train1, imgs_train2, imgs_train3, imgs_train4, imgs_label):
def resize_data_aug(imgs_train1, imgs_train2, imgs_train3, imgs_label):
    # prepare data for CNNs with the softmax activation
    # concat the original and augmented data
    nslices = 0
    for n in range(imgs_train1.shape[0]):
        label_temp = imgs_label[n] # imgs_label[n][:][:]; n, 240, 240
        edges = sobel(label_temp)
        # print(label_temp.shape)
        c = np.count_nonzero(edges)
        # print(c)   
        if c > 149:
            train_resz1 = imgs_train1[n] # keep the original size of data
            train_resz2 = imgs_train2[n]
            train_resz3 = imgs_train3[n]
            # train_resz4 = imgs_train4[n]
            train_resz1 = train_resz1[..., np.newaxis]
            train_resz2 = train_resz2[..., np.newaxis]
            train_resz3 = train_resz3[..., np.newaxis]
            # train_resz4 = train_resz4[..., np.newaxis]
            label_temp2 = label_temp.astype('int32')
            label_temp2 = label_temp2[..., np.newaxis]
            
            train_aug1, train_aug2, train_aug3, label_aug = distort_imgs([train_resz1, train_resz2, train_resz3, label_temp2])
            # print(train_aug1.shape, label_aug.shape) # 240, 240, 1
            
            train_sum = np.concatenate((train_resz1, train_resz2, train_resz3), axis=-1) # 240, 240, 3                      
            train_sum = train_sum[np.newaxis, ...] # 1, 240, 240, 3
            
            train_sum2 = np.concatenate((train_aug1, train_aug2, train_aug3), axis=-1) # 240, 240, 3
            train_sum2 = train_sum2[np.newaxis, ...] # 1, 240, 240, 3
                        
            # label_resz = label_temp
            label_resz = np.reshape(label_temp, 240*240).astype('int32')
            label_resz = to_categorical(label_resz, nclasses)
            label_resz = label_resz[np.newaxis, ...] # 1, 240*240, nclasses
            
            label_aug2 = label_aug[:, :, 0] # 240, 240, 1 to 240, 240
            label_resz2 = np.reshape(label_aug2, 240*240).astype('int32')
            label_resz2 = to_categorical(label_resz2, nclasses)                       
            label_resz2 = label_resz2[np.newaxis, ...] # 1, 240*240, nclasses
            
            if nslices == 0:                
                data_sum = np.concatenate((train_sum, train_sum2), axis=0) # faster                
                label_sum = np.concatenate((label_resz, label_resz2), axis=0)
            else:                
                data_sum = np.concatenate((data_sum, train_sum, train_sum2), axis=0) # faster                
                label_sum = np.concatenate((label_sum, label_resz, label_resz2), axis=0)
            
            nslices += 1
    
    return data_sum, label_sum

# for testing data
def resize_data_full(imgs_train1, imgs_train2, imgs_train3, imgs_label):
# def resize_data_full(imgs_train1, imgs_train2, imgs_train3, imgs_train4, imgs_label):
    
    # prepare data for CNNs with the softmax activation
    for n in range(imgs_label.shape[0]):        
        label_resz = imgs_label[n]
        label_resz2 = np.reshape(label_resz, 240*240).astype('int32')
        label_resz2 = to_categorical(label_resz2, nclasses)  
        label_resz2 = label_resz2[np.newaxis, ...] # 1, 240*240, nclasses                             
            
        if n == 0:            
            label_sum = label_resz2             
        else:
            label_sum = np.concatenate((label_sum, label_resz2), axis=0)                
    
    imgs_train1 = imgs_train1[..., np.newaxis] # 155, 240, 240, 1
    imgs_train2 = imgs_train2[..., np.newaxis]
    imgs_train3 = imgs_train3[..., np.newaxis]
    # concat data to 155, 240, 240, 3
    data_sum = np.concatenate((imgs_train1, imgs_train2, imgs_train3), axis=-1)
        
    return data_sum, label_sum

def resize_data_noGT(imgs_train1, imgs_train2, imgs_train3, imgs_train4):
    # prepare data for CNNs with the softmax activation
    nslices = 0
    for n in range(imgs_train1.shape[0]):        
        train_resz1 = imgs_train1[n] # keep the original size of data
        train_resz2 = imgs_train2[n]
        train_resz3 = imgs_train3[n]
        train_resz4 = imgs_train4[n]
        train_resz1 = train_resz1[..., np.newaxis]
        train_resz2 = train_resz2[..., np.newaxis]
        train_resz3 = train_resz3[..., np.newaxis]
        train_resz4 = train_resz4[..., np.newaxis]
            
        train_sum = np.concatenate((train_resz1, train_resz2), axis=-1)
        train_sum = np.concatenate((train_sum, train_resz3), axis=-1) # 240, 240, 3
        train_sum = np.concatenate((train_sum, train_resz4), axis=-1) # 240, 240, 4
            
        train_sum = train_sum[np.newaxis, ...] # 1, 240, 240, 4                                 
            
        if nslices == 0:            
            data_sum = train_sum                
        else:               
            data_sum = np.concatenate((data_sum, train_sum), axis=0) # faster                
            
        nslices += 1
    
    return data_sum

def create_train_data(type_data='HG'):
    # flairs = glob('D:\mhafiles\HGG\*\*Flair*\*Flair*.mha')
    # t1cs = glob('D:\mhafiles\HGG\*\*T1c*\*T1c.*.mha')
    # t2s = glob('D:\mhafiles\HGG\*\*T2*\*T2*.mha')
    # t1s = glob('D:\mhafiles\HGG\*\*T1*\*T1.*.mha')
    # gts = glob('D:\mhafiles\HGG\*\*OT*\*OT*.mha')
    
    if type_data == 'HG':
        # full HGG BRATS 2013 training data, 20 patients
        flairs = glob(r'D:\mhafiles\B2015_Training\HG\*2013*\*Flair*\N4ITK_*Flair*.mha')
        t1cs = glob(r'D:\mhafiles\B2015_Training\HG\*2013*\*T1c*\N4ITK_*T1c*.mha')
        t2s = glob(r'D:\mhafiles\B2015_Training\HG\*2013*\*T2*\N4ITK_*T2*.mha')
        # t1s = glob('D:\mhafiles\B2015_Training\HG\*2013*\*T1.*\*T1.*.mha')
        gts = glob('D:\mhafiles\B2015_Training\HG\*2013*\*OT*\*OT*.mha')
        # print(len(flairs))
    elif type_data == 'LG':
        # full LGG BRATS 2013 training data
        flairs = glob('D:\mhafiles\B2015_Training\LG\*2013*\*Flair*\*Flair*.mha')
        t1cs = glob('D:\mhafiles\B2015_Training\LG\*2013*\*T1c*\*T1c*.mha')
        t2s = glob('D:\mhafiles\B2015_Training\LG\*2013*\*T2*\*T2*.mha')
        # t1s = glob('D:\mhafiles\B2015_Training\LG\*2013*\*T1.*\*T1.*.mha')
        gts = glob('D:\mhafiles\B2015_Training\LG\*2013*\*OT*\*OT*.mha')
    elif type_data == 'Full_HG':
        # full HGG BRATS 2015 training data, 220 patients
        flairs = glob('D:\mhafiles\B2015_Training\HG\*\*Flair.*\*Flair*.mha')
        t1cs = glob('D:\mhafiles\B2015_Training\HG\*\*T1c.*\*T1c*.mha')
        t2s = glob('D:\mhafiles\B2015_Training\HG\*\*T2.*\*T2*.mha')
        # t1s = glob('D:\mhafiles\B2015_Training\HG\*\*T1.*\*T1.*.mha')
        gts = glob('D:\mhafiles\B2015_Training\HG\*\*OT.*\*OT*.mha')
        # print(len(flairs))
    
    flairs.sort(key=convert)
    t1cs.sort(key=convert)
    t2s.sort(key=convert)
    # t1s.sort(key=convert)
    gts.sort(key=convert)
    # print(flairs)
    # nfiles = len(flairs)
    
    flair_sum = read_scans(flairs, True)
    # flair_sum = read_scans_IN(flairs, 0)
    # flair_sum = read_scans_Nyul(flairs, 0)
    # flair_sum = read_scans_Nyul_IN(flairs, 0)
    print(flair_sum.shape)
    t1c_sum = read_scans(t1cs, True)
    # t1c_sum = read_scans_IN(t1cs, 1)
    # t1c_sum = read_scans_Nyul(t1cs, 1)
    # t1c_sum = read_scans_Nyul_IN(t1cs, 1)
    print(t1c_sum.shape)
    t2_sum = read_scans(t2s, True)
    # t2_sum = read_scans_IN(t2s, 2)
    # t2_sum = read_scans_Nyul(t2s, 2)
    # t2_sum = read_scans_Nyul_IN(t2s, 2)
    print(t2_sum.shape)
    # t1_sum = read_scans(t1s, True)
    # t1_sum = read_scans_IN(t1s, 2)
    # print(t1_sum.shape)
    gt_sum = read_scans(gts)
    print(gt_sum.shape)
    
    print('Combining training data for the softmax activation...')
    # total3_train, gt_train = resize_data(flair_sum, t1c_sum, t2_sum, gt_sum)
    # total3_train, gt_train = resize_data(flair_sum, t1c_sum, t2_sum, t1_sum, gt_sum)
    total3_train, gt_train = resize_data_aug(flair_sum, t1c_sum, t2_sum, gt_sum)
    print(total3_train.shape)
    print(gt_train.shape)
    
    if type_data == 'HG':      
        # full HGG data of BRATS 2013
        np.save('D:\mhafiles\Data\imgs_train_unet_HG.npy', total3_train)
        np.save('D:\mhafiles\Data\imgs_label_train_unet_HG.npy', gt_train)
        # np.save('D:\mhafiles\Data\imgs_train_unet_IN.npy', total3_train)
        # np.save('D:\mhafiles\Data\imgs_label_train_unet_IN.npy', gt_train)
        print('Saving all HGG training data to .npy files done.')          
    elif type_data == 'LG': 
        # full LGG data of BRATS 2013               
        np.save('D:\mhafiles\Data\imgs_train_unet_LG.npy', total3_train)
        np.save('D:\mhafiles\Data\imgs_label_train_unet_LG.npy', gt_train)
        print('Saving all LGG training data to .npy files done.') 
    elif type_data == 'Full_HG':      
        # full HGG data of BRATS 2015
        np.save('D:\mhafiles\Data\imgs_train_unet_FHG.npy', total3_train)
        np.save('D:\mhafiles\Data\imgs_label_train_unet_FHG.npy', gt_train)
        # np.save('D:\mhafiles\Data\imgs_train_unet_IN.npy', total3_train)
        # np.save('D:\mhafiles\Data\imgs_label_train_unet_IN.npy', gt_train)
        print('Saving all HGG training data to .npy files done.')
    else:
        print('Cannot save type of data as you want')   
    
    for i in range(30):
        gc.collect()

def load_train_data(type_data='HG'):
    if type_data == 'HG':        
        imgs_train = np.load('D:\mhafiles\Data\imgs_train_unet_HG.npy')
        imgs_label = np.load('D:\mhafiles\Data\imgs_label_train_unet_HG.npy')
        # imgs_train = np.load('D:\mhafiles\Data\imgs_train_unet_IN.npy')
        # imgs_label = np.load('D:\mhafiles\Data\imgs_label_train_unet_IN.npy')
    elif type_data == 'LG':
        imgs_train = np.load('D:\mhafiles\Data\imgs_train_unet_LG.npy')
        imgs_label = np.load('D:\mhafiles\Data\imgs_label_train_unet_LG.npy')
    elif type_data == 'Full_HG':
        imgs_train = np.load('D:\mhafiles\Data\imgs_train_unet_FHG.npy')
        imgs_label = np.load('D:\mhafiles\Data\imgs_label_train_unet_FHG.npy')
    else:
        print('No type of data as you want')
        
    return imgs_train, imgs_label
    
def create_test_data():    
    # flairs_test = glob('D:\mhafiles\HGG_Flair_6.mha')
    # t1cs_test = glob('D:\mhafiles\HGG_T1c_6.mha')
    # t2s_test = glob('D:\mhafiles\HGG_T2_6.mha')
    # gts = glob('D:\mhafiles\HGG_OT_6.mha')
    
    # BRATS 2015
    flairs_test = glob(r'D:\mhafiles\B2015_Training\HG\*pat113*\*Flair*\N4ITK_*Flair*.mha')
    t1cs_test = glob(r'D:\mhafiles\B2015_Training\HG\*pat113*\*T1c*\N4ITK_*T1c*.mha')
    t2s_test = glob(r'D:\mhafiles\B2015_Training\HG\*pat113*\*T2*\N4ITK_*T2*.mha')
    # t1s_test = glob('D:\mhafiles\B2015_Training\HG\*pat113*\*T1.*\*T1.*.mha')
    gts = glob('D:\mhafiles\B2015_Training\HG\*pat113*\*OT*\*OT*.mha') 
    # flairs_test = glob('D:\mhafiles\HGG_Flair_pat111.mha')
    # t1cs_test = glob('D:\mhafiles\HGG_T1c_pat111.mha')
    # t2s_test = glob('D:\mhafiles\HGG_T2_pat111.mha')
    # t1s_test = glob('D:\mhafiles\HGG_T1_pat105.mha')
    # gts = glob('D:\mhafiles\HGG_OT_pat111.mha') 
    # flairs_test = glob('D:\mhafiles\HGG_Flair_2.mha')
    # t1cs_test = glob('D:\mhafiles\HGG_T1c_2.mha')
    # t2s_test = glob('D:\mhafiles\HGG_T2_2.mha')
    # t1s_test = glob('D:\mhafiles\HGG_T1_2.mha')
    # gts = glob('D:\mhafiles\HGG_OT_2.mha') 
                
    flair_sum = read_scans(flairs_test, True)
    # flair_sum = read_scans_IN(flairs_test, 0)
    # flair_sum = read_scans_Nyul(flairs_test, 0)
    # flair_sum = read_scans_Nyul_IN(flairs_test, 0)
    print(flair_sum.shape)
    t1c_sum = read_scans(t1cs_test, True)
    # t1c_sum = read_scans_IN(t1cs_test, 1)
    # t1c_sum = read_scans_Nyul(t1cs_test, 1)
    # t1c_sum = read_scans_Nyul_IN(t1cs_test, 1)
    print(t1c_sum.shape)
    t2_sum = read_scans(t2s_test, True)
    # t2_sum = read_scans_IN(t2s_test, 2)
    # t2_sum = read_scans_Nyul(t2s_test, 2)
    # t2_sum = read_scans_Nyul_IN(t2s_test, 2)
    print(t2_sum.shape)
    # t1_sum = read_scans(t1s_test, True)
    # t1_sum = read_scans_IN(t1s_test, 2)
    # print(t1_sum.shape)
    gt_sum = read_scans(gts)
    print(gt_sum.shape)
    
    print('Resizing testing data for the softmax activation...')
    # total3_test, gt_test = resize_data(flair_sum, t1c_sum, t2_sum, gt_sum) # for validation
    total3_test, gt_test = resize_data_full(flair_sum, t1c_sum, t2_sum, gt_sum)
    # total3_test, gt_test = resize_data_full(flair_sum, t1c_sum, t2_sum, t1_sum, gt_sum)
    print(total3_test.shape)
    print(gt_test.shape)
        
    np.save('D:\mhafiles\Data\imgs_test_unet_HN.npy', total3_test)
    np.save('D:\mhafiles\Data\imgs_label_test_unet_HN.npy', gt_test)
    # np.save('D:\mhafiles\Data\imgs_test_unet_IN.npy', total3_test)
    # np.save('D:\mhafiles\Data\imgs_label_test_unet_IN.npy', gt_test)        
    print('Saving testing data to .npy files done.')
    
    for i in range(30):
        gc.collect()     
    
def create_test_data_noGT():          
    # Test data without ground truth
    flairs_test = glob('D:\mhafiles\H_LG_Test\*pat0105*\*Flair*\*Flair*.mha')
    t1cs_test = glob('D:\mhafiles\H_LG_Test\*pat0105*\*T1c*\*T1c*.mha')
    t2s_test = glob('D:\mhafiles\H_LG_Test\*pat0105*\*T2*\*T2*.mha')
    t1s_test = glob('D:\mhafiles\H_LG_Test\*pat0105*\*T1.*\*T1.*.mha')
    print(flairs_test)
        
    flair_sum = read_scans(flairs_test, True)
    # flair_sum = read_scans_IN(flairs_test, 0)
    print(flair_sum.shape)
    t1c_sum = read_scans(t1cs_test, True)
    # t1c_sum = read_scans_IN(t1cs_test, 1)
    print(t1c_sum.shape)
    t2_sum = read_scans(t2s_test, True)
    # t2_sum = read_scans_IN(t2s_test, 2)
    print(t2_sum.shape) 
    t1_sum = read_scans(t1s_test, True)
    # t1_sum = read_scans_IN(t1s_test, 2)
    print(t1_sum.shape)
    
    print('Combining testing data ...')
    # total3_test = resize_data_noGT(flair_sum, t1c_sum, t2_sum)
    total3_test = resize_data_noGT(flair_sum, t1c_sum, t2_sum, t1_sum)
    print(total3_test.shape)
            
    np.save('D:\mhafiles\Data\imgs_test_unet_noGT.npy', total3_test)             
    print('Saving testing data to .npy files done.')
    
    for i in range(30):
        gc.collect()
    
def load_test_data_noGT():
    imgs_test = np.load('D:\mhafiles\Data\imgs_test_unet_noGT.npy')
             
    return imgs_test

def load_test_data():
    imgs_test = np.load('D:\mhafiles\Data\imgs_test_unet_HN.npy')
    imgs_label_test = np.load('D:\mhafiles\Data\imgs_label_test_unet_HN.npy')
    # imgs_test = np.load('D:\mhafiles\Data\imgs_test_unet_IN.npy')
    # imgs_label_test = np.load('D:\mhafiles\Data\imgs_label_test_unet_IN.npy')
            
    return imgs_test, imgs_label_test

def load_val_data():
    imgs_val = np.load('D:\mhafiles\Data\imgs_val_unet.npy')
    imgs_label_val = np.load('D:\mhafiles\Data\imgs_label_val_unet.npy')    
            
    return imgs_val, imgs_label_val

def Nyul_find_landmarks():
    # full HGG BRATS 2013 training data
    # flairs = glob('D:\mhafiles\B2015_Training\HG\*2013*\*Flair*\*Flair*.mha')
    # t1cs = glob('D:\mhafiles\B2015_Training\HG\*2013*\*T1c*\*T1c*.mha')
    # t2s = glob('D:\mhafiles\B2015_Training\HG\*2013*\*T2*\*T2*.mha')
    # t1s = glob('D:\mhafiles\B2015_Training\HG\*2013*\*T1.*\*T1.*.mha')
    flairs = glob('D:\mhafiles\HGG_Flair_1.mha')
    t1cs = glob('D:\mhafiles\HGG_T1c_1.mha')
    t2s = glob('D:\mhafiles\HGG_T2_1.mha')
    
    Nyul_train(flairs, 0)
    Nyul_train(t1cs, 1)
    Nyul_train(t2s, 2)

if __name__ == '__main__':
    # Nyul_find_landmarks() # run once
    create_train_data('HG')
    # create_train_data('LG')
    # create_train_data('Full_HG')
    create_test_data() 
    # create_test_data_noGT()