#!/usr/bin/env python3
import os, sys, re
import SimpleITK as sitk
import six
import nibabel as nib
import numpy as np
import numpy.ma as ma
import cv2 
imgDir = '/mnt/c/Users/Michael/Desktop/PyRadiomics/FinishedFeatureGen'
dataDir = '/mnt/c/Users/Michael/Desktop/PyRadiomics/pyradiomics'
sys.path.append(dataDir)
from radiomics import featureextractor

def threshold(mask_dat):
    for i in range(0,256):
        for c in range(0,256):
            if mask_dat[i][c] > 0:
                mask_dat[i][c] = 1

def edge_check(mask_dat, img_data):
    for i in range(0,256):
        for c in range(0,256):
            if img_data[i][c] <= 0:
                mask_dat[i][c] = 0
                #mask_dat[i][c-1] = 0
                #if(c < 255):
                #   mask_dat[i][c+1] = 0
                #mask_dat[i-1][c] = 0
                #if(i < 255):
                #    mask_dat[i+1][c] = 0

def make_mask(maskName, imgfold, dilations, imgName): 
    mask_arr = nib.load(maskName)
    mask_dat = mask_arr.get_fdata()
    img_arr = nib.load(imgName)
    img_dat = img_arr.get_fdata()
    for i in range(0, mask_arr.shape[2]):
        if(np.sum(mask_dat[:,:,i]) > 0):
            threshold(mask_dat[:,:,i])
            kernel = np.ones((3,3), np.uint8)
            mask_dat[:,:,i] = cv2.dilate(mask_dat[:,:,i], kernel, iterations=dilations)
            edge_check(mask_dat[:,:,i], img_dat[:,:,i])
    thresh_mask = nib.Nifti1Image(mask_dat, mask_arr.affine)
    nib.save(thresh_mask, os.path.join(imgfold.decode('utf-8'), (str(dilations) + "_dilated_mask.nii.gz")))

for folder in os.listdir(imgDir):
    image_folder = os.fsencode(os.path.join(imgDir, folder))
    for img in os.listdir(image_folder):
        image_file = os.fsencode(img)
        if('Segmentation' in image_file.decode('utf-8')):
            mask_file = image_file
        if('T1' in image_file.decode('utf-8')):
           img_file = image_file
    maskName = os.path.join(image_folder.decode('utf-8'), mask_file.decode('utf-8'))
    imgName = os.path.join(image_folder.decode('utf-8'), img_file.decode('utf-8'))
    for i in [1,2,3,4,5]:
        make_mask(maskName, image_folder, i, imgName)

