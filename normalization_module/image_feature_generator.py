#!/usr/bin/env python3
import os, sys, statistics, dicom2nifti, math
import numpy as np
import numpy.ma as ma
import nibabel as nib
import nipype.interfaces.fsl as fsl
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy
from nipype.interfaces.fsl import BET
from intensity_normalization.normalize.zscore import zscore_normalize
import SimpleITK as sitk
import six 
import pywt
dataDir = '/mnt/c/Users/Michael/Desktop/PyRadiomics/pyradiomics'
imageDir = '/mnt/c/Users/Michael/Desktop/PyRadiomics/1_finished_dilation_new'
#'/mnt/c/Users/Michael/Desktop/PyRadiomics/NiFTiSegmentationsEdited'
sys.path.append(dataDir)
from radiomics import featureextractor
from normalize_image_module import normalize_image
from slice_generator import *
from image_registration import *

def name_generator(slice_pos, file_name):
  imageID = file_name.split('.')[0]
  if(slice_pos == 0):
    slice = 'b'
  if(slice_pos == 1):
    slice = 'c'
  if(slice_pos == 2):
    slice = 'a'
  if(slice_pos == 3):
    slice = 'e'
  return imageID + ('-%s' %slice)

directory = os.fsencode(imageDir)
print(directory)

for folder in os.listdir(directory):
  image_folder = os.fsencode(os.path.join(directory, folder))
  print(image_folder)
  patient_folder = os.path.join('/mnt/c/Users/Michael/Desktop/PyRadiomics/5_dilation_new', folder.decode('utf-8'))
  os.mkdir(patient_folder)
  for file in os.listdir(image_folder):
    file_name = os.fsencode(file)
    if('5_dilated' in file_name.decode('utf-8') and not('!' in file_name.decode('utf-8'))):
      mask_file = file_name
      #mask = nib.load(os.path.join(image_folder.decode('utf-8'), mask_file.decode('utf-8')))
      #print(mask.header)
  for img in os.listdir(image_folder):
    image_file = os.fsencode(img)
    ImageID = (image_file.decode('utf-8')).split('.')[0]
    if((not('Segmentation' in image_file.decode('utf-8'))) and (not('strip' in image_file.decode('utf-8'))) and (not('mask' in image_file.decode('utf-8')))):
      normalized_img = normalize_image(image_file.decode('utf-8'), image_folder.decode('utf-8'))
      mask_reg(image_file.decode('utf-8'), image_folder.decode('utf-8'), mask_file.decode('utf-8'))
      img_slices = slice_index(mask_file.decode('utf-8'), image_folder.decode('utf-8'))
      #normal = nib.load(os.path.join(image_folder.decode('utf-8'), normalized_img))
      #print(normal.header)
      for i in range(len(img_slices)):
        mask_slice_generator(mask_file.decode('utf-8'), img_slices[i], image_folder.decode('utf-8'))
        feature_dict = slice_feature_dict(normalized_img, image_folder.decode('utf-8'))
        save_loc = os.path.join(patient_folder,  name_generator(i, image_file.decode('utf-8')))
        scipy.io.savemat(save_loc, feature_dict, True)
        


