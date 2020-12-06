#!/usr/bin/env python3
import os, sys, re
import SimpleITK as sitk
import six
import nibabel as nib
import numpy as np
import numpy.ma as ma
import cv2
dataDir = '/mnt/c/Users/Michael/Desktop/PyRadiomics/pyradiomics'
sys.path.append(dataDir)
from radiomics import featureextractor

def slice_index(mask_name, dir_name):
  maskName = os.path.join(dir_name, mask_name)
  mask_arr = nib.load(maskName)
  mask_dat = mask_arr.get_fdata()
  mask_slices = []
  for i in range(0, mask_arr.shape[2]):
    if(np.sum(mask_dat[:,:,i]) > 0):
      threshold(mask_dat[:,:,i])
      mask_slices.append(i)
  thresh_mask = nib.Nifti1Image(mask_dat, mask_arr.affine)
  nib.save(thresh_mask, maskName)
  return mask_slices

def threshold(mask_dat):
    for i in range(0,256):
        for c in range(0,256):
            if mask_dat[i][c] > 0:
                mask_dat[i][c] = 1

def mask_slice_generator(mask_name, slice_num, dir_name):
  maskName = os.path.join(dir_name, mask_name)
  mask_arr = nib.load(maskName)
  mask_dat = mask_arr.get_fdata()
  mask_slices = []
  new_name = os.path.join(dir_name, "_temp_mask.nii.gz")
  for i in range(0, mask_arr.shape[2]):
    if((np.sum(mask_dat[:,:,i]) > 0) & (i != slice_num)):
        mask_dat[:,:,i].fill(0)
  newm = nib.Nifti1Image(mask_dat, mask_arr.affine)
  nib.save(newm, new_name)


def slice_feature_dict(image_name, dir_name):
  imageName = os.path.join(dir_name, image_name)
  newMask = os.path.join(dir_name, "_temp_mask.nii.gz")
  image = nib.load(imageName)
  params = os.path.join(dataDir, "examples", "exampleSettings", "Params - Copy.yaml")
  extractor = featureextractor.RadiomicsFeatureExtractor(params)
  extractor.correctMask = True
  feature_dict = {}
  feature_list = []
  diag = 0
  glcm = 0
  glszm = 0
  glrlm = 0
  ngtdm = 0
  firstorder = 0
  shape = 0
  results = extractor.execute(imageName, newMask)
  for key, val in six.iteritems(results):
    #feature_list.append(key)
    if('diagnostics' in key):
    #   diag += 1
    #if('glcm' in key):
    #    glcm += 1
    #if('glszm' in key):
    #    glszm += 1
    #if('glrlm' in key):
    #    glrlm += 1
    #if('ngtdm' in key):
    #    ngtdm += 1
    #if('firstorder' in key):
    #    firstorder += 1
    #if('shape' in key):
    #    shape += 1
      pass
    else:
        feature_dict[key] = float(val)
  #print([diag, glcm, glszm, glrlm, ngtdm, firstorder, shape])
  return feature_dict



#idata = image.get_fdata()
#mdata = mask.get_fdata()
#new = nib.Nifti1Image(idata[:,:,slices[0]], image.affine)
#newm = nib.Nifti1Image(mdata[:,:,slices[0]], image.affine)
#nib.save(new, newName)
#nib.save(newm, newMask)
#print(new.header)


