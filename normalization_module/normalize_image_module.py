#!/usr/bin/env python3
import os, sys, statistics, dicom2nifti, math
import numpy as np
import numpy.ma as ma
import nibabel as nib
import nipype.interfaces.fsl as fsl
import matplotlib as mpl
import scipy
from scipy import ndimage
mpl.use('Agg')
import matplotlib.pyplot as plt
from nipype.interfaces.fsl import BET
from intensity_normalization.normalize.zscore import zscore_normalize

def normalize_image(image_name, dir_name):
  imageID = image_name.split('.')[0]
  filename = os.path.join(dir_name, image_name)
  skullstrip = BET()
  skullstrip.inputs.in_file = filename
  skullstrip.inputs.mask = True
  skullstrip_file = imageID + '_strip.nii.gz'
  skullstrip.inputs.out_file = os.path.join(dir_name, skullstrip_file)
  res = skullstrip.run()
  normalized_file = imageID + '_normalizedstrip.nii.gz'
  normfile = os.path.join(dir_name, normalized_file)
  img = nib.load(os.path.join(dir_name, skullstrip_file))
  maskloc = os.path.join(dir_name, (imageID + '_strip_mask.nii.gz'))
  masker = nib.load(maskloc)
  mask_arr = masker.get_fdata()
  for i in range(0, mask_arr.shape[2]):
      mask_arr[:,:,i] = ndimage.morphology.binary_fill_holes(mask_arr[:,:,i]).astype('float64')
  #for i in range(0, mask_arr.shape[0]):
  #    mask_arr[i] = ndimage.morphology.binary_fill_holes(mask_arr[i]).astype('float64')
  #for i in range(0, mask_arr.shape[1]):
  #    mask_arr[:,i] = ndimage.morphology.binary_fill_holes(mask_arr[:,i]).astype('float64')
  new_mask = nib.Nifti1Image(mask_arr, masker.affine)
  nib.save(new_mask, maskloc)
  new_masker = nib.load(maskloc)
  normalized = zscore_normalize(img, new_masker)
  nib.save(normalized, normfile)
  return normalized_file


