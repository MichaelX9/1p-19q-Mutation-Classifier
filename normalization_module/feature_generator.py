#!/usr/bin/env python3
import os, sys
import SimpleITK as sitk
import six 
import pywt
import nibabel as nib
import numpy as np
import numpy.ma as ma
dataDir = '/mnt/c/Users/Michael/Desktop/PyRadiomics/pyradiomics'
sys.path.append(dataDir)
from radiomics import featureextractor

def generate_features(image_name, mask_name):
  imageName = os.path.join(dir_name, image_name)
  maskName = os.path.join(dir_name, mask_name)
  #image = nib.load(imageName)
  #mask = nib.load(maskName)
  ##idata = image.getfdata()
  #newdata = np.where(mask.get_fdata(), 50, 0)
  #new = nib.Nifti1Image(newdata, image.affine, image.header)
  #print(new.header)
  #nib.save(new, imageName)
  params = os.path.join(dataDir, "examples", "exampleSettings", "Params.yaml")
  extractor = featureextractor.RadiomicsFeatureExtractor(params)
  results = extractor.execute(imageName, maskName)
  for key, val in six.iteritems(results):
    print("\t%s: %s" %(key,val))



