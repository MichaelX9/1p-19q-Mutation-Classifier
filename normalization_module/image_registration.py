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
from nipype.interfaces.fsl import FLIRT
from intensity_normalization.normalize.zscore import zscore_normalize
import SimpleITK as sitk
import six 
import pywt
#dataDir = '/mnt/c/Users/Michael/Desktop/PyRadiomics/exception_testing'
#imageDir = '/mnt/c/Users/Michael/Desktop/PyRadiomics/ExceptionTesting/LGG-246'
#sys.path.append(dataDir)
from normalize_image_module import normalize_image
from slice_generator import *


def mask_reg(image_name, folder_name, mask_name):
    mask = os.path.join(folder_name, mask_name)
    MRImage = os.path.join(folder_name, image_name)
    image = sitk.ReadImage(MRImage)
    mask_i = sitk.ReadImage(mask)
    rif = sitk.ResampleImageFilter()
    rif.SetReferenceImage(image)
    rif.SetOutputPixelType(mask_i.GetPixelID())
    rif.SetInterpolator(sitk.sitkNearestNeighbor)
    #rif.SetOutputSpacing(image.GetSpacing())
    #rif.SetOutputOrigin(image.GetOrigin())
    #rif.SetOutputDirection(image.GetDirection())
    #for k in reader.GetMetaDataKeys():
    #    v = reader.GetMetaData(k)
    #    print("({0}) = = \"{1}\"".format(k,v))
    resMask = rif.Execute(mask_i)
    sitk.WriteImage(resMask, mask, False)
    #reader = sitk.ImageFileReader()
    #reader.SetFileName(MRImage)
    #reader.ReadImageInformation()
    #print(reader.GetDirection())
    #print(reader.GetOrigin())
    #print(reader.GetSpacing())
    #print(image.GetDirection())
    #print(image.GetSpacing())
    #print(image.GetOrigin())
    #reader.SetFileName(mask)
    #print(reader.GetDirection())
    #print(reader.GetOrigin())
    #print(reader.GetSpacing())
    #print(mask_i.GetDirection())
    #print(mask_i.GetSpacing())
    #print(mask_i.GetOrigin())
    #skullstrip = FLIRT()
    #skullstrip.inputs.in_file = mask
    #skullstrip.inputs.reference = MRImage
    #skullstrip.inputs.out_file = mask
    #res = skullstrip.run()