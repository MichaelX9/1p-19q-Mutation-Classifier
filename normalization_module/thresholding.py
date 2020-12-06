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
from nipype.interfaces.fsl import maths
from intensity_normalization.normalize.zscore import zscore_normalize
import SimpleITK as sitk
import six 
import pywt
#dataDir = '/mnt/c/Users/Michael/Desktop/PyRadiomics/exception_testing'
imageDir = '/mnt/c/Users/Michael/Desktop/PyRadiomics/ExceptionTesting/LGG-518/LGG-518-Segmentation.nii.gz'
#sys.path.append(dataDir)
from normalize_image_module import normalize_image
from slice_generator import *

threshold = maths.Threshold()
threshold.inputs.in_file = imageDir
threshold.inputs.thresh = 0
threshold.inputs.out_file = imageDir
res = threshold.run()
