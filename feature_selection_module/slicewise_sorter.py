### Image sorting by slice relation ####

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
import scipy.io
import csv
import pandas as pd

dataPath = '/mnt/c/Users/Michael/Desktop/PyRadiomics/GeneratedFeatures/slicewise-features/T2/selected_.5.csv'
abovePath = '/mnt/c/Users/Michael/Desktop/PyRadiomics/GeneratedFeatures/slicewise-features/T2/above_slices'
belowPath = '/mnt/c/Users/Michael/Desktop/PyRadiomics/GeneratedFeatures/slicewise-features/T2/below_slices'
centerPath = '/mnt/c/Users/Michael/Desktop/PyRadiomics/GeneratedFeatures/slicewise-features/T2/center_slices'

input = csv.DictReader(open(dataPath, encoding='utf-8-sig'))
above = []
below = []
center = []
tracker = 1
for row in input:
    if (tracker == 3):
        center.append(row)
        tracker = 1
    elif (tracker == 2):
        below.append(row)
        tracker += 1
    elif (tracker == 1):
        above.append(row)
        tracker += 1
above_df = pd.DataFrame(above, columns=above[0].keys())
below_df = pd.DataFrame(below, columns=below[0].keys())
center_df = pd.DataFrame(center, columns=center[0].keys())
above_df.to_csv(os.path.join(abovePath, 'selected_.5.csv'))
below_df.to_csv(os.path.join(belowPath, 'selected_.5.csv'))
center_df.to_csv(os.path.join(centerPath, 'selected_.5.csv'))
