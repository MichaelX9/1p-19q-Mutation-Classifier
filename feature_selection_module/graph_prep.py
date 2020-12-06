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

dataPath = '/mnt/c/Users/Michael/Desktop/PyRadiomics/5_dilation/above_graph.csv'
savespot = '/mnt/c/Users/Michael/Desktop/PyRadiomics/5_dilation'

input = csv.DictReader(open(dataPath, encoding='utf-8-sig'))
dict_created = False
for row in input:
    if dict_created is False:
        data_dict = dict(row)
        for key in data_dict.keys():
            data_dict[key] = [data_dict[key]]
        dict_created = True
    else:
        for key in data_dict.keys():
            data_dict[key].append(dict(row)[key])
for key in data_dict.keys():
    for i in range(len(data_dict[key])):
        data_dict[key][i] = float(data_dict[key][i])
print(data_dict)
scipy.io.savemat(os.path.join(savespot, 'above_graph_data'), data_dict, True)