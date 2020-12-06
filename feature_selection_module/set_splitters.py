#!/usr/bin/env python3
import os
import pandas as pd
import shutil
import numpy as np
dataPath = '/mnt/c/Users/Michael/Desktop/PyRadiomics/TestSetFeatures'
onePath = '/mnt/c/Users/Michael/Desktop/PyRadiomics/GeneratedFeatures/centerslices/T1'
twoPath = '/mnt/c/Users/Michael/Desktop/PyRadiomics/GeneratedFeatures/centerslices/T2'
a1Path = '/mnt/c/Users/Michael/Desktop/PyRadiomics/TestSetFeatures/aboveslices/T1'
b1path = '/mnt/c/Users/Michael/Desktop/PyRadiomics/TestSetFeatures/centerslices/T1'
c1path = '/mnt/c/Users/Michael/Desktop/PyRadiomics/TestSetFeatures/belowslices/T1'
a2Path = '/mnt/c/Users/Michael/Desktop/PyRadiomics/TestSetFeatures/aboveslices/T2'
b2path = '/mnt/c/Users/Michael/Desktop/PyRadiomics/TestSetFeatures/centerslices/T2'
c2path = '/mnt/c/Users/Michael/Desktop/PyRadiomics/TestSetFeatures/belowslices/T2'

#colnames = ['1', 'name', '3', '4', '5', '6', '7', '8']
#data = pd.read_csv(dataPath, names=colnames)
#test_set = data.name.tolist()
#for folder in os.listdir(dirPath):
#  image_folder = os.fsencode(os.path.join(dirPath, folder))
#  if(folder in test_set):
#      os.rename(image_folder, os.path.join(newPath, folder))

#perms = np.random.permutation(105)[0:11]
#i = 0
#for folder in os.listdir(dirPath):
#    image_folder = os.fsencode(os.path.join(dirPath, folder))
#    if('test' in folder):
#        break
    #if (i in perms):
    #    os.rename(image_folder, os.path.join(dataPath, folder))
    #i += 1
for folder in os.listdir(dataPath):
    for file in os.listdir(os.path.join(dataPath, folder)):
            file_name = os.fsencode(file)
            filepath = os.path.join(dataPath, folder, file_name.decode('utf-8'))
            #if('T1' in file):
            #    os.rename(filepath, os.path.join(onePath, file))
            #if('T2' in file):
            #    os.rename(filepath, os.path.join(twoPath, file))
             #if i in perms:
             #    os.rename(filepath, os.path.join(dirPath, file_name.decode('utf-8')))
             #i += 1
            if(('-a' in file_name.decode('utf-8')) and ('T1' in file_name.decode('utf-8'))):
                shutil.copy(filepath, os.path.join(a1Path, file_name.decode('utf-8')))
            if(('-b' in file_name.decode('utf-8')) and ('T1' in file_name.decode('utf-8'))):
                shutil.copy(filepath, os.path.join(b1path, file_name.decode('utf-8')))
            if(('-c' in file_name.decode('utf-8')) and ('T1' in file_name.decode('utf-8'))):
                shutil.copy(filepath, os.path.join(c1path, file_name.decode('utf-8')))
            if(('-a' in file_name.decode('utf-8')) and ('T2' in file_name.decode('utf-8'))):
                shutil.copy(filepath, os.path.join(a2Path, file_name.decode('utf-8')))
            if(('-b' in file_name.decode('utf-8')) and ('T2' in file_name.decode('utf-8'))):
                shutil.copy(filepath, os.path.join(b2path, file_name.decode('utf-8')))
            if(('-c' in file_name.decode('utf-8')) and ('T2' in file_name.decode('utf-8'))):
                shutil.copy(filepath, os.path.join(c2path, file_name.decode('utf-8')))

