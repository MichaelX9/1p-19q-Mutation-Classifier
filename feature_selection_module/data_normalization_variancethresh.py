#!/usr/bin/env python3
import numpy as np
import pandas as pd 
import scipy
import scipy.io
import sklearn 
import os
from sklearn.preprocessing import MinMaxScaler
dataPath = '/mnt/c/Users/Michael/Desktop/PyRadiomics/GeneratedFeatures/belowslices/T2/deletion'
corr_file ='/mnt/c/Users/Michael/Desktop/PyRadiomics/GeneratedFeatures/feature_correlation_below.csv' 

dict_created = False
data_dict = {}
value_dict = {}
i = 0
for file in os.listdir(dataPath):
    file_name = os.fsencode(file)
    print(file_name.decode('utf-8'))
    if(('stat_values' in file_name.decode('utf-8')) or ('correlations' in file_name.decode('utf-8'))):
        pass
    else:
        mat = scipy.io.loadmat(os.path.join(dataPath, file_name.decode('utf-8')))
        mat.pop('__header__')
        mat.pop('__version__')
        mat.pop('__globals__')
        if dict_created == False:
            data_dict = mat
            dict_created = True
        else:
            for k in data_dict.keys():
               data_dict[k] = np.append(data_dict[k], mat[k])
print(i)
print(data_dict)
scaler = MinMaxScaler()
for k in data_dict.keys():
    if((not('ZonePercentage' in k)) & (not('RunPercentage' in k)) & (not('Icm1' in k))):
        transformed_data = scaler.fit_transform(data_dict[k].reshape(-1,1)).reshape(1,-1)
        data_dict[k] = transformed_data[0][0:94]
print(data_dict)
scipy.io.savemat(os.path.join(dataPath, 'normalized_data'), data_dict, True)