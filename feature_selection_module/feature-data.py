### generature csv for feature statistics ###

#!/usr/bin/env python3
import numpy as np
import pandas as pd 
import scipy
import scipy.io
import sklearn 
import os
dataPath = '/mnt/c/Users/Michael/Desktop/PyRadiomics/GeneratedFeatures/centerslices/T1'
corr_file ='/mnt/c/Users/Michael/Desktop/PyRadiomics/GeneratedFeatures/feature_correlation_below.csv' 


def max_min_finder(data):
    max = data[0]
    min = data[0]
    for i in range(len(data)):
        if data[i] > max:
            max = data[i]
        if data[i] < min:
            min = data[i]
    return [max, min]

dict_created = False
data_dict = {}
value_dict = {}
for file in os.listdir(dataPath):
    file_name = os.fsencode(file)
    print(file_name.decode('utf-8'))
    if('testing_fold' in file_name.decode('utf-8')):
        break
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
for k in data_dict.keys():
    wanted = []
    max_min = max_min_finder(data_dict[k])
    wanted.append(max_min[0])
    wanted.append(max_min[1])
    wanted.append(np.average(data_dict[k]))
    wanted.append(np.std(data_dict[k]))
    value_dict[k] = wanted
new_file = os.path.join(dataPath, 'stat_values.csv')
df = pd.DataFrame.from_dict(value_dict)
df.to_csv(new_file)
