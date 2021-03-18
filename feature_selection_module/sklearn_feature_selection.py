### Selection of features to use using SKlearn ### 

#!/usr/bin/env python3
import numpy as np
import pandas as pd 
import scipy
import scipy.io
import sklearn 
import os
from sklearn.feature_selection import VarianceThreshold
dataPath = '/mnt/c/Users/Michael/Desktop/PyRadiomics/GeneratedFeatures/belowslices/T1/normalized_data'
corr_file ='/mnt/c/Users/Michael/Desktop/PyRadiomics/GeneratedFeatures/feature_correlation_below.csv' 

#ds = []
#d = {}
#for file in os.listdir(dataPath):
#   file_name = os.fsencode(file)
#   mat = scipy.io.loadmat(os.path.join(dataPath, file_name.decode('utf-8')))
#   mat.pop('__header__')
#   mat.pop('__version__')
#   mat.pop('__globals__')
#   ds.append(mat) 
#for k in ds[0]:
#    d[k] = np.concatenate(list(d[k] for d in ds))
d = scipy.io.loadmat(dataPath)
d.pop('__header__')
d.pop('__version__')
d.pop('__globals__')
for key in d.keys():
    d[key] = list(d[key][0])
    print(d[key])
df = pd.DataFrame(d)
print(df)
print(df.values)
for col in df.columns:
    df[col] = pd.DataFrame([x for x in df[col]])
df.astype(float)
collected_features = []
#for row in df:
#    collected_features.append(df[row].tolist())
#old = df[row].tolist()
selector = VarianceThreshold(threshold=0.10)
var_values = selector.fit_transform(df.values)
variance_df = pd.DataFrame(var_values)
for row in variance_df:
    print(variance_df[row].tolist())