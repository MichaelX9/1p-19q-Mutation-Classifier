#!/usr/bin/env python3
import numpy as np
import pandas as pd 
import scipy
import scipy.io
import sklearn 
import os
dataPath = '/mnt/c/Users/Michael/Desktop/PyRadiomics/TestSetFeatures/aboveslices/T1'

ds = []
d = {}
#for folder in os.listdir(dataPath):
#  image_folder = os.fsencode(os.path.join(dataPath, folder))
#  print(image_folder)
for file in os.listdir(dataPath):
   if (('correlations' in file) or ('selected' in file) or ('slices' in file) or ('all' in file)):
       continue
   file_name = os.fsencode(file)
   mat = scipy.io.loadmat(os.path.join(dataPath, file_name.decode('utf-8')))
   mat.pop('__header__')
   mat.pop('__version__')
   mat.pop('__globals__')
   ds.append(mat)
#mdata = mat['dataStruct']
#mtype = mdata.dtype
#ndata = {n: mdata[n][0,0] for n in mtype.names}
#print(ndata)
#data_raw = ndata['data']
#data_headline = ndata['channelIndices']
for k in ds[0]:
    d[k] = np.concatenate(list(d[k] for d in ds))
for key in d:
    d[key] = list(d[key])
df = pd.DataFrame(d)
for col in df.columns:
    df[col] = pd.DataFrame([x for x in df[col]])
df.astype(float)
print(df)
print(df.head())
print(df.shape)
print(df.columns)
high_corr = []
corr = df.corr()
df.to_csv(os.path.join(dataPath, 'features.csv'))
corr.to_csv(os.path.join(dataPath, 'correlations.csv'))

columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range((i+1), corr.shape[0]):
        if corr.iloc[i,j] >= 0.7:
            if columns[j]:
                columns[j] = False
selected_columns = df.columns[columns]
df = df[selected_columns]
df.to_csv(os.path.join(dataPath, 'selected_.5.csv'))
#correlation_df = correlation_table.loc['original_shape_SurfaceArea']
#cur_ind = 0
#ind = correlation_df.index
#for row in correlation_df:
#    print(row)
#    if (row > 0.5):
#        high_corr.append((row, ind[cur_ind]))
#    cur_ind += 1
#print(high_corr)
##print(df[df.columns[]]




