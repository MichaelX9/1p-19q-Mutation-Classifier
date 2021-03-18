### sort out deletive mutation images from non-deletion images ###
s
#!/usr/bin/env python3
import numpy as np
import pandas as pd
import scipy
import scipy.io
import sklearn 
import os
dataPath = '/mnt/c/Users/Michael/Desktop/PyRadiomics/GeneratedFeatures/centerslices/T2'
deletion_folder ='/mnt/c/Users/Michael/Desktop/PyRadiomics/GeneratedFeatures/centerslices/T2/deletion'
deletion_nums = ['LGG-241', 'LGG-313', 'LGG-334', 'LGG-500', 'LGG-506', 'LGG-518', 'LGG-537', 'LGG-219', 'LGG-273', 'LGG-277', 'LGG-280', 'LGG-285', 'LGG-297', 'LGG-327', 'LGG-338', 'LGG-343', 'LGG-351', 'LGG-354', 'LGG-363', 'LGG-374', 'LGG-375', 'LGG-391', 'LGG-516', 'LGG-203', 'LGG-286', 'LGG-533', 'LGG-545', 'LGG-558', 'LGG-574', 'LGG-585', 'LGG-589', 'LGG-601', 'LGG-622', 'LGG-625', 'LGG-306', 'LGG-631']

for img in os.listdir(dataPath):
    image_file = os.fsencode(img)
    ImageID = (image_file.decode('utf-8')).split('.')[0]
    for i in range(len(deletion_nums)):
        if (deletion_nums[i] in ImageID):
            print(ImageID)
            os.rename(os.path.join(dataPath, img), os.path.join(deletion_folder, (ImageID + '.mat')))