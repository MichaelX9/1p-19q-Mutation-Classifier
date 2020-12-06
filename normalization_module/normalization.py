import pydicom
from pydicom.data import get_testdata_files
import os, sys, statistics, numpy
import normalization_ops 

##filename = os.path.join(os.path.dirname(sys.argv[0]), "testing.dcm")
##image1 = pydicom.dcmread(filename)
b = numpy.array([[1,1],[1,1],[1,1]])
a = normalization_operations.find_std(arr)
print(arr)
print(mean)
print(std)
##image1.PixelData = image1.pixel_array.tobytes()
##print(image1.pixel_array)
##print(image1.pixel_array.shape)
##print(image1.pixel_array[256])
##image1.save_as(filename)

