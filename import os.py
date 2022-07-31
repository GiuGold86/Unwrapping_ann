import os
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
 
# Specify import
inVV = "D:\INGV_GROTTAMINARDA\progetto nuovo algoritmo sar\data\S1A_IW_SLC__1SDV_20200316T050403_20200316T050430_031696_03A798_A557.SAFE\measurement\s1a-iw1-slc-vv-20200316t050403-20200316t050428-031696-03a798-004.tiff"
inVH = "D:\INGV_GROTTAMINARDA\progetto nuovo algoritmo sar\data\S1A_IW_SLC__1SDV_20200316T050403_20200316T050430_031696_03A798_A557.SAFE\measurement\s1a-iw1-slc-vh-20200316t050403-20200316t050428-031696-03a798-001.tiff"
 
 
# get VV
dsVV = gdal.Open(inVV)
band1 = dsVV.GetRasterBand(1).ReadAsArray()
band1Real = dsVV.GetRasterBand(1).ReadAsArray().real.astype(np.float16)
band1Complex = dsVV.GetRasterBand(1).ReadAsArray().imag

print(np.max(band1Complex))
print(np.min(band1Complex))

fig = plt.figure()
plt.imshow(band1Real, cmap = 'gray', interpolation = None)

plt.show()
#bandVV = dsVV.GetRasterBand(1)
#arrVV = bandVV.ReadAsArray()
 
# get VH
#dsVH = gdal.Open(inVH)
#bandVH = dsVH.GetRasterBand(1)
#arrVH = bandVH.ReadAsArray()

#print (arrVH.size)
 
