from osgeo import gdal
from skimage.feature import graycomatrix, graycoprops
import numpy as np
import rasterio as rs
import matplotlib.pyplot as plt
# Falcataria Moluccana|2187|3018|1841|2541
DIRECTIONS : list = [0, np.pi/2, np.pi/4, 3*np.pi/4]
STEP_SIZES: list = [1]
LEVEL: int = 256
re = gdal.Open("source/raw_18Dec2020_result_RedEdge.tif")
y0, y1, x0, x1 = 2187, 3018, 1841, 2541
b_re = re.GetRasterBand(1)
re_arr = b_re.ReadAsArray()
plt.imshow(re_arr)
plt.show()
print("running")
# glcm = graycomatrix(b_re[y0:y1+1, x0:x1+1], distances=STEP_,)

plt.imshow(rs.open("source/raw_18Dec2020_result_RedEdge.tif").read()[0])
plt.show()