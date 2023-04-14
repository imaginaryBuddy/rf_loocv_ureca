from osgeo import gdal
from skimage.feature import graycomatrix, graycoprops
import numpy as np
import rasterio as rs
import matplotlib.pyplot as plt
# Falcataria Moluccana|2187|3018|1841|2541
# DIRECTIONS : list = [0, np.pi/2, np.pi/4, 3*np.pi/4]
# STEP_SIZES: list = [1]
LEVEL: int = 128
# dec_r = gdal.Open("source/dec/result_Red.tif")
# # Campnosperma Auriculatum|671|1006|3777|4026
# b_red = dec_r.GetRasterBand(1)
# b_red_arr = b_red.ReadAsArray()
#
#
# def generate_glcm_mean(self, glcm, name):
#     glcm_mean = []
#
#     for d in range(len(DIRECTIONS)):
#         m = 0
#         g = glcm[:, :, 0, d]
#         shape = g.shape
#         np.savetxt(f"{name}_{shape}_dec_glcm.csv", g, delimiter=",")
#         for i in range(LEVEL):
#             for j in range(i, LEVEL):  # symmetrical
#                 m += i * glcm[i, j, 0, d]
#         glcm_mean.append(m)
#     return glcm_mean
# print("running")
# # glcm = graycomatrix(b_re[y0:y1+1, x0:x1+1], distances=STEP_,)
#
# plt.imshow(rs.open("source/raw_18Dec2020_result_RedEdge.tif").read()[0])
# plt.show()

'''
To see how the downscaling affects the distribution of pixels 
'''
may_dir = "source/may/"
may_R = gdal.Open(may_dir + "raw_10May2021_90deg43m85pct255deg_result_Red.tif")
band = may_R.GetRasterBand(1)
band = band.ReadAsArray()
# Falcataria Moluccana|2310|3100|2323|2918
# Ficus Variegata|2565|3063|1089|1612
# tree_name, y0, y1, x0, x1 = tree.str.split("|")[0]

y0 = 2310
y1 = 3100
x0 = 2323
x1 = 2918

y0_ = 2565
y1_ = 3063
x0_ = 1089
x1_ = 1612

falcataria = band[int(y0):int(y1) + 1, int(x0):int(x1) + 1].flatten()
ficus = band[int(y0_):int(y1_) + 1, int(x0_):int(x1_) + 1].flatten()
plt.hist(falcataria, alpha=0.5, label='falcataria', color="blue", bins=100)
plt.hist(ficus, alpha=0.5, label='ficus', color="red", bins=100)
plt.title("Before hardcode downscale")
plt.show()

plt.hist(falcataria/(2**7),alpha=0.5, label='falcataria', color="blue", bins=100)
plt.hist(ficus/(2**7), alpha=0.5, label='ficus', color="red", bins=100)
plt.title("After hardcode downscale")
plt.show()


def binning(band_arr, bin_to):
    num_bits = int(np.ceil(np.log2((np.nanmax(band_arr)))))
    print(num_bits)
    band_arr = np.rint(band_arr)
    bins = np.linspace(np.nanmin(band_arr), 2 ** num_bits, LEVEL - 1)
    quantized = np.digitize(band_arr, bins)
    return quantized

# binned = binning(falcataria, 128)
# plt.hist(binned, color="blue")
# plt.title("binned falcataria")
# plt.show()

binned_whole_img = binning(band, 128)
binned_falcataria = binned_whole_img[int(y0):int(y1) + 1, int(x0):int(x1) + 1].flatten()
binned_ficus = binned_whole_img[int(y0_):int(y1_) + 1, int(x0_):int(x1_) + 1].flatten()
plt.hist(binned_ficus, alpha=0.5, label='binned_ficus', color='blue', bins=100)
plt.hist(binned_falcataria, alpha=0.5, label='binned_falcataria', color='red', bins=100)
plt.title("binned")
plt.show()