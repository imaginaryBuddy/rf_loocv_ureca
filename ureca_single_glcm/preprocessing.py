import numpy as np
import pandas as pd
from osgeo import gdal
from typing import List, DefaultDict
import rasterio
import math
import enum
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import LabelEncoder
from random_forest import RandomForestManager, RandomForestOnly
from functools import reduce

SMALL_TREES = ["Calophyllum", "Dillenia Suffruticosa", "Shorea Leprosula", "Sterculia Parviflora"]
LARGE_TREES = ["Falcataria Moluccana", "Ficus Variegata", "Spathodea Campanulatum", "Campnosperma Auriculatum"]


may_dir = "source/may/"
dec_dir = "source/dec/"
mean = {}
SAVE_AS: str = "glcm_for_each_crown"

# DIRECTIONS : list = [0, np.pi/2, np.pi/4, 3*np.pi/4]
DIRECTIONS : list = [0]
STEP_SIZES: list = [1]
LEVEL: int = 128
# mean_homogeneity means taking the avg in all directions mentioned
FEATURES: list = ['mean_homogeneity', 'mean_contrast', 'mean_dissimilarity', 'mean_ASM', 'mean_correlation', 'mean_energy']

class Band(enum.Enum):
    red = 0
    green = 1
    blue = 2
    RE = 3
    NIR = 4
    WR = 5
    WG = 6
    WB = 7


class CrownGLCMGenerator():
    def __init__(self, img_file, bounds_df, RGB_or_not: bool, bin_to: int=None, wideband: bool=True):
        self.bounds_df = bounds_df
        self.wide_band = wideband
        # for rgb only
        self.img_file = img_file
        self.RGB = RGB_or_not
        self.img, self.bands = self.get_img(img_file)
        self.bands_species_glcm: DefaultDict[List] =  {}
        self.pd_data = pd.DataFrame()
        self.bin_to = bin_to

    def generate_glcm_mean(self, glcm, name):
        glcm_mean = []

        for d in range(len(DIRECTIONS)):
            m = 0
            g = glcm[:,:,0,d]
            shape = g.shape
            np.savetxt(f"{name}_{shape}_dec_glcm.csv",g , delimiter=",")
            for i in range(LEVEL):
                for j in range(i, LEVEL): #symmetrical
                    m += i*glcm[i, j, 0, d]
            glcm_mean.append(m)
        return glcm_mean

    def generate_glcm_with_features(self, band: Band):
        if self.RGB: # if rgb : i.e, result.tif
            index = band.value
        else:
            index = 0

        for i, tree in self.bounds_df.iterrows():
            tree_name, y0, y1, x0, x1 = tree.str.split("|")[0]
            # if self.bin_to:
            #     b_arr = self.binning(self.bands[index][int(y0):int(y1) + 1, int(x0):int(x1) + 1], LEVEL)
            b_arr = self.hardcode_downscale(self.bands[index][int(y0):int(y1) + 1, int(x0):int(x1) + 1])
            print(np.nanmax(b_arr))
            b_arr_uint = b_arr.astype(np.uint)
            # else:
            #     b_arr = self.bands[index][int(y0):int(y1) + 1, int(x0):int(x1) + 1]
            glcm = graycomatrix(b_arr_uint, distances=STEP_SIZES, \
                                angles=DIRECTIONS, levels=LEVEL, symmetric=True, normed=True)
            # g = np.zeros((LEVEL, LEVEL))
            # for angle_index in range(len(DIRECTIONS)):
            #     g += glcm[:, :, 0, angle_index]

            if band.name not in self.bands_species_glcm.keys():
                self.bands_species_glcm[band.name] = []
            '''
            tree_name, raw_mean, glcm_mean, homogeneity, contrast 
            '''
            self.bands_species_glcm[band.name].append(
                [tree_name, np.mean(self.generate_glcm_mean(glcm, band.name)),
                 np.mean(graycoprops(glcm,'homogeneity')), np.mean(graycoprops(glcm, 'contrast')), np.mean(graycoprops(glcm, 'dissimilarity')),
                 np.mean(graycoprops(glcm, 'ASM')), np.mean(graycoprops(glcm, 'correlation')),
                 np.mean(graycoprops(glcm, 'energy'))])

            # self.bands_species_glcm[band.name][tree_name] = glcm if tree_name not in self.bands_species_glcm[band.name].keys() \
            #     else np.append(self.bands_species_glcm[band.name][tree_name], glcm) # if axis is None, out is a flattened array for np.append
    # def show_img(self):
    #     plt.imshow(self.img)
    def generate_hist(self, band, name):
        plt.hist(band, 100, range=[0, np.nanmax(band)])
        plt.title = name
        plt.show()

    def get_img(self, file):
        if self.RGB: # if it's a file with RGB
            im = gdal.Open(file)

            b1 = im.GetRasterBand(1).ReadAsArray() # r
            b2 = im.GetRasterBand(2).ReadAsArray() # g
            b3 = im.GetRasterBand(3).ReadAsArray() # b
            b4 = im.GetRasterBand(4).ReadAsArray() # alpha band

            img = np.dstack((b1, b2, b3, b4))
            # print(np.min(b1.flatten()), np.max(b1.flatten()))
            return img, [b1, b2, b3, b4]
        elif self.wide_band == True:
            im = gdal.Open(file)
            b1 = im.GetRasterBand(1).ReadAsArray()
            img = np.dstack(b1)
            return img, [b1]

        else:
            im = gdal.Open(file)
            b1 = im.GetRasterBand(1)
            b1_arr = b1.ReadAsArray()
            plt.imshow(b1_arr)
            print(self.img_file)
            plt.savefig(f"{str(self.img_file.split('.')[0])}_test_.png")
            print(np.nanmax(b1_arr))
            num_bits = int(np.ceil(np.log2((np.nanmax(b1_arr)))))
            print(num_bits)
            b1_arr = np.rint(b1_arr)
            bins = np.linspace(np.nanmin(b1_arr), 2**num_bits, LEVEL-1)
            print(bins)
            quantized = np.digitize(b1_arr, bins)
            plt.imshow(quantized)
            plt.savefig(f"{str(self.img_file.split('.')[0])}_quantized.png")
            # b2 = im.GetRasterBand(2).ReadAsArray()
            img = np.dstack(b1_arr)
            return img, [quantized]

    def binning(self, band_arr, bin_to):
        num_bits = int(np.ceil(np.log2((np.nanmax(band_arr)))))
        print(num_bits)
        band_arr = np.rint(band_arr)
        bins = np.linspace(np.nanmin(band_arr), 2 ** num_bits, LEVEL - 1)
        quantized = np.digitize(band_arr, bins)
        return quantized

    def hardcode_downscale(self, band_arr):
        band_arr = band_arr//(2**7)
        band_arr.astype(np.uint)
        return band_arr
def merge(name, list_of_dfs: List[pd.DataFrame]):
    merge_df = pd.DataFrame()
    for i in range(len(list_of_dfs)):
        if i == 0:
            merge_df = list_of_dfs[i]
        else:
            to_merge = list_of_dfs[i].copy().drop('tree_name', axis=1)
            merge_df = merge_df.join(to_merge, how="outer")
    merge_df.to_csv(f"{name}_merged_data.csv")
    return merge_df

def encode_tree_species(data):
    '''
    uses LabelEncoder from sklearn.preprocessing to encode the species. Note that duplicated species names have the same ID
    :param data:
    :return:
    '''
    species = data["tree_name"]
    le = LabelEncoder()
    le.fit(species)
    encoded_species = le.transform(species)

    data_new = data.copy()
    data_new["id_based_on_tree_name"] = encoded_species
    print(data_new)
    return data_new

def preprocessing():
    dec = pd.read_csv("data/glcm_merged_windowed_redo/dec_merged_glcm_windowed.csv")
    may = pd.read_csv("data/glcm_merged_windowed_redo/may_merged_glcm_windowed.csv")
    all = pd.concat([dec, may], axis=0)
    all.reset_index(drop=True, inplace=True)
    all.drop("index", axis=1, inplace=True)
    all.index.name= "index"
    all.to_csv("data/glcm_merged_windowed_redo/all_merged_glcm_windowed.csv")
if __name__ == "__main__":
    preprocessing()
