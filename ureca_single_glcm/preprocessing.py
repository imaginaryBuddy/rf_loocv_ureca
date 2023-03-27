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

DIRECTIONS : list = [0, np.pi/2, np.pi/4, 3*np.pi/4]
STEP_SIZES: list = [1]
LEVEL: int = 256
# mean_homogeneity means taking the avg in all directions mentioned
FEATURES: list = ['mean_homogeneity', 'mean_contrast', 'mean_dissimilarity', 'mean_ASM', 'mean_correlation', 'mean_energy']

class Band(enum.Enum):
    red = 0
    green = 1
    blue = 2
    RE = 3
    NIR = 4
    W_RE = 5
    W_NIR = 6


class CrownGLCMGenerator():
    def __init__(self, img_file, bounds_df, RGB_or_not: bool):
        self.bounds_df = bounds_df
        # for rgb only
        self.img_file = img_file
        self.RGB = RGB_or_not
        self.img, self.bands = self.get_img(img_file)
        self.bands_species_glcm: DefaultDict[List] =  {}
        self.pd_data = pd.DataFrame()

    def generate_glcm_mean(self, glcm):
        glcm_mean = []
        for d in range(len(DIRECTIONS)):
            m = 0
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
            glcm = graycomatrix(self.bands[index][int(y0):int(y1) + 1, int(x0):int(x1) + 1], distances=STEP_SIZES, \
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
                [tree_name, np.mean(self.generate_glcm_mean(glcm)),
                 np.mean(graycoprops(glcm,'homogeneity')), np.mean(graycoprops(glcm, 'contrast')), np.mean(graycoprops(glcm, 'dissimilarity')),
                 np.mean(graycoprops(glcm, 'ASM')), np.mean(graycoprops(glcm, 'correlation')),
                 np.mean(graycoprops(glcm, 'energy'))])

            # self.bands_species_glcm[band.name][tree_name] = glcm if tree_name not in self.bands_species_glcm[band.name].keys() \
            #     else np.append(self.bands_species_glcm[band.name][tree_name], glcm) # if axis is None, out is a flattened array for np.append
    # def show_img(self):
    #     plt.imshow(self.img)

    def get_img(self, file):
        if self.RGB: # if it's a file with RGB
            im = gdal.Open(file)

            b1 = im.GetRasterBand(1).ReadAsArray() # r
            b2 = im.GetRasterBand(2).ReadAsArray() # g
            b3 = im.GetRasterBand(3).ReadAsArray() # b
            b4 = im.GetRasterBand(4).ReadAsArray() #alpha band

            img = np.dstack((b1, b2, b3, b4))
            # print(np.min(b1.flatten()), np.max(b1.flatten()))
            return img, [b1, b2, b3, b4]
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
    # dec_features_r = pd.read_csv('data/redo/dec_features_r.csv')
    # dec_features_g = pd.read_csv('data/redo/dec_features_g.csv')
    # dec_features_b = pd.read_csv('data/redo/dec_features_b.csv')
    # dec_features_RE = pd.read_csv('data/redo/dec_features_RE.csv')
    # dec_features_NIR = pd.read_csv('data/redo/dec_features_NIR.csv')
    #
    #
    # may_features_r = pd.read_csv('data/redo/may_features_r.csv')
    # may_features_g = pd.read_csv('data/redo/may_features_g.csv')
    # may_features_b = pd.read_csv('data/redo/may_features_b.csv')
    # may_features_RE = pd.read_csv('data/redo/may_features_RE.csv')
    # may_features_NIR = pd.read_csv('data/redo/may_features_NIR.csv')
    #
    # dec_dfs = [dec_features_r, dec_features_g, dec_features_b, dec_features_RE, dec_features_NIR]
    # may_dfs = [may_features_r, may_features_g, may_features_b, may_features_RE, may_features_NIR]
    # dec_merged_df = reduce(lambda left, right: pd.merge(left, right, on=['index', 'tree_name'],
    #                                                 how='outer'), dec_dfs)
    # may_merged_df = reduce(lambda left, right: pd.merge(left, right, on=['index', 'tree_name'],
    #                                                     how='outer'), may_dfs)
    # dec_merged_df.drop(columns=["index"])
    # may_merged_df.drop(columns=["index"])
    # # dec_merged_df = merge("dec",
    # #                       [dec_features_r, dec_features_g, dec_features_b, dec_features_RE, dec_features_NIR])
    # # may_merged_df = merge("may",
    # #                       [may_features_r, may_features_g, may_features_b, may_features_RE, may_features_NIR])
    #
    # # make sure that when you concatenate, the column names are the same
    # sets = ["glcm_mean", *FEATURES]
    # col_names = ["index", "tree_name", *["r_"+x for x in sets],
    #              *["g_"+x for x in sets], *["b_"+x for x in sets], *["RE_"+x for x in sets], *["NIR_"+x for x in sets]]
    #
    # dec_merged_df.rename(columns=lambda x: col_names[dec_merged_df.columns.get_loc(x)], inplace=True)
    # may_merged_df.rename(columns=lambda x: col_names[may_merged_df.columns.get_loc(x)], inplace=True)
    # all_merged_df = pd.concat([dec_merged_df, may_merged_df], axis=0, ignore_index=True)
    #
    # # encode the species values and put in column = 'id_based_on_tree_name'
    # dec_merged_df_e = encode_tree_species(dec_merged_df)
    # may_merged_df_e = encode_tree_species(may_merged_df)
    # all_merged_df_e = encode_tree_species(all_merged_df)
    #
    #
    # #
    # all_merged_df_e.to_csv("data/redo/all_merged_data_e.csv")
    # dec_merged_df_e.to_csv("data/redo/dec_merged_data_e.csv")
    # may_merged_df_e.to_csv("data/redo/may_merged_data_e.csv")
    #
    dec_merged_df_e = pd.read_csv("data/redo/dec_merged_data_e.csv").drop(columns=["id_based_on_tree_name"])
    may_merged_df_e = pd.read_csv("data/redo/may_merged_data_e.csv").drop(columns=["id_based_on_tree_name"])
    all_merged_df_e = pd.read_csv("data/redo/all_merged_data_e.csv").drop(columns=["id_based_on_tree_name"])
    windowed_glcm_dec = pd.read_csv("data/windowed_glcm/glcm_data_dec_all_7rad_15step_128bins.csv").sort_index().drop(columns=["id_based_on_tree_name"])
    windowed_glcm_may = pd.read_csv("data/windowed_glcm/glcm_data_may_all_7rad_15step_128bins.csv").sort_index().drop(columns=["id_based_on_tree_name"])
    merged_data_dec_windowed_glcm = pd.merge(dec_merged_df_e, windowed_glcm_dec, on=['index_', 'tree_name']).set_index('index_')
    merged_data_may_windowed_glcm = pd.merge(may_merged_df_e, windowed_glcm_may, on=['index_', 'tree_name']).set_index('index_')

    merged_data_dec_windowed_glcm.to_csv("data/glcm_merged_windowed/dec_merged_glcm_windowed.csv")
    merged_data_may_windowed_glcm.to_csv("data/glcm_merged_windowed/may_merged_glcm_windowed.csv")
    print(merged_data_may_windowed_glcm.columns)
    large_trees_dec = merged_data_dec_windowed_glcm[merged_data_dec_windowed_glcm["tree_name"].isin(LARGE_TREES)]
    small_trees_dec = merged_data_dec_windowed_glcm[merged_data_may_windowed_glcm["tree_name"].isin(SMALL_TREES)]
    
    large_trees_dec.to_csv("data/large_trees/large_trees_dec_merged.csv")
    small_trees_dec.to_csv("data/large_trees/small_trees_dec_merged.csv")

    large_trees_may = merged_data_may_windowed_glcm[merged_data_may_windowed_glcm["tree_name"].isin(LARGE_TREES)]
    small_trees_may = merged_data_may_windowed_glcm[merged_data_may_windowed_glcm["tree_name"].isin(SMALL_TREES)]

    large_trees_may.to_csv("data/large_trees/large_trees_may_merged.csv")
    small_trees_may.to_csv("data/large_trees/small_trees_may_merged.csv")

if __name__ == "__main__":
    preprocessing()



