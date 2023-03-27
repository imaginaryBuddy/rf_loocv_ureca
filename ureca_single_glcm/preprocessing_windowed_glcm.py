'''
tree ndarray structure
(119, 74, 8,7)
(Height, Width, Channel, Features)

channels:

Wideband Red = 0
Wideband Green = 1
Wideband Blue = 2
RedEdge = 3
Blue = 4
NIR = 5
Red = 6
Green = 7

features:

NONE = 0
HOMOGENEITY = 1
CONTRAST = 2
ASM = 3
MEAN_I = 4
MEAN_J = 5
VAR_I = 6
VAR_J = 7
CORRELATION = 8


'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from FRModelYSH.data import dec_all_7rad_15step_128bins, may_all_3rad_3step_32bins
# from FRModelYSH import augment_tree, dataloader, generate_features, glcm_loader
# from FRModelYSH.glcm_loader import load_glcm
from FRModelYSH.generate_features import *
from preprocessing import encode_tree_species
from abc import ABC
import os
import pickle
import random
from enum import Enum

class band_names(Enum):
    Wideband_Red = 0
    Wideband_Green = 1
    Wideband_Blue = 2
    RedEdge = 3
    Blue = 4
    NIR = 5
    Red = 6
    Green = 7
class feature_names(Enum):
    NONE = 0
    HOMOGENEITY = 1
    CONTRAST = 2
    ASM = 3
    MEAN_I = 4
    MEAN_J = 5
    VAR_I = 6
    VAR_J = 7
    CORRELATION = 8


features = {0:[4],
            1:[4],
            2:[0,4],
            3:[0,1,4],
            4:[1,3,4],
            5:[0,1,4],
            6:[1,4],
            7:[0,1,4]}

bands = [0,1,2,3,4,5,6,7]

DIRECTORY = 'FRModelYSH/data/may_all_7rad_15step_128bins'

class GLCMGenerator(ABC):
    def __init__(self, directory=DIRECTORY):
        self.directory = directory
        # self.block_out = np.zeros((max_x, max_y), dtype=int)
        self.data = []
        self.df = None
    def save_data(self):
        col_names = ["index", "tree_name"]
        for i,f in enumerate(features):
            for num in features.get(i):
                col_names.append(f"{band_names(i).name}_{feature_names(num).name}_mean")
                col_names.append(f"{band_names(i).name}_{feature_names(num).name}_median")
                col_names.append(f"{band_names(i).name}_{feature_names(num).name}_25th")
                col_names.append(f"{band_names(i).name}_{feature_names(num).name}_75th")
        self.df = pd.DataFrame(self.data, columns=col_names)
        self.df.to_csv("data/windowed_glcm/glcm_data_may_all_7rad_15step_128bins.csv")
    def get_glcm_data(self):
        for filename in os.listdir(self.directory):
            tree_data = []
            f = os.path.join(self.directory, filename)
            # checking if it is a file
            print(filename)
            tree_name = filename.split("_")[0]
            index = int(filename.split("_")[1].split(".")[0])
            if os.path.isfile(f):
                tree_data.append(index)
                tree_data.append(tree_name)
                tree = self.load_glcm(f)
                tree_data.extend(generate_feature_from_tree_all(tree))
                self.data.append(tree_data)
            else:
                print("error")

    def load_glcm(self, data_dir):
        with open(data_dir, "rb") as f:
            tree = pickle.load(f)
        return tree

    def already_sampled(self, x, y, bb_size, block_out):
        if block_out[x, y] == 1:
            return True
        else:
            return False

    def sample_a_tree(self, tree, bb_size, block_out):
        nTries = 0
        max_x = (tree.shape[0] - bb_size)
        max_y = (tree.shape[1] - bb_size)
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        # nTries is to try for another pair of coordinates for n times if there is overlap.
        while (self.already_sampled and nTries < 3):
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            nTries += 1

        sample = (tree[x:x + bb_size, y:y + bb_size, :, :])
        block_out[x:x + bb_size, y:y + bb_size] = np.ones((bb_size, bb_size), dtype=int)

        return sample


if __name__ == "__main__":
    # data = pd.DataFrame()
    # for filename in os.listdir(DIRECTORY):
    #     f = os.path.join(DIRECTORY, filename)
    #     # checking if it is a file
    #     if os.path.isfile(f):
    #         print(f)
    #     else:
    #         print("error")
    # glcm_gen = GLCMGenerator()
    # glcm_gen.get_glcm_data()
    # glcm_gen.save_data()
    data = pd.read_csv("data/windowed_glcm/glcm_data_may_all_7rad_15step_128bins.csv", index_col='index')
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    data_e = encode_tree_species(data)
    data_e.to_csv("data/windowed_glcm/glcm_data_may_all_7rad_15step_128bins.csv")

