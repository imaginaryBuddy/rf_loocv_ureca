# -*- coding: utf-8 -*-

'''
@author: chris, shen hwei
'''

import pandas as pd 
import numpy as np 
from FRModelYSH.generate_features import generate_feature_from_tree_mean
from FRModelYSH.glcm_loader import load_glcm
from FRModelYSH.augment_tree import sample_a_tree

def dataloader(data_folder, tree_files, sample_count, sample_crop_size):
    glcm_features = pd.DataFrame()
    df_sizes = pd.DataFrame()
    for treefile in tree_files:
        tree = load_glcm(data_folder + treefile)
        x = treefile.split('_')
        name = x[-2] # get name of trees from file name
        block_out = np.zeros(tree.shape[0:2], dtype=int)
        for i in range(sample_count):
            tree_sample = sample_a_tree(tree, sample_crop_size, block_out)
            sample_features = generate_feature_from_tree_mean(tree_sample)
            sample_features.insert(0, name)
            sample_features_series = pd.Series(sample_features)
            glcm_features = glcm_features.append(sample_features_series, ignore_index=True)
    return glcm_features