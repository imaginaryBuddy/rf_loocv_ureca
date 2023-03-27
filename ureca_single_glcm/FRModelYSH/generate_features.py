# -*- coding: utf-8 -*-

'''
from a ndarray structure e.g. (119, 74, 8,7) or (20,20, 8,7), generate feature of the tree.
for e.g. a feature of the tree can be any combination of MEAN, VAR, SKEW, KURTOSIS of
ANY CHANNELS of ANY FEATURES
Example:
np.mean[:,:, i, j] where i=7, j=8 means MEAN_GREEN_CORRELATION of the tree

@author: chris
'''

#bands:
'''
Wideband Red = 0
Wideband Green = 1
Wideband Blue = 2
RedEdge = 3
Blue = 4
NIR = 5
Red = 6
Green = 7
'''

#features:
'''
NONE = 0
HOMOGENEITY = 1
CONTRAST = 2
ASM = 3
MEAN = 4
VAR = 5
CORRELATION = 6

'''

import numpy as np

#CONFIGS 
features = {0:[4],
            1:[4],
            2:[0,4],
            3:[0,1,4],
            4:[1,3,4],
            5:[0,1,4],
            6:[1,4],
            7:[0,1,4]}
bands = [0,1,2,3,4,5,6,7]

'''
@author: shen hwei

'''
def cherry_pick(configs, name):
    '''
    Band | Features
    [0]WIR: Mean [4]
    [1]WIG: Mean [4]
    [2]WIB: Mean, None [0, 4] 
    [3]RE: None, Homogeineity, Mean [0,1,4] 
    [4]Blue: Homogeneity, ASM, Mean [1,3,4]
    [5]NIR: None, Homogeneity, Mean [0,1,4]
    [6]Red: Homogeineity, Mean [1,4]
    [7]Green: None, Homogeineity, Mean [0,1,4]
    '''
    pass 


'''
@author: chris, shen hwei
'''
def generate_feature_from_tree_mean(tree):
    # print("tree shape: ", tree.shape)
    tree_features = []
    # for each band
    for i in bands:
        # for each band
        for j in features.get(i):
            tree_features.append(np.mean(tree[:,:, i, j].flatten())) # is flatten the right way to do it?
    # tree_features_series = pd.Series(tree_features) # make it so easier to put in table later
    return tree_features



'''
@author: shen hwei
'''
def generate_feature_from_tree_median(tree):
    tree_features_median = []
    for i in bands: 
        for j in features.get(i):
            tree_features_median.append(np.median(tree[:,:, i, j].flatten()))
    return tree_features_median


'''
@author: shen hwei
'''
def generate_feature_from_tree_percentile(tree, percentile):
    tree_features_percentile = []
    for i in bands:
        for j in features.get(i):
            tree_features_percentile.append(np.percentile(tree[:,:,i,j].flatten(), percentile))
    return tree_features_percentile

'''
@author: shen hwei
'''
def generate_feature_from_tree_all(tree):
    tree_features_mean = generate_feature_from_tree_mean(tree)
    tree_features_median = generate_feature_from_tree_median(tree)
    tree_features_25th = generate_feature_from_tree_percentile(tree, 25)
    tree_features_75th = generate_feature_from_tree_percentile(tree, 75)
    
    return [*tree_features_mean, *tree_features_median, *tree_features_25th, *tree_features_75th]




def generate_for_all_percentiles(tree):
    upper = generate_feature_from_tree_percentile(tree, 75)
    median = generate_feature_from_tree_percentile(tree, 50)
    lower = generate_feature_from_tree_percentile(tree, 25)
    
    return [*lower, *median, *upper]