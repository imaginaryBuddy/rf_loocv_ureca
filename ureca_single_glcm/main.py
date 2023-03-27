import numpy as np
import pandas as pd
from osgeo import gdal
from typing import List, DefaultDict
import math
import enum
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import LabelEncoder
from random_forest import RandomForestManager, RandomForestOnly
from functools import reduce
from preprocessing import encode_tree_species
SMALL_TREES = ["Calophyllum", "Dillenia Suffruticosa", "Shorea Leprosula", "Sterculia Parviflora"]
LARGE_TREES = ["Falcataria Moluccana", "Ficus Variegata", "Spathodea Campanulatum", "Campnosperma Auriculatum"]

if __name__ == "__main__":
    # import warnings
    #
    # warnings.filterwarnings("ignore",
    #                         message="The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.")
    # dec_merged_df_e = pd.read_csv("data/dec_merged_data_e.csv", index_col=[0])
    # may_merged_df_e = pd.read_csv("data/may_merged_data_e.csv", index_col=[0])
    # dec_merged_df_e.index.name = "index"
    # may_merged_df_e.index.name = "index"
    # # all_merged_df_e = pd.read_csv("data/all_merged_data_e.csv", index_col=[0])

    # # execute algo
    # dec_algo_mng = RandomForestManager(dec_merged_df_e)
    # may_algo_mng = RandomForestManager(may_merged_df_e)
    # all_algo_mng = RandomForestManager(all_merged_df_e)
    #
    # print("dec")
    # _acc_dec, dec_importances = dec_algo_mng.LOOCV_RF()
    # dec_importances.to_csv("dec_importances.csv")
    # print("acc: ", _acc_dec)
    # print("-" * 40)
    #
    # print("dec")
    # _acc_may, may_importances = may_algo_mng.LOOCV_RF()
    # may_importances.to_csv("may_importances.csv")
    # print("acc: ", _acc_may)
    # print("-" * 40)
    #
    # print("dec")
    # _acc_all, all_importances = all_algo_mng.LOOCV_RF()
    # all_importances.to_csv("all_importances.csv")
    # print("acc: ", _acc_all)
    # print("-" * 40)


    # columns_to_use = ['tree_name', 'r_raw_mean', 'g_raw_mean', 'b_raw_mean',
    #                   'RE_glcm_mean', 'NIR_glcm_mean', 'r_mean_dissimilarity']
    # rf = RandomForestOnly(train_data=dec_merged_df_e[columns_to_use], test_data=may_merged_df_e[columns_to_use], name="trained_dec_tested_may")
    # report = rf.execute()
    # print(report)
    # may_windowed = pd.read_csv("data/windowed_glcm/glcm_data_may_all_7rad_15step_128bins.csv",index_col="index")
    # dec_windowed = pd.read_csv("data/windowed_glcm/glcm_data_dec_all_7rad_15step_128bins.csv",index_col="index")
    #
    # rf = RandomForestManager(may_windowed)
    # _acc_may_windowed, may_windowed_importances = rf.LOOCV_RF()
    # print("accuracy: ", _acc_may_windowed)
    #
    # may_windowed_importances.to_csv("importances/may_windowed_importances.csv")
    # may_windowed.sort_values(inplace=True, by="index", ascending=True)
    # dec_windowed.sort_values(inplace=True, by="index", ascending=True)
    #
    # total_data_may = pd.merge(may_merged_df_e, may_windowed, on=["index", "tree_name"])
    # total_data_dec = pd.merge(dec_merged_df_e, dec_windowed, on=["index", "tree_name"])

    #
    # columns_to_use = pd.read_csv("importances/windowed_glcm_only_feature_importance.csv").iloc[:,0].head(11).values.tolist()
    # columns_to_use.insert(0, 'tree_name')
    # columns_to_use.extend(columns_to_use_norm)
    # dec = encode_tree_species(total_data_dec[columns_to_use])
    # may = encode_tree_species(total_data_may[columns_to_use])
    # dec.to_csv("total_data_dec_extracted.csv")
    # may.to_csv("total_data_may_extracted.csv")
    # print(columns_to_use)
    # total_data_dec = pd.read_csv("total_data_dec_extracted.csv").drop(columns=["b_glcm_mean", "g_glcm_mean"])
    #
    # total_data_may = pd.read_csv("total_data_may_extracted.csv").drop(columns=["b_glcm_mean", "g_glcm_mean"])

    # large_dec = total_data_dec.loc[total_data_dec['tree_name'].isin(LARGE_TREES)]
    # large_may = total_data_may.loc[total_data_may['tree_name'].isin(LARGE_TREES)]

    '''
    LOOCV_RF
    '''
    # print("dec acc", acc_dec)
    # print(imp_dec)
    # rf_may = RandomForestManager(total_data_may)
    # _acc_may_windowed, may_windowed_importances = rf_may.LOOCV_RF()
    # print("may accuracy: ", _acc_may_windowed)
    #

    # rf_may = RandomForestManager(total_data_may)
    #
    # acc_may, imp_may = rf_may.LOOCV_RF()
    # print(imp_may)
    # # rf_only.get_feature_importances()
    # print("may acc", acc_may)


    '''
    Random Forest only 
    '''
    # print(large_dec)
    # rf_large = RandomForestOnly(large_dec, large_may, "large_trees")
    # rf_large.execute()
    # columns_to_use_norm = ['index', 'tree_name', 'b_mean_dissimilarity', 'b_glcm_mean',
    #                        'NIR_mean_contrast', 'g_glcm_mean', 'r_mean_contrast',
    #                        'RE_mean_contrast', 'RE_mean_homogeneity',
    #                        'NIR_mean_correlation', 'NIR_mean_ASM', '']
    data_dec = pd.read_csv("data/glcm_merged_windowed/dec_merged_glcm_windowed.csv")
    data_may = pd.read_csv("data/glcm_merged_windowed/may_merged_glcm_windowed.csv")
    importance = pd.read_csv("importances/ALL FROM MERGED DATA (GLCM FEATURES + WINDOWED GLCM)_feature_importance.csv")
    importance.rename(columns={importance.columns[0]:"feature"}, inplace=True)
    # num = 12
    # top = pd.Series(importance.head(num)["feature"]).tolist()
    #

    # cols = data_dec.columns
    # columns_to_use_norm = list(cols[:2])
    # # columns = ['RedEdge_HOMOGENEITY_75th', 'r_mean_contrast', 'Green_HOMOGENEITY_median', 'b_mean_correlation',
    # #            'r_mean_dissimilarity', 'b_mean_homogeneity', 'g_mean_dissimilarity', 'r_glcm_mean', 'g_glcm_mean', 'b_glcm_mean', 'RE_glcm_mean',
    # #            'NIR_glcm_mean', '']
    # columns_to_use_norm.extend(top)
    # # results = []
    # # for i in range(len(all)):
    # #     results.append()
    # #     pass

    # results = {}
    # print(len(importance["feature"]))
    # columns_to_use_norm = ['index_', 'tree_name']
    # for index, row in importance.iterrows():
    #     columns_to_use_norm.append(row["feature"])
    #     if index == 0:
    #         results[row["feature"]] = 0.0
    #         continue
    #     print(columns_to_use_norm)
    #     data_dec_amended = pd.DataFrame(data_dec, columns=columns_to_use_norm)
    #     print(data_dec_amended)
    #     data_may_amended = pd.DataFrame(data_may, columns=columns_to_use_norm)
    #     rf = RandomForestOnly(data_dec_amended, data_may_amended,
    #                           f"Selected_top{index+1}_1 FROM MERGED DATA (GLCM FEATURES + WINDOWED GLCM)")
    #     acc, bal_accuracy = rf.execute()
    #     results[row["feature"]] = bal_accuracy
    # convert to df
    # df = pd.DataFrame.from_dict(results, orient='index')
    df = pd.read_csv("accuracy_and_features.csv")

    # # create a bar chart
    # fig, ax = plt.subplots()
    # ax.bar(df.index, df['balanced_accuracy'])
    #
    # plt.xticks(np.arange(0, 10*df.shape[0]+1, 10))
    # ax.set_xticklabels(df['feature'])
    #
    # # add labels and title
    # ax.set_xlabel('feature')
    # ax.set_ylabel('balanced accuracy')
    # ax.set_title('balanced accuracy with adding a feature each time in order of importance')
    #
    # # show the plot
    # plt.show()
    #
    # N = df.shape[0]
    # data = df['balanced_accuracy']
    #
    # plt.plot(data)
    # plt.xticks(range(N))  # add loads of ticks
    # plt.grid()
    # plt.title('balanced accuracy with adding a feature each time in order of importance')
    # # plt.set_xticklabels(df['feature'])
    # plt.gca().margins(x=0)
    # plt.gcf().canvas.draw()
    # tl = plt.gca().get_xticklabels()
    # maxsize = max([t.get_window_extent().width for t in tl])
    # m = 0.2  # inch margin
    # s = maxsize / plt.gcf().dpi * N + 2 * m
    # margin = m / plt.gcf().get_size_inches()[0]
    #
    # plt.gcf().subplots_adjust(left=margin, right=1. - margin)
    # plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
    # #
    # # plt.savefig(__file__ + ".png")
    # plt.show()

    # plt.rcParams["figure.figsize"] = [10.0, 3.5]
    # plt.rcParams["figure.dpi"] = 400
    # plt.rcParams["figure.autolayout"] = True
    #
    # x = [i for i in range(df.shape[0])]
    # ax1 = plt.subplot()
    # ax1.set_xticks(x)
    # # ax1.set_yticks(x)
    # ax1.set_xticklabels(df["feature"].tolist(), rotation=90, fontsize = 4)
    # # ax1.set_yticklabels(["one", "two", "three", "four"], rotation=45)
    # # ax1.tick_params(axis="both", direction="in", pad=15)
    # plt.bar(df["feature"].tolist(), df["balanced_accuracy"].tolist())
    # plt.show()

    top = df.sort_values("balanced_accuracy")
    top = top.head(16)["feature"].tolist()
    columns_to_use_norm = ['index_', 'tree_name']
    columns_to_use_norm.extend(top)
    data_dec_amended = pd.DataFrame(data_dec, columns=columns_to_use_norm)
    data_may_amended = pd.DataFrame(data_may, columns=columns_to_use_norm)
    rf = RandomForestOnly(data_dec_amended, data_may_amended,
                          f"Selected_top from importance{len(top)}_1(GLCM FEATURES + WINDOWED GLCM)")
    acc, bal_accuracy = rf.execute()
    # print("acc: ", acc)
    # print("bal_acc: ", bal_accuracy)
