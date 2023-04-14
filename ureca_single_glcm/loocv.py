import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random_forest import RandomForestLOOCV
from datetime import datetime
from configs import *
import time
all_glcm_merged_window = pd.read_csv("data/glcm_merged_windowed_redo/all_merged_glcm_windowed.csv")
species_15_data = all_glcm_merged_window.loc[all_glcm_merged_window['tree_name'].isin(species_15)]
species_15_data.reset_index(inplace=True)
print(species_15_data.shape)
def execute_loocv_with_data(data, top_features_list, top_num_features, name=""):
    print("-"*10)
    print(f"top_num_features = ", top_num_features)
    top = top_features_list.sort_values("balanced_accuracy")
    top = top["feature"].head(top_num_features).tolist()
    columns_to_use = ["tree_name"]
    columns_to_use.extend(top)
    data_to_use = data[columns_to_use]

    # carry out LOOCV
    rf_loocv = RandomForestLOOCV(data_to_use)
    res = rf_loocv.execute_LOOCV()
    res.to_csv(f"loocv_res/{name}_results_{top_num_features}.csv")
    time.sleep(1)
    # evaluate the true positive

    res['test'] = res['test'].astype(str)
    res['pred'] = res['pred'].astype(str)
    tp_count = res.groupby(['test', 'pred']).size()
    tp_count = pd.DataFrame(tp_count)
    tp_count.reset_index(inplace=True)
    tp_count.rename(columns={0: "tp_count"}, inplace=True)
    tp_count.to_csv(f"loocv_res/pred_test_count/{name}_pred_test_count_{top_num_features}.csv")
    tp_count = tp_count[tp_count["pred"] == tp_count["test"]]
    tp_count.to_csv(f"loocv_res/tp_count/{name}_tp_count_{top_num_features}.csv")
    print("-" * 10)
if __name__ == "__main__":
    top_flist = pd.read_csv("accuracy_and_features.csv")
    top_flist = top_flist.sort_values("balanced_accuracy")
    for num in range(10, 40):
        execute_loocv_with_data(species_15_data, top_flist, num, name="species_15")

# df = pd.read_csv("accuracy_and_features.csv")
# top = df.sort_values("balanced_accuracy")
# top = top["feature"].head(40).tolist()
# num = len(top)
# # print(num) there are 105 features in total
# columns_to_use = ["tree_name"]
# columns_to_use.extend(top)
# data_to_use = all_glcm_merged_window[columns_to_use]
#
# rf_loocv = RandomForestLOOCV(data_to_use)
#
# res = rf_loocv.execute_LOOCV()
#
# res.to_csv(f"loocv_res/{datetime.now().minute}_res.csv")
#
# # evaluate the true positive
# tp_count = res.groupby(["test", "pred"]).size()
# tp_count = pd.DataFrame(tp_count)
# tp_count.reset_index(inplace=True)
# tp_count.rename(columns={0: "tp_count"}, inplace=True)
# tp_count.to_csv(f"loocv_res/pred_test_count_{num}.csv")
# tp_count = tp_count[tp_count["pred"] == tp_count["test"]]
# tp_count.to_csv(f"loocv_res/tp_count_{num}.csv")
#
#
# # #%%
# # tree_instances_count = all_glcm_merged_window.groupby("tree_name").size()
# # tree_instances_count = pd.DataFrame(tree_instances_count)
# # tree_instances_count.reset_index(inplace=True)
# # tree_instances_count.rename(columns={0: "tree_count"}, inplace=True)
# # tree_instances_count.to_csv("loocv_res/tree_instances_count.csv")
# # #%%
#
#
# tree_instances_count = pd.read_csv("loocv_res/tree_instances_count.csv")
# tp_count["test"] = tp_count["test"].str.replace("'", "").str.replace("]","").str.replace("[","")
# tp_count.rename(columns={"test": "tree_name"}, inplace=True)
# tp_count.drop(inplace=True, columns=["pred"])
#
#
# tp_count = pd.merge(tp_count, tree_instances_count, on="tree_name")
# tp_count.to_csv(f"loocv_res/tp_count_{num}_refined.csv")
