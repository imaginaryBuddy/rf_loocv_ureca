import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random_forest import RandomForestLOOCV
from datetime import datetime

all_glcm_merged_window = pd.read_csv("data/glcm_merged_windowed/all_glcm_merged_window.csv")


df = pd.read_csv("accuracy_and_features.csv")
top = df.sort_values("balanced_accuracy")
top = top["feature"].head(40).tolist()
num = len(top)
# print(num) there are 105 features in total
columns_to_use = ["tree_name"]
columns_to_use.extend(top)
data_to_use = all_glcm_merged_window[columns_to_use]

rf_loocv = RandomForestLOOCV(data_to_use)

res = rf_loocv.execute_LOOCV()

res.to_csv(f"loocv_res/{datetime.now().minute}_res.csv")

# evaluate the true positive
tp_count = res.groupby(["test", "pred"]).size()
tp_count = pd.DataFrame(tp_count)
tp_count.reset_index(inplace=True)
tp_count.rename(columns={0: "tp_count"}, inplace=True)
tp_count.to_csv(f"loocv_res/pred_test_count_{num}.csv")
tp_count = tp_count[tp_count["pred"] == tp_count["test"]]
tp_count.to_csv(f"loocv_res/tp_count_{num}.csv")


# #%%
# tree_instances_count = all_glcm_merged_window.groupby("tree_name").size()
# tree_instances_count = pd.DataFrame(tree_instances_count)
# tree_instances_count.reset_index(inplace=True)
# tree_instances_count.rename(columns={0: "tree_count"}, inplace=True)
# tree_instances_count.to_csv("loocv_res/tree_instances_count.csv")
# #%%


tree_instances_count = pd.read_csv("loocv_res/tree_instances_count.csv")
tp_count["test"] = tp_count["test"].str.replace("'", "").str.replace("]","").str.replace("[","")
tp_count.rename(columns={"test": "tree_name"}, inplace=True)
tp_count.drop(inplace=True, columns=["pred"])


tp_count = pd.merge(tp_count, tree_instances_count, on="tree_name")
tp_count.to_csv(f"loocv_res/tp_count_{num}_refined.csv")
