# %%
import pandas as pd
import os
import os.path as osp

from DataLoader.BasicLoader import *
from Model.BasicCF import *

# 数据加载
data = pd.read_table("../data/ml-100k.data", header=None)
data_loader = BasicLoader()
data_loader.load_df(data)
# 训练集验证集分割
ur_train, ur_test, ir_train, ir_test = data_loader.train_test_split()
# 模型加载
model = UserCF()
model.fit(ur_train)
res = model.pred(ur_test, data_loader.rating_scale, k=20, similarity='iif')
# print(ur_test)
for uid, items in ur_test.items():
    true_info = [iid for iid, rating in items if rating > (data_loader.rating_scale[0]+data_loader.rating_scale[1])/2]
    # print(true_info)
    # print(res[uid])
    true_set = set(true_info)
    pred_set = set(res[uid])
    print("user_id: {:d} jaccard_similarity: {:.5f}".format(
        uid,
        len(true_set.intersection(pred_set)) / len(true_set.union(pred_set))
    ))
    # print(len(true_set.union(pred_set)))
# %%
