# %%
# 21.11.17
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
# 21.11.24
import pandas as pd
import os
import os.path as osp
import numpy as np
import torch

from DataLoader.BasicLoader import *
from Model.BasicCF import *
from Model.LFM import *

# 数据加载
data = pd.read_table("../data/ml-100k.data", header=None)
data_loader = BasicLoader()
data_loader.load_df(data)
# 获取svd训练矩阵
mat = data_loader.train_mat()
# 试验用训练矩阵
# 21.12.04
v = np.array([[4,0,2,0,1],
            [0,2,3,0,0],
            [1,0,2,4,0],
            [5,0,0,3,1],
            [0,0,1,5,1],
            [0,3,2,4,1],], dtype=float)
# 模型加载
model = BasicLFM(5, lr=0.01)
model.fit(v, data_loader.rating_scale, 100)
print(model.pred([0,1,2],[0,1,2,3], True))
# %%
# 21.12.05
import pandas as pd
import os
import os.path as osp
import numpy as np
from scipy import sparse

from DataLoader.BasicLoader import *
from Model.Graph import *

# 数据加载
data = pd.read_table("../data/ml-100k.data", header=None)
data_loader = BasicLoader()
data_loader.load_df(data)
# 获取转移矩阵
mat, x, y = data_loader.train_trans_mat()
# %%
# 21.12.08
import numpy as np
from scipy import sparse
from Model.Graph import *
M=np.matrix([[0,        0,        0,        0.5,      0,        0.5,      0],
                [0,        0,        0,        0.25,     0.25,     0.25,     0.25],
                [0,        0,        0,        0,        0,        0.5,      0.5],
                [0.5,      0.5,      0,        0,        0,        0,        0],
                [0,        1.0,      0,        0,        0,        0,        0],
                [0.333,    0.333,    0.333,    0,        0,        0,        0],
                [0,        0.5,      0.5,      0,        0,        0,        0]])
model = PersonalRank(0.8)
model.fit(M, 4, 3)
r_u, r_i = model.pred([0, 1, 2], [0, 1, 2, 3])
# %%
