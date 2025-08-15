import torch
import numpy as np
import pandas as pd
from config import *


exp_file_path = 'data/ltv_exp_data_final_sampling'
exp_new_file_path = 'data/exp_all.csv'

data = pd.read_csv(exp_file_path, sep='\t')
data.to_csv(exp_new_file_path, index=False)

# obs_file_path = 'data/ltv_obs_data_100_sample.txt'
# obs_new_file_path = 'data/ltv_obs_data_100_sample.csv'

# data = pd.read_csv(obs_file_path, nrows=10, sep='\t')
# data = pd.read_csv(obs_file_path, sep='\t')
# data.to_csv(obs_new_file_path, index=False)



# EXP 
# 0 userid
# 1-7 Y
# [8, 252] X -> [8, 159] 序列特征 + [160, 250] 静态特征
# 251 开始时间
# 252 窗口大小
# 253 T
# [254, 405] S


# OBS
# 0 userid
# 1-7 Y
# [8, 252] X -> [8, 159] 序列特征 152 + [160, 250] 静态特征 91
# 251 开始时间
# 252 窗口大小
# [253, 404] S 152


# model 1 X,T -> S 
# model 2 X,S,T -> Y


# 7.30 batch_nomolization + 跑通,不加attention

# 1. 补充一些推荐系统的基础知识
# 2. 把现在的特征处理搞对
# 3. 简化S和label -> label ltv
# 4. 把在线数据的baseline补起来 Sind
# 5. 初步的对比
# 6. 关于S_seq可以参考我们的代码，思考时间序列的特征该怎么处理




