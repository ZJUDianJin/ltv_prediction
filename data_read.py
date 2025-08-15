import torch
import pandas as pd
import numpy as np
import re
import ast
import sys
from config import *
from sklearn.preprocessing import StandardScaler

T_mapping = {
    '["base"]': 0,
    '["test_1"]': 1,
    '["test_2"]': 2
}

def extract_and_convert(column):
    def convert(value):
        if isinstance(value, str):
            match = re.search(r'-?\d+\.?\d*', value)
            if match:
                return float(match.group()) 
        return value 

    return column.apply(convert)

def convert(dataframe):
    if isinstance(dataframe, pd.Series):
        dataframe = dataframe.to_frame()
    for column in dataframe.columns:
        if dataframe[column].dtype == 'object':
            try:
                dataframe[column] = extract_and_convert(dataframe[column])
            except ValueError:
                print(f"警告: 列 '{column}' 包含非数值数据，无法转换")
                print(dataframe[column])
                
    return dataframe

def extract_and_convert_list(series):
    def convert(value):
        if isinstance(value, str):
            try:
                value_list = ast.literal_eval(value)
                if isinstance(value_list, list):
                    return [float(item) for item in value_list]
            except (SyntaxError, ValueError):
                print(f"无法转换 '{value}'，这可能导致数据丢失")
        elif isinstance(value, list):
            return [float(item) for item in value]
        
        return [float(value)]*180
    
    return series.apply(convert)

def list_convert(dataframe, target_length=180, default_value=0.0):
    """
    将 DataFrame 转换为 NumPy 数组 [n, dim, 180]。
    :param dataframe: 包含各维度特征的信息。
    :param target_length: 每个列表应当转换成的标准长度。
    :param default_value: 用于填补不足长度的默认值。
    :return: 三维 NumPy 数组格式的转换结果。
    """
    def process_value(value):
        if isinstance(value, str):
            try:
                parsed_list = ast.literal_eval(value)
                if isinstance(parsed_list, list):
                    return [float(item) for item in parsed_list] + [default_value] * (target_length - len(parsed_list))
            except (SyntaxError, ValueError):
                return [default_value] * target_length
        elif isinstance(value, list):
            return [float(item) for item in value] + [default_value] * (target_length - len(value))
        return [default_value] * target_length
    
    expanded_data = np.array([
        [process_value(value) for value in dataframe[col]]
        for col in dataframe.columns
    ])
    
    converted_data = expanded_data.swapaxes(0, 1)
    
    return converted_data


def exp_data_read(file_path):
    # read
    data = pd.read_csv(file_path)
    ID = data.iloc[:, 0]
    X = data.iloc[:, 160:217]
    X_sparse = data.iloc[:, 217:251]
    X_seq = data.iloc[:, 8:160]
    T = data.iloc[:, 253]
    S = data.loc[:, selected_short]
    Y = data.iloc[:, 1]

    # T mapping
    T = T.map(T_mapping)
    T = T.to_numpy(dtype=np.float32)

    # convert
    X = convert(X)
    X_sparse = convert(X_sparse).to_numpy()
    X_seq = list_convert(X_seq)
    S = list_convert(S, target_length=S_window)
    Y = convert(Y).to_numpy()
    S = S.reshape(S.shape[0], -1)

    # scaler
    scaler = StandardScaler()
    S = scaler.fit_transform(S)
    X = scaler.fit_transform(X)
    X_sparse = scaler.fit_transform(X_sparse)

    print("exp_read_over")

    return {"ID": ID, "X": X, "X_sparse": X_sparse, "X_seq": X_seq, "T": T, "S": S, "Y": Y}

def obs_data_read(file_path):
    # read
    data = pd.read_csv(file_path)
    ID = data.iloc[:, 0]
    X = data.iloc[:, 160:217]
    X_sparse = data.iloc[:, 217:251]
    X_seq = data.iloc[:, 8:160]
    S = data.loc[:, selected_short]
    Y = data.iloc[:, 1]

    # convert
    X = convert(X)
    X_sparse = convert(X_sparse).to_numpy()
    X_seq = list_convert(X_seq)
    S = list_convert(S, target_length=S_window)
    Y = convert(Y).to_numpy()
    S = S.reshape(S.shape[0], -1)

    # scaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_sparse = scaler.fit_transform(X_sparse)
    S = scaler.fit_transform(S)
    
    T = np.random.randint(0, 2, size=len(data))

    print("obs_read_over")

    return {"ID": ID, "X": X, "X_sparse": X_sparse, "X_seq": X_seq, "T": T, "S": S, "Y": Y}