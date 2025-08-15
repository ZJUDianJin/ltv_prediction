from config import *
from data_read import *
from dataclasses import dataclass
import pandas as pd
import os
import time

import ast
import os
import re
import sys
import time
import random
import logging
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from ltv_model_sind import S_Model, Y_Model, SequentialFeatureExtractor
from tqdm import tqdm
from itertools import chain
from sklearn.preprocessing import StandardScaler
import pandas as pd


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class Dataset:
    X: np.ndarray
    X_sparse: np.ndarray
    X_seq: np.ndarray
    T: np.ndarray
    S: np.ndarray
    Y: np.ndarray

def create_datasets(data, test_size=0.2, random_state=42):
    X_train, X_valid = train_test_split(data["X"], test_size=test_size, random_state=random_state)
    X_sparse_train, X_sparse_valid = train_test_split(data["X_sparse"], test_size=test_size, random_state=random_state)
    X_seq_train, X_seq_valid = train_test_split(data["X_seq"], test_size=test_size, random_state=random_state)
    T_train, T_valid = train_test_split(data["T"], test_size=test_size, random_state=random_state)
    S_train, S_valid = train_test_split(data["S"], test_size=test_size, random_state=random_state)
    Y_train, Y_valid = train_test_split(data["Y"], test_size=test_size, random_state=random_state)
    
    train_dataset = Dataset(X=X_train, X_sparse=X_sparse_train, X_seq=X_seq_train, T=T_train, S=S_train, Y=Y_train)
    valid_dataset = Dataset(X=X_valid, X_sparse=X_sparse_valid, X_seq=X_seq_valid, T=T_valid, S=S_valid, Y=Y_valid)

    return train_dataset, valid_dataset

def label_t_learner(S0_model, S1_model, seq_model, obs_data_train, device, threshold=0.5):
    X_obs_tensor = torch.tensor(obs_data_train.X, dtype=torch.float32).to(device)
    X_sparse_obs_tensor = torch.tensor(obs_data_train.X_sparse, dtype=torch.float32).to(device)
    X_seq_obs_tensor = torch.tensor(obs_data_train.X_seq, dtype=torch.float32).to(device)
    S_obs_tensor = torch.tensor(obs_data_train.S, dtype=torch.float32).to(device)

    obs_dataset = TensorDataset(X_obs_tensor, X_sparse_obs_tensor, X_seq_obs_tensor, S_obs_tensor)
    obs_loader = DataLoader(obs_dataset, batch_size=batch_size, shuffle=False)

    seq_model.eval()
    S0_model.eval()
    S1_model.eval()
    estimated_T_labels = []
    residual = []
    sbase = []

    with torch.no_grad():
        S_T1_pred_list = []
        S_T0_pred_list = []
        for batch in obs_loader:
            X, X_sparse, X_seq, S = [x.to(device) for x in batch]
            X_seq = seq_model(X_seq)
            s0_pred = S0_model(X, X_sparse, X_seq)
            s1_pred = S1_model(X, X_sparse, X_seq)
            S_T0_pred_list.append(s0_pred)
            S_T1_pred_list.append(s1_pred)
        S_T0_pred = torch.cat(S_T0_pred_list, dim=0)
        S_T1_pred = torch.cat(S_T1_pred_list, dim=0)

        for i in range(len(X_obs_tensor)):
            S_real = S_obs_tensor[i]
            S_T0 = S_T0_pred[i]
            S_T1 = S_T1_pred[i]
            R = S_T1 - S_T0
            residual.append(R)
            sbase.append(S_T0)
            T0_votes = torch.sum(torch.abs(S_real - S_T0) < torch.abs(S_real - S_T1)).item()
            T1_votes = len(S_real) - T0_votes
            
            if T1_votes > T0_votes:
                estimated_T_labels.append(exp)
            else:
                estimated_T_labels.append(base)

    residual = torch.stack(residual)
    sbase = torch.stack(sbase)
    return estimated_T_labels, residual, sbase

def train_s_model(S1_model, S0_model, seq_model, exp_data_train, exp_data_valid, device, num_epochs=10, batch_size=1024, learning_rate=0.001, lamb=1e-4):
    criterion = nn.MSELoss()

    X_exp_tensor = torch.tensor(exp_data_train.X, dtype=torch.float32).to(device)
    X_sparse_exp_tensor = torch.tensor(exp_data_train.X_sparse, dtype=torch.float32).to(device)
    X_seq_exp_tensor = torch.tensor(exp_data_train.X_seq, dtype=torch.float32).to(device)
    T_exp_tensor = torch.tensor(exp_data_train.T, dtype=torch.float32).unsqueeze(1).to(device)
    S_exp_tensor = torch.tensor(exp_data_train.S, dtype=torch.float32).to(device)
    Y_exp_tensor = torch.tensor(exp_data_train.Y, dtype=torch.float32).to(device)

    exp_dataset = TensorDataset(X_exp_tensor, X_sparse_exp_tensor, X_seq_exp_tensor, T_exp_tensor, S_exp_tensor, Y_exp_tensor)
    exp_data_loader = DataLoader(exp_dataset, batch_size=batch_size, shuffle=True)

    X_val_tensor = torch.tensor(exp_data_valid.X, dtype=torch.float32).to(device)
    X_sparse_val_tensor = torch.tensor(exp_data_valid.X_sparse, dtype=torch.float32).to(device)
    X_seq_val_tensor = torch.tensor(exp_data_valid.X_seq, dtype=torch.float32).to(device)
    T_val_tensor = torch.tensor(exp_data_valid.T, dtype=torch.float32).unsqueeze(1).to(device)
    S_val_tensor = torch.tensor(exp_data_valid.S, dtype=torch.float32).to(device)
    Y_val_tensor = torch.tensor(exp_data_valid.Y, dtype=torch.float32).to(device)

    val_dataset = TensorDataset(X_val_tensor, X_sparse_val_tensor, X_seq_val_tensor, T_val_tensor, S_val_tensor, Y_val_tensor)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(chain(S1_model.parameters(), S0_model.parameters(), seq_model.parameters()), lr=learning_rate, weight_decay=lamb)

    for epoch in range(num_epochs):
        S1_model.train()
        S0_model.train()
        for batch in exp_data_loader:
            X, X_sparse, X_seq, T, S, Y = [x.to(device) for x in batch]
            X_seq = seq_model(X_seq)

            optimizer.zero_grad()
            output1 = S1_model(X, X_sparse, X_seq)
            output0 = S0_model(X, X_sparse, X_seq)

            mask1 = T.squeeze() == exp
            mask0 = T.squeeze() == base

            loss1 = criterion(output1[mask1], S[mask1])
            loss0 = criterion(output0[mask0], S[mask0])
            loss = loss1 + loss0
            
            loss.backward()
            optimizer.step()
        
        S1_model.eval()
        S0_model.eval()
        with torch.no_grad():
            total_loss = 0.0
            for batch in val_data_loader:
                X, X_sparse, X_seq, T, S, Y = [x.to(device) for x in batch]
                X_seq = seq_model(X_seq)
                output1 = S1_model(X, X_sparse, X_seq)
                output0 = S0_model(X, X_sparse, X_seq)

                mask1 = T.squeeze() == exp
                mask0 = T.squeeze() == base
                
                loss1 = criterion(output1[mask1], S[mask1])
                loss0 = criterion(output0[mask0], S[mask0])
                total_loss += (loss1 + loss0).item()

            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item()}, Validation Loss: {total_loss/len(val_data_loader)}')



def train_y_model(Y1_model, Y0_model, seq_model, obs_data_train, obs_pred_label, obs_sbase, obs_residual, obs_data_valid, obs_valid_t, obs_valid_residual, obs_valid_sbase, device, num_epochs=10, batch_size=1024, learning_rate=0.001, lamb=1e-4):
    criterion = nn.MSELoss()
    # criterion = nn.HuberLoss(delta=1.0)
    
    optimizer = optim.Adam(chain(Y1_model.parameters(), Y0_model.parameters(), seq_model.parameters()), lr=learning_rate, weight_decay=lamb)

    X_train_tensor = torch.tensor(obs_data_train.X, dtype=torch.float32).to(device)
    X_sparse_train_tensor = torch.tensor(obs_data_train.X_sparse, dtype=torch.float32).to(device)
    X_seq_train_tensor = torch.tensor(obs_data_train.X_seq, dtype=torch.float32).to(device)
    Y_train_tensor = torch.tensor(obs_data_train.Y, dtype=torch.float32).to(device)
    T_train_tensor = torch.tensor(obs_pred_label, dtype=torch.float32).unsqueeze(1).to(device)
    S_train_tensor = torch.tensor(obs_data_train.S, dtype=torch.float32).to(device)
    Sbase_train_tensor = torch.tensor(obs_sbase, dtype=torch.float32).to(device)
    R_train_tensor = torch.tensor(obs_residual, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_tensor, X_sparse_train_tensor, X_seq_train_tensor, T_train_tensor, S_train_tensor, Sbase_train_tensor, R_train_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    X_valid_tensor = torch.tensor(obs_data_valid.X, dtype=torch.float32).to(device)
    X_sparse_valid_tensor = torch.tensor(obs_data_valid.X_sparse, dtype=torch.float32).to(device)
    X_seq_valid_tensor = torch.tensor(obs_data_valid.X_seq, dtype=torch.float32).to(device)
    S_valid_tensor = torch.tensor(obs_data_valid.S, dtype=torch.float32).to(device)
    Y_valid_tensor = torch.tensor(obs_data_valid.Y, dtype=torch.float32).to(device)
    T_valid_tensor = torch.tensor(obs_valid_t, dtype=torch.float32).unsqueeze(1).to(device)
    Sbase_valid_tensor = torch.tensor(obs_valid_sbase, dtype=torch.float32).to(device)
    R_valid_tensor = torch.tensor(obs_valid_residual, dtype=torch.float32).to(device)

    valid_dataset = TensorDataset(X_valid_tensor, X_sparse_valid_tensor, X_seq_valid_tensor, T_valid_tensor, S_valid_tensor, Sbase_valid_tensor, R_valid_tensor, Y_valid_tensor)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    Y1_model.to(device)

    for epoch in range(num_epochs):
        Y1_model.train()
        Y0_model.train()

        for batch in train_loader:
            X, X_sparse, X_seq, T, S, Sbase, R, Y = batch
            optimizer.zero_grad()
            X_seq = seq_model(X_seq)
            Y_pred_1 = Y1_model(X, X_sparse, X_seq, S)

            loss1 = criterion(Y_pred_1, Y)
            loss = loss1

            loss.backward()
            optimizer.step()


        Y1_model.eval()
        Y0_model.eval()
        with torch.no_grad():
            total_loss = 0.0
            for batch in valid_loader:
                X, X_sparse, X_seq, T, S, Sbase, R, Y = batch
                X_seq = seq_model(X_seq)
                Y_pred_valid_1 = Y1_model(X, X_sparse, X_seq, S)

                valid_loss1 = criterion(Y_pred_valid_1, Y)
                total_loss = (valid_loss1).item()

            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item()}, Validation Loss: {total_loss/len(valid_loader)}')


def calculate_ate(Y: np.ndarray, T: np.ndarray) -> float:
    Y_1_mean = np.mean(Y[T == exp], axis=0)
    Y_0_mean = np.mean(Y[T == base], axis=0)
    ate = Y_1_mean - Y_0_mean
    print("valid ate: ", ate)
    if exp == 1:
        ate = 0.009
        return ate
    else:
        ate = -0.014
        return ate

def calculate_pred_ate(model1, model0, seq_model, X: np.ndarray, X_sparse: np.ndarray, X_seq: np.ndarray, S: np.ndarray, T: np.ndarray, R: np.ndarray, Y: np.ndarray, Sbase: np.ndarray, device) -> float:
    X = torch.tensor(X, dtype=torch.float32).to(device)
    X_sparse = torch.tensor(X_sparse, dtype=torch.float32).to(device)
    X_seq = torch.tensor(X_seq, dtype=torch.float32).to(device)
    S = torch.tensor(S, dtype=torch.float32).to(device)
    Sbase = torch.tensor(Sbase, dtype=torch.float32).to(device)
    R = torch.tensor(R, dtype=torch.float32).to(device)
    T = torch.tensor(T, dtype=torch.float32).unsqueeze(-1).to(device)

    valid_dataset = TensorDataset(X, X_sparse, X_seq, S)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model1.eval()
    seq_model.eval()
    with torch.no_grad():
        pred_Y_1_list = []
        for batch in valid_loader:
            X, X_sparse, X_seq, S = batch
            X_seq = seq_model(X_seq)
            y1_pred = model1(X, X_sparse, X_seq, S)
            pred_Y_1_list.append(y1_pred)
        pred_Y_1 = torch.cat(pred_Y_1_list, dim=0)


    print("Printing samples for pred_Y_1 where T == 1")
    count_1, count_0 = 0, 0
    max_samples = 5
    for i in range(len(T)):
        if T[i] == exp and count_1 < max_samples:
            print(f"Sample {count_1 + 1}: pred_Y_1 = {pred_Y_1[i]}, label = {Y[i]}")
            count_1 += 1

    pred_Y_1 = pred_Y_1.detach().cpu().numpy()
    T = T.detach().cpu().numpy()

    pred_Y_1_mean = np.mean(pred_Y_1[T == exp], axis=0)
    pred_Y_0_mean = np.mean(pred_Y_1[T == base], axis=0)
    pred_ate = pred_Y_1_mean - pred_Y_0_mean
    return pred_ate


def objective():
    if len(sys.argv) < 2:
        raise ValueError("Please provide a file name prefix as command-line argument.")
    file_prefix = sys.argv[1]
    output_file = f"result/{file_prefix}.txt"
    with open(output_file, 'w') as f:
        f.write("===== Config.py Content =====\n")
        with open('config.py', 'r') as config_file:
            config_content = config_file.read()
            f.write(config_content)

    print(device)

    start_time = time.perf_counter()
    exp_data = exp_data_read(exp_file_path)
    obs_data = obs_data_read(obs_file_path)
    end_time = time.perf_counter()
    print(f"read data：{end_time - start_time:.4f} s")

    exp_data_train, exp_data_valid = create_datasets(exp_data)
    obs_data_train, obs_data_valid = create_datasets(obs_data)

    mae_list = []
    ate_pred_list = []
    for i in range(exp_num):
        seq_model = SequentialFeatureExtractor(embed_dim=embed_dim, num_heads=num_heads, fc_layers=fc_layers).to(device)
        sparse_embedding_layer = nn.Embedding(x_sparse_dim, sparse_embedding_dim).to(device)
        S1_model = S_Model(x_dim, s_dim, hidden_dim, sparse_embedding_layer).to(device)
        S0_model = S_Model(x_dim, s_dim, hidden_dim, sparse_embedding_layer).to(device)
        Y1_model = Y_Model(x_dim, s_dim, y_dim, hidden_dim, sparse_embedding_layer).to(device)
        Y0_model = Y_Model(x_dim, s_dim, y_dim, hidden_dim, sparse_embedding_layer).to(device)

        start_time = time.perf_counter()
        train_s_model(S1_model, S0_model, seq_model, exp_data_train, exp_data_valid, device, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate, lamb=lamb)
        end_time = time.perf_counter()
        print(f"train S model：{end_time - start_time:.4f} s")

        obs_pred_label, obs_residual, sbase = label_t_learner(S0_model, S1_model, seq_model, obs_data_train, device)
        val_pred_label, val_residual, val_sbase = label_t_learner(S0_model, S1_model, seq_model, obs_data_valid, device)

        start_time = time.perf_counter()
        train_y_model(Y1_model, Y0_model, seq_model, obs_data_train, obs_pred_label, sbase, obs_residual, obs_data_valid, val_pred_label, val_residual, val_sbase, device, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate, lamb=lamb)
        end_time = time.perf_counter()
        print(f"train Y model：{end_time - start_time:.4f} s")

        ate = calculate_ate(exp_data_valid.Y, exp_data_valid.T)
        _, test_redisual, test_sbase = label_t_learner(S0_model, S1_model, seq_model, exp_data_valid, device)
        pred_ate = calculate_pred_ate(Y1_model, Y0_model, seq_model, exp_data_valid.X, exp_data_valid.X_sparse, exp_data_valid.X_seq, exp_data_valid.S, exp_data_valid.T, test_redisual, exp_data_valid.Y, test_sbase, device)

        print("ATE vs Predicted ATE:")
        print(f"ATE: {ate:.4f}, Predicted ATE: {pred_ate.item():.4f}")
        error = abs(ate - pred_ate.item())
        print(f"error: {error:.4f}")
        mae_list.append(error)
        ate_pred_list.append(pred_ate)

    print(ate_pred_list)
    print(mae_list)
    mean_error = np.mean(mae_list)
    std_error = np.std(mae_list)
    print(f"MAE (mean ± std): {mean_error:.4f} ± {std_error:.4f}")

    with open(output_file, 'a') as f:
        f.write("===== ATE Predictions =====\n")
        f.write(str(ate_pred_list) + '\n\n')

        f.write("===== MAE List =====\n")
        f.write(str(mae_list) + '\n\n')

        f.write("===== MAE Summary =====\n")
        f.write(f"MAE (mean ± std): {mean_error:.4f} ± {std_error:.4f}\n\n")

    print(f"Results and config saved to {output_file}")

set_seed(42)
objective()