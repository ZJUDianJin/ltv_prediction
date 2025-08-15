from config import *

import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence

class S_Model(torch.nn.Module):
    def __init__(self, x_dim, s_dim, hidden_dim, sparse_embedding_layer):
        super(S_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.x_dim = x_dim
        self.s_dim = s_dim
        self.fc1 = torch.nn.Linear(x_dim + x_sparse_dim * sparse_embedding_dim + s_seq_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, s_dim)  
        self.sparse_embedding_layer = sparse_embedding_layer

        self.softmax = nn.Softmax(dim=-1)
        self.Q_w = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)
        self.K_w = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)
        self.V_w = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)
        self.x_rep = nn.Embedding(x_dim + x_sparse_dim * sparse_embedding_dim + s_seq_dim, hidden_dim)

    def self_attn(self, q, k, v):
        Q, K, V = self.Q_w(q), self.K_w(k), self.V_w(v)
        attn_weights = Q.matmul(torch.transpose(K, 1, 2)) / (K.shape[-1] ** 0.5)
        # attn_weights = self.softmax(torch.sigmoid(attn_weights))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        outputs = attn_weights.matmul(V)

        return outputs, attn_weights


    def forward(self, X, X_sparse, X_seq):
        X_sparse = X_sparse.unsqueeze(2) * self.sparse_embedding_layer.weight.unsqueeze(0)
        X_sparse = X_sparse.view(-1, sparse_embedding_dim * x_sparse_dim)
        X = torch.concat((X, X_sparse, X_seq), dim=1)
        # stage 1
        X_rep = X.unsqueeze(2) * self.x_rep.weight.unsqueeze(0)
        
        # sbase net
        dims = X_rep.size()
        _X_rep = X_rep / torch.linalg.norm(X_rep, dim=1, keepdim=True)
        xx, xx_weight = self.self_attn(_X_rep, _X_rep, _X_rep)

        X_attn_feature = X.unsqueeze(1)  
        X_attn_feature = torch.bmm(X_attn_feature, xx_weight)  
        X_attn_feature = X_attn_feature.squeeze(1)  

        x = F.relu(self.fc1(X_attn_feature))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Y_Model(torch.nn.Module):
    def __init__(self, x_dim, s_dim, y_dim, hidden_dim, sparse_embedding_layer):
        super(Y_Model, self).__init__()
        self.fc1 = torch.nn.Linear(x_dim + x_sparse_dim * sparse_embedding_dim + s_dim * 2 + s_seq_dim, hidden_dim) 
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, y_dim)  
        self.hidden_dim = hidden_dim
        self.x_dim = x_dim
        self.s_dim = s_dim
        self.y_dim = y_dim
        self.sparse_embedding_layer = sparse_embedding_layer

        self.softmax = nn.Softmax(dim=-1)
        self.Q_w = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)
        self.K_w = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)
        self.V_w = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)
        self.x_rep = nn.Embedding(x_dim + x_sparse_dim * sparse_embedding_dim + s_seq_dim, hidden_dim)

        self.att_embed_s_1 = nn.Linear(1, hidden_dim, bias=True)
        self.att_embed_x_2 = nn.Linear(x_dim + x_sparse_dim * sparse_embedding_dim + s_seq_dim, hidden_dim, bias=True)
        self.att_embed_sx_33 = nn.Linear(hidden_dim, 1, bias=True)

        self.att_embed_r_1 = nn.Linear(1, hidden_dim, bias=True)
        self.att_embed_xx_2 = nn.Linear(x_dim + x_sparse_dim * sparse_embedding_dim + s_seq_dim, hidden_dim, bias=True)
        self.att_embed_rx_33 = nn.Linear(hidden_dim, 1, bias=True)

    def self_attn(self, q, k, v):
        Q, K, V = self.Q_w(q), self.K_w(k), self.V_w(v)
        attn_weights = Q.matmul(torch.transpose(K, 1, 2)) / (K.shape[-1] ** 0.5)
        # attn_weights = self.softmax(torch.sigmoid(attn_weights))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        outputs = attn_weights.matmul(V)

        return outputs, attn_weights

    def interaction_attn_sx(self, s, x):
        # s: [N, D_s], x: [N, D_x]
        attention = []
        outputs = []
        for i in range(self.s_dim):  
            s_i = s[:, i].unsqueeze(1)                    
            h_s = torch.sigmoid(self.att_embed_s_1(s_i)) 
            h_x = torch.sigmoid(self.att_embed_x_2(x))   
            h = torch.relu(h_s + h_x)                     
            attn_score = self.att_embed_sx_33(h)
            attention.append(attn_score)
            outputs.append(s_i * attn_score)

        attention = torch.cat(attention, dim=1)
        attention = torch.softmax(attention, dim=1)

        s_out = s * attention
        return s_out, attention
    
    def interaction_attn_rx(self, r, x):
        # r: [N, D_r], x: [N, D_x]
        attention = []
        outputs = []
        for i in range(self.s_dim):
            r_i = r[:, i].unsqueeze(1)                     
            h_s = torch.sigmoid(self.att_embed_r_1(r_i))   
            h_x = torch.sigmoid(self.att_embed_xx_2(x))    
            h = torch.relu(h_s + h_x)                      
            attn_score = self.att_embed_rx_33(h)           
            attention.append(attn_score)
            outputs.append(r_i * attn_score)               

        attention = torch.cat(attention, dim=1)            
        attention = torch.softmax(attention, dim=1)        

        r_out = r * attention                          
        return r_out, attention

    def forward(self, X, X_sparse, X_seq, S, R):
        X_sparse = X_sparse.unsqueeze(2) * self.sparse_embedding_layer.weight.unsqueeze(0)
        X_sparse = X_sparse.view(-1, sparse_embedding_dim * x_sparse_dim)
        X = torch.concat((X, X_sparse, X_seq), dim=1)
        # stage 1
        X_rep = X.unsqueeze(2) * self.x_rep.weight.unsqueeze(0)
        
        # sbase net
        dims = X_rep.size()
        _X_rep = X_rep / torch.linalg.norm(X_rep, dim=1, keepdim=True)
        xx, xx_weight = self.self_attn(_X_rep, _X_rep, _X_rep)

        X_attn_feature = X.unsqueeze(1)  # [1024, 1, 12]
        X_attn_feature = torch.bmm(X_attn_feature, xx_weight)  # [1024, 1, 12]
        X_attn_feature = X_attn_feature.squeeze(1)  # [1024, 12]

        sx, sx_weight = self.interaction_attn_sx(S, X)
        rx, rx_weight = self.interaction_attn_rx(R, X)

        x = torch.cat([X_attn_feature, sx, rx], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, query, key, value, mask=None):
        attn_output, _ = self.attention(query, key, value, attn_mask=mask)
        return attn_output

class SequentialFeatureExtractor(nn.Module):
    def __init__(self, embed_dim, num_heads, fc_layers):
        super(SequentialFeatureExtractor, self).__init__()
        self.multi_head_attention = MultiHeadAttention(embed_dim, num_heads)
        self.fc_layers = nn.ModuleList([nn.Linear(in_features, out_features) for in_features, out_features in zip(fc_layers[:-1], fc_layers[1:])])

    def forward(self, x):
        # Reorder dimensions for attention: [seq_len, batch_size, seq_dim]
        x = x.permute(2, 0, 1)
        attn_out = self.multi_head_attention(x, x, x)

        # Reshape and flatten: [seq_len, batch_size, embed_dim] -> [batch_size, seq_len * embed_dim]
        attn_out = attn_out.permute(1, 0, 2).contiguous()
        attn_out_flat = attn_out.view(attn_out.size(0), -1)

        # Pass through fully connected layers
        out = attn_out_flat
        for fc_layer in self.fc_layers:
            out = F.relu(fc_layer(out))

        return out



