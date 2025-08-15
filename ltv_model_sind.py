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
        self.fc1 = torch.nn.Linear(x_dim + x_sparse_dim * sparse_embedding_dim + s_seq_dim, hidden_dim * 4) 
        self.fc2 = torch.nn.Linear(hidden_dim * 4, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, s_dim)  
        self.sparse_embedding_layer = sparse_embedding_layer


    def forward(self, X, X_sparse, X_seq):
        X_sparse = X_sparse.unsqueeze(2) * self.sparse_embedding_layer.weight.unsqueeze(0)
        X_sparse = X_sparse.view(-1, sparse_embedding_dim * x_sparse_dim)
        X = torch.concat((X, X_sparse, X_seq), dim=1)

        x = F.relu(self.fc1(X))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Y_Model(torch.nn.Module):
    def __init__(self, x_dim, s_dim, y_dim, hidden_dim, sparse_embedding_layer):
        super(Y_Model, self).__init__()
        self.fc1 = torch.nn.Linear(x_dim + x_sparse_dim * sparse_embedding_dim + s_dim + s_seq_dim, hidden_dim * 4) 
        self.fc2 = torch.nn.Linear(hidden_dim * 4, hidden_dim * 4)
        self.fc3 = torch.nn.Linear(hidden_dim * 4, y_dim)  
        self.hidden_dim = hidden_dim
        self.x_dim = x_dim
        self.s_dim = s_dim
        self.y_dim = y_dim
        self.sparse_embedding_layer = sparse_embedding_layer

    def forward(self, X, X_sparse, X_seq, S):
        X_sparse = X_sparse.unsqueeze(2) * self.sparse_embedding_layer.weight.unsqueeze(0)
        X_sparse = X_sparse.view(-1, sparse_embedding_dim * x_sparse_dim)
        X = torch.concat((X, X_sparse, X_seq), dim=1)

        x = torch.cat([X, S], dim=1)
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

    




