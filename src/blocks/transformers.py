import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class SinCosPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()

        # Create position indices (shape: [max_len, 1])
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)

        # Compute the divisors for sine and cosine terms
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))

        # Compute the positional encodings
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0) 

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, L, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # Each is (B, L, num_heads, head_dim)

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = (attn_weights @ v).transpose(1, 2).reshape(B, L, C)
        return self.out_proj(out)

class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1, prenorm=True):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(embed_dim, ff_hidden_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        if self.prenorm:
            x = x + self.dropout(self.attn(self.norm1(x), mask))
            x = x + self.dropout(self.ffn(self.norm2(x)))
        else:
            x = self.norm1(x + self.dropout(self.attn(x, mask)))
            x = self.norm2(x + self.dropout(self.ffn(x)))
        return x

class Aggregation(nn.Module):
    # TODO check this 
    def __init__(self, d_model, method='mean'):
        super().__init__()
        self.method = method
        if method == 'weighted':
            self.attn_weights = nn.Linear(d_model, 1)  # Learnable attention scores

    def forward(self, hidden_states):
        if self.method == 'mean':
            return hidden_states.mean(dim=1)  # Mean pooling
        elif self.method == 'max':
            return hidden_states.max(dim=1)[0]  # Max pooling
        elif self.method == 'weighted':
            attn_scores = torch.softmax(self.attn_weights(hidden_states), dim=1)  # Compute weights
            return torch.sum(attn_scores * hidden_states, dim=1)  # Weighted sum
        
class Conv1by1(nn.Module):
    # TODO check this 
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, d_model, kernel_size=1)
    def forward(self, x):
        # x (N, L, c)
        x = x.permute(0,2,1)
        return self.conv(x).permute(0,2,1)