import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        x = x.view(batch_size, seq_len, self.n_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, x, mask = True):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask:
            mask = torch.triu(torch.ones(attn.size(-2), attn.size(-1), device=attn.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous()
        out = out.view(out.size(0), out.size(1), self.d_model)
        out = self.W_o(out)
        return out

