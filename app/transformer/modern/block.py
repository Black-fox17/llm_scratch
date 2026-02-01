import torch.nn as nn
from app.attention.modern.attention import MultiHeadAttention
from app.norms.rmsnorm import RMSNorm
from app.neural_nets.feedforward.modern import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, mlp_ratio=4, dropout = 0.1, window_size = 512):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, nhead, dropout, window_size)
        self.ln2 = RMSNorm(d_model)

        self.mlp = FeedForward(d_model, hidden_ratio=mlp_ratio, dropout = dropout)
    def forward(self, x, kv_cache=None):
        attn_out, kv_cache = self.attn(self.ln1(x), kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, kv_cache
    
