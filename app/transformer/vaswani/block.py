import torch.nn as nn
from app.norms.layernorm import LayerNorm
from app.attention.vaswani.attention import MultiHeadAttention
from app.neural_nets.feedforward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model)

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.attn(x)
        x = self.ln2(x)
        x = x + self.ff(x)
        return x