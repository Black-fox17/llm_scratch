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

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_ratio = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * hidden_ratio),
            nn.GELU(),
            nn.Linear(d_model * hidden_ratio, d_model),
            # nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.net(x)

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps) * self.gamma + self.beta

class GPTBlock(nn.Module):
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

class GPTSmall(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, block_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([GPTBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_out = LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.token_embedding(x) + self.position_embedding(torch.arange(x.size(1), device=x.device))
        for block in self.blocks:
            x = block(x)
        x = self.ln_out(x)
        x = self.fc_out(x)
        return x