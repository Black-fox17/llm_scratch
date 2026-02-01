import torch.nn as nn
from app.utils.rope import get_rope_sincos, apply_rope

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout = 0.1, window_size = 512):
        super().__init__()
        assert d_model % nhead == 0

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.window_size = window_size

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.last_attn = None

    def split_heads(self, x):
        B, T, D = x.shape
        x = x.view(B, T, self.nhead, self.d_k)
        return x.transpose(1, 2)   # (B, h, T, d_k)

    def forward(self, x, kv_cache=None):
        B, T_new, _ = x.shape
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))
    
        past_len = kv_cache["k"].size(2) if kv_cache is not None else 0
        sincos = get_rope_sincos(T_new, self.d_k, x.device, offset=past_len)
        Q = apply_rope(Q, sincos)
        K = apply_rope(K, sincos)
    
        if kv_cache is not None:
            K = torch.cat([kv_cache["k"], K], dim=2)
            V = torch.cat([kv_cache["v"], V], dim=2)
            
            if K.size(2) > self.window_size:
                K = K[:, :, -self.window_size:]
                V = V[:, :, -self.window_size:]
    
        seq_len = K.size(2)
        causal_mask = torch.ones(T_new, seq_len, device=x.device, dtype=torch.bool)
        for i in range(T_new):
            causal_mask[i, past_len + i + 1:] = False
    
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        self.last_attn = attn
    
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, T_new, self.d_model)
        out = self.proj_dropout(self.W_o(out))
        new_cache = {"k": K, "v": V}
        return out, new_cache