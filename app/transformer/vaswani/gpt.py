import torch
import torch.nn as nn
from app.transformer.vaswani.block import TransformerBlock
from app.norms.layernorm import LayerNorm

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, block_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_out = LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.token_embedding(x) + self.position_embedding(torch.arange(x.size(1), device=x.device))
        for block in self.blocks:
            x = block(x)
        x = self.ln_out(x)
        x = self.fc_out(x)
        return x
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)[:, -1, :]
            logits = logits / max(temperature, 1e-6)
            logits = torch.clamp(logits, -50, 50)
            probs = torch.softmax(logits, dim=-1)
            if top_k is not None:
                top_k = min(top_k, probs.size(-1))
                v, _ = torch.topk(probs, top_k, dim=-1)
                cutoff = v[:, -1].unsqueeze(-1)
                probs = torch.where(probs < cutoff, 0.0, probs)
            if top_p is not None:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                probs.zero_().scatter_(dim=-1, index=sorted_idx, src=sorted_probs)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            row_sums = probs.sum(dim=-1, keepdim=True)
            if (row_sums == 0).any():
                raise RuntimeError("Invalid sampling distribution: all probabilities are zero")
            probs = probs / row_sums
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx