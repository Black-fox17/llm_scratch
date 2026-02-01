import torch.nn as nn
import torch
from app.transformer.modern.block import TransformerBlock

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, n_layers, dropout, window_size = 512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dropout=dropout, window_size = window_size)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.Identity()
        self.lm_head = nn.Linear(d_model, vocab_size, bias = False)
        self.window_size = window_size

    def forward_without_kv_cache(self, idx):
        B, T = idx.shape

        pos = torch.arange(T, device=idx.device)

        x = self.token_emb(idx)

        for block in self.blocks:
            x,_ = block(x, None)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits
    def forward_with_kv_cache(self, idx, kv_caches):
        B, T = idx.shape
    
        pos_start = 0
        if kv_caches[0] is not None:
            pos_start = kv_caches[0]["k"].size(2)
    
        pos = torch.arange(pos_start, pos_start + T, device=idx.device)
    
        x = self.token_emb(idx)
    
        new_kv_caches = []
        for block, kv in zip(self.blocks, kv_caches):
            x, new_kv = block(x, kv)
            new_kv_caches.append(new_kv)
    
        x = self.ln_f(x)
        logits = self.lm_head(x)
    
        return logits, new_kv_caches

    @torch.no_grad()
    def generate_with_kv_cache(self, idx, max_new_tokens=150, temperature=1.0, top_k=50, top_p=None):
        self.eval()
        B = idx.size(0)
        kv_caches = [None] * len(self.blocks)
        
        logits, kv_caches = self.forward_with_kv_cache(idx, kv_caches)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        logits = torch.clamp(logits, -50, 50)
        probs = torch.softmax(logits, dim=-1)
        
        if top_k is not None:
            top_k_val = min(top_k, probs.size(-1))
            v, _ = torch.topk(probs, top_k_val, dim=-1)
            probs = torch.where(probs < v[:, [-1]], 0.0, probs)
        
        if top_p is not None:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            probs.zero_().scatter_(dim=-1, index=sorted_idx, src=sorted_probs)
        
        probs_sum = probs.sum(dim=-1, keepdim=True)
        probs = probs / probs_sum
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
        
        for _ in range(max_new_tokens - 1):
            last_token = idx[:, -1:].contiguous()
            logits, kv_caches = self.forward_with_kv_cache(last_token, kv_caches)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            logits = torch.clamp(logits, -50, 50)
            probs = torch.softmax(logits, dim=-1)
            
            if top_k is not None:
                top_k_val = min(top_k, probs.size(-1))
                v, _ = torch.topk(probs, top_k_val, dim=-1)
                probs = torch.where(probs < v[:, [-1]], 0.0, probs)
            
            if top_p is not None:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                probs.zero_().scatter_(dim=-1, index=sorted_idx, src=sorted_probs)
            
            probs_sum = probs.sum(dim=-1, keepdim=True)
            probs = probs / probs_sum
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        
        return idx

    @torch.no_grad()
    def generate_without_kv_cache(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 150,
        temperature: float = 1.0,
        top_k: int | None = 50,
        top_p: float | None = None,
    ):
        self.eval()
    
        if idx.size(1) == 0:
            idx = torch.full((idx.size(0), 1), 10, dtype=torch.long, device=idx.device)
    
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.window_size:]
            logits = self(idx_cond)[:, -1, :]
    
            logits = logits / max(temperature, 1e-6)
            logits = torch.clamp(logits, -50, 50)
    
            probs = torch.softmax(logits, dim=-1)
            
            # --- top-k ---
            if top_k is not None:
                top_k = min(top_k, probs.size(-1))
                v, _ = torch.topk(probs, top_k, dim=-1)
                cutoff = v[:, -1].unsqueeze(-1)
                probs = torch.where(probs < cutoff, 0.0, probs)
                
    
            # --- top-p (nucleus) ---
            if top_p is not None:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
    
                mask = cumulative > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
    
                sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                probs.zero_().scatter_(dim=-1, index=sorted_idx, src=sorted_probs)
    
            # --- final safety ---
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            row_sums = probs.sum(dim=-1, keepdim=True)
    
            if (row_sums == 0).any():
                raise RuntimeError("Invalid sampling distribution: all probabilities are zero")
    
            # probs = probs + 1e-9
            probs = probs / row_sums
    
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
    
        return idx


