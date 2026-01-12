import numpy as np
import torch
import math


class Position:
    def __init__(self, X):
        self.X = X
        self.seq_len = X.shape[0]
        self.d_model = X.shape[1]

    def apply_sinusodal_positional_encoding(self):
        pe = self._sinusodal_positional_encoding(self.seq_len, self.d_model)
        return self.X + pe
    def _sinusodal_positional_encoding(self, seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
        
    def rotary_positional_encoding(self):
        rotated_sequences = []
        def apply_rope(x, pos, d_model):
            x_even = x[0::2]
            x_odd = x[1::2]

            dim_indices = torch.arange(0, d_model, 2, device=x.device).float()
            inv_freq = 1.0 / (10000 ** (dim_indices / d_model))
            thetas = pos * inv_freq
            cos = torch.cos(thetas)
            sin = torch.sin(thetas)
            x_rot_even = x_even * cos - x_odd * sin
            x_rot_odd  = x_even * sin + x_odd * cos
            out = torch.empty_like(x)
            out[0::2] = x_rot_even
            out[1::2] = x_rot_odd
            return out
        for p in range(self.seq_len):
            rotated_vec = apply_rope(self.X, p, self.d_model)
            rotated_sequences.append(rotated_vec)
        return torch.stack(rotated_sequences)
