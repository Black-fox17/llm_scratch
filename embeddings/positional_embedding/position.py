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
        d = self.d_model
        seq_len = self.seq_len
        X = self.X  # shape: (seq_len, d)

        dim = torch.arange(0, d, 2, device=X.device).float()
        inv_freq = 1.0 / (10000 ** (dim / d))

        out = []

        for pos in range(seq_len):
            x = X[pos]

            x_even = x[0::2]
            x_odd  = x[1::2]

            theta = pos * inv_freq
            cos = torch.cos(theta)
            sin = torch.sin(theta)

            rot_even = x_even * cos - x_odd * sin
            rot_odd  = x_even * sin + x_odd * cos

            x_rot = torch.empty_like(x)
            x_rot[0::2] = rot_even
            x_rot[1::2] = rot_odd

            out.append(x_rot)

        return torch.stack(out)
