import torch

def get_rope_sincos(seq_len, dim, device, offset=0, max_position=512):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device) / dim))
    positions = torch.arange(offset, offset + seq_len, device=device)
    positions = torch.clamp(positions, max=max_position - 1)
    freqs = torch.einsum('i,j->ij', positions, inv_freq)
    sin = freqs.sin()
    cos = freqs.cos()
    return sin, cos

def apply_rope(x, sincos):
    # x: (B, h, T, d_k)
    sin, cos = sincos  # (T, d_k/2)
    x1, x2 = x[..., ::2], x[..., 1::2]
    x_rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rotated.flatten(-2)