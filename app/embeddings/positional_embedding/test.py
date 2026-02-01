from position import Position
from visualizer import visualize_position_embeddings
import numpy as np
import torch

seq_len = 16
d_model = 64
e = torch.randn(d_model)
X_without_position = e.repeat(seq_len, 1)
pos = Position(X_without_position)
X_with_sinuisodal_position = pos.apply_sinusodal_positional_encoding()
X_with_rope_position = pos.rotary_positional_encoding()
visualize_position_embeddings(X_without_position.numpy(),"Without Position", "without_position.gif")
visualize_position_embeddings(X_with_sinuisodal_position.numpy(), "With Sinuisodal Position", "with_sinuisodal_position.gif")
visualize_position_embeddings(X_with_rope_position.numpy(), "With Rotary Position", "with_rope_position.gif")