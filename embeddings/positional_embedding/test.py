from position import Position
from visualizer import visualize_position_embeddings
import numpy as np

X_without_position = np.random.randn(10, 10)
pos = Position().sinusodal_positional_encoding(10,10)
X_with_sinuisodal_position = X_without_position + pos
visualize_position_embeddings(X_without_position, "without_position.gif")
visualize_position_embeddings(X_with_sinuisodal_position, "with_sinuisodal_position.gif")