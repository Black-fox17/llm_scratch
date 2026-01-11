import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
import numpy as np
import os

def visualize_position_embeddings(X: np.ndarray, save_path="position_embeddings.gif"):
    if not os.path.exists("assets"):
        os.makedirs("assets")
    tensor_3d = PCA(n_components=3).fit_transform(X)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    seq_len = X.shape[0]
    def update(frame):
        ax.clear()
        ax.scatter(tensor_3d[:frame+1, 0], tensor_3d[:frame+1, 1], tensor_3d[:frame+1, 2],
            c=np.arange(frame+1), cmap='viridis', s=100)
        ax.set_xlim(tensor_3d[:, 0].min() - 0.1, tensor_3d[:, 0].max() + 0.1)
        ax.set_ylim(tensor_3d[:, 1].min() - 0.1, tensor_3d[:, 1].max() + 0.1)
        ax.set_zlim(tensor_3d[:, 2].min() - 0.1, tensor_3d[:, 2].max() + 0.1)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title(f'Position {frame}/{seq_len-1}')

    ani = FuncAnimation(fig, update, frames=seq_len, interval=200, repeat=True)
    ani.save(os.path.join("assets",save_path), writer='pillow', fps=10)
    plt.close(fig)