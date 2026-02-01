import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
import numpy as np
import os

def visualize_position_embeddings(X: np.ndarray,title, save_path="position_embeddings.gif"):
    if not os.path.exists("assets"):
        os.makedirs("assets")
    tensor_3d = PCA(n_components=3).fit_transform(X)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    seq_len = X.shape[0]

    lims = {
        "x": (tensor_3d[:, 0].min(), tensor_3d[:, 0].max()),
        "y": (tensor_3d[:, 1].min(), tensor_3d[:, 1].max()),
        "z": (tensor_3d[:, 2].min(), tensor_3d[:, 2].max()),
    }

    def update(frame):
        ax.clear()
        ax.scatter(
            tensor_3d[:frame+1, 0],
            tensor_3d[:frame+1, 1],
            tensor_3d[:frame+1, 2],
            c=np.arange(frame+1),
            cmap="viridis",
            s=100
        )
        ax.set_xlim(lims["x"])
        ax.set_ylim(lims["y"])
        ax.set_zlim(lims["z"])
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(f"{title} — Position {frame}")
        
    ani = FuncAnimation(fig, update, frames=seq_len, interval=200, repeat=True)
    ani.save(os.path.join("assets",save_path), writer='pillow', fps=10)
    plt.close(fig)