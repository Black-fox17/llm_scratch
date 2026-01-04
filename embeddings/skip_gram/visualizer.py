
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
import numpy as np

def visualize_embeddings(W_in, tokenizer, save_path="word_embeddings.png"):
    """
    Visualizes the embedding space by projecting high-dimensional vectors to 2D
    using SVD (Singular Value Decomposition).
    
    Args:
        W_in (torch.Tensor or numpy.ndarray): Input word embeddings.
        tokenizer (SimpleTokenizer): Tokenizer containing vocab mappings.
        save_path (str): File path to save the plot.
    """
    print(f"Generating visualization: {save_path}")
    
    # Ensure CPU numpy array
    if hasattr(W_in, 'cpu'):
        embeddings_cpu = W_in.cpu().detach().numpy()
    else:
        embeddings_cpu = W_in
    
    # Reduce dimensions
    # We use TruncatedSVD (similar to PCA) to flatten 10D/300D -> 2D
    svd = TruncatedSVD(n_components=2)
    try:
        W1_dec = svd.fit_transform(embeddings_cpu)
    except ValueError:
        # Fallback if vocab is too small for SVD n=2
        print("Vocab too small for SVD, using first 2 dims")
        W1_dec = embeddings_cpu[:, :2]

    x = W1_dec[:, 0]
    y = W1_dec[:, 1]

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=x, y=y)

    # Add labels
    # Iterate through vocab to label points
    for i in range(len(x)):
        if i in tokenizer.id2word:
            word = tokenizer.id2word[i]
            plt.text(x[i]+0.01, y[i], word, fontsize=10)
         
    plt.title("Word Embeddings Visualization (Skip-Gram)")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
