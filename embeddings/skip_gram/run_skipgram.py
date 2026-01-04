
"""
Skip-Gram Word2Vec Example Runner.

This script demonstrates how to:
1. Load a dataset (Toy or Wikitext)
2. Train a Skip-Gram model using Full Softmax
3. Train a Skip-Gram model using Negative Sampling
4. Visualize the resulting embeddings

Usage:
    python run_skipgram.py
"""
import torch
import os
from dataset import SkipGramDataset
from trainer import SkipGramTrainer
from visualizer import visualize_embeddings

def main():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Data Loading
    # Options: "toy" or "wikitext"
    print("Initializing Dataset...")
    dataset = SkipGramDataset(source="toy", window_size=2, device=device)
    
    # 3. Initialize Trainer
    trainer = SkipGramTrainer(dataset, embedding_dim=10, device=device)
    
    # ---------------------------------------------------------
    # Experiment A: Full Softmax
    # ---------------------------------------------------------
    print("\n[Experiment A] Training with Full Softmax...")
    # Using fewer epochs for demonstration speed
    w_in_softmax = trainer.train_full_softmax(num_epochs=1000, lr=0.01)
    
    visualize_embeddings(
        w_in_softmax, 
        dataset.tokenizer, 
        save_path="embeddings_softmax.png"
    )

    # ---------------------------------------------------------
    # Experiment B: Negative Sampling
    # ---------------------------------------------------------
    print("\n[Experiment B] Training with Negative Sampling...")
    w_in_neg = trainer.train_negative_sampling(num_epochs=2000, lr=0.01, neg_samples=5)
    
    visualize_embeddings(
        w_in_neg, 
        dataset.tokenizer, 
        save_path="embeddings_neg_sampling.png"
    )

    print("\nDone! Check the .png files for visualizations.")

if __name__ == "__main__":
    main()
