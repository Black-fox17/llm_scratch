import os
from dataloader import get_wikitext_dataset, create_corpus
from tokenizer import BPETokenizer

def main():
    print("Loading dataset...")
    ds = get_wikitext_dataset()
    
    print("Creating corpus...")
    train_corpus = create_corpus(ds)
    
    print(f"Corpus created with {len(train_corpus)} entries.")
    
    vocab_size = 5000
    tokenizer = BPETokenizer()
    
    print(f"Starting BPE training to vocab size {vocab_size}...")
    tokenizer.train(train_corpus, vocab_size)
    
    print("Saving tokenizer...")
    tokenizer.save("vocab.json", "merges.txt")
    print("Done! Tokenizer saved to vocab.json and merges.txt")

if __name__ == "__main__":
    main()
