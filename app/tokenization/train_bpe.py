from tokenizer import BPETokenizer
from dataloader import get_dataset, create_corpus

def main():
    print("Loading dataset...")
    ds = get_dataset()
    
    print("Creating corpus...")
    train_corpus = create_corpus(ds)
    print(f"Corpus created with {len(train_corpus)} entries.")
    
    vocab_size = 500
    tokenizer = BPETokenizer()
    
    print(f"Starting BPE training to vocab size {vocab_size}...")
    tokenizer.train(train_corpus, vocab_size)
    

if __name__ == "__main__":
    main()
