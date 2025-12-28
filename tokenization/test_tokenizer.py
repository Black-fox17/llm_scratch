from tokenizer import BPETokenizer
import os

def test_tokenizer():
    corpus = [
        "This is a test sentence.",
        "BPE stands for Byte Pair Encoding.",
        "Tokenization is important for LLMs.",
        "Let's test if it can merge correctly.",
        "Byte pair encoding is a simple form of data compression."
    ]
    
    # Train
    print("Training on small corpus...")
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=100)
    
    # Test tokenize
    test_text = "Byte pair encoding is test."
    tokens = tokenizer.tokenize(test_text)
    print(f"Tokenized '{test_text}': {tokens}")
    
    # Test encode/decode
    ids = tokenizer.encode(test_text)
    print(f"Encoded '{test_text}': {ids}")
    decoded = tokenizer.decode(ids)
    print(f"Decoded: '{decoded}'")
    
    assert decoded == test_text, f"Decoded text '{decoded}' does not match original '{test_text}'"
    print("Encode/Decode verification passed!")
    
    # Test save/load
    print("Testing Save/Load...")
    tokenizer.save("test_vocab.json", "test_merges.txt")
    
    new_tokenizer = BPETokenizer()
    new_tokenizer.load("test_vocab.json", "test_merges.txt")
    
    new_ids = new_tokenizer.encode(test_text)
    assert new_ids == ids, "Loaded tokenizer produced different IDs"
    print("Save/Load verification passed!")
    
    # Cleanup
    os.remove("test_vocab.json")
    os.remove("test_merges.txt")
    print("Test cleanup done.")

if __name__ == "__main__":
    test_tokenizer()
