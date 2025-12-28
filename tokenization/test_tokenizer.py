from tokenizer import BPETokenizer
import os

def test_tokenizer():
    corpus = [
        "One day, a little girl named Lily found a needle in her room.",
        "She knew it was difficult to play with because it was sharp.",
        "Lily wanted to share the needle with her mom.",
        "She wanted her mom to sew a button on her shirt.",
        "Lily went to her mom and told her about the needle.",
        "She asked her mom to share the needle and sew her shirt.",
        "Her mom smiled and agreed to help.",
        "Together, they shared the needle and sewed the button.",
        "It was not difficult because they worked together.",
        "After finishing, Lily thanked her mom.",
        "They both felt happy because they shared and helped each other."
    ]

    
    tokenizer = BPETokenizer().load()

    tokenized_corpus = [tokenizer.tokenize(token) for token in corpus]
    encoded_corpus = [tokenizer.encode(token) for token in tokenized_corpus]
    for tokenized in tokenized_corpus:
        print(tokenized)

    for encoded in encoded_corpus:
        print(encoded)

    print(tokenizer.decode([11, 7, 22, 45, 47, 45, 45, 45, 2, 45, 35, 31, 7, 45, 45, 45, 27, 45, 22, 14, 31, 45, 45, 45, 7, 32, 45, 28, 28, 45, 47, 45, 45, 45, 2, 45, 22, 45, 20, 32, 45, 18, 45, 10, 22, 45, 45, 45, 31, 45, 7, 22, 45, 47, 45, 45, 45, 10, 45, 7, 32, 45, 46, 22, 21, 45, 45, 45, 32, 45, 26, 21, 45, 45, 45, 7, 22, 45, 14, 28, 45, 22, 21, 45, 45, 45, 22, 45, 32, 45, 20, 7, 45, 45, 45, 35, 45, 31, 7, 22, 46, 45, 33]))

if __name__ == "__main__":
    test_tokenizer()
