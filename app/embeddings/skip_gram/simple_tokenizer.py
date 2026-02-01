
import re
from collections import Counter

class SimpleTokenizer:
    """
    A simple whitespace-based tokenizer for educational purposes.
    Reflects the 'under the hood' logic of standardizing and mapping words to IDs.
    """
    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self.vocab_size = 0
        self.word_counts = Counter()

    def clean_sentence(self, tokens):
        """
        Standardizes tokens: lowercase and alphanumeric check.
        This mimics basic normalization steps in NLP pipelines.
        """
        return [
            t.lower()
            for t in tokens
            if re.search(r"[a-z0-9]", t)
        ]

    def train(self, corpus):
        """
        Builds vocabulary from a list of sentences (corpus).
        
        Args:
            corpus (list of str): List of sentences.
        """
        self.word_counts = Counter()
        for sent in corpus:
            tokens = self.clean_sentence(sent.split())
            self.word_counts.update(tokens)
        
        vocab = list(self.word_counts.keys())
        self.word2id = {w: i for i, w in enumerate(vocab)}
        self.id2word = {i: w for w, i in self.word2id.items()}
        self.vocab_size = len(vocab)
        
    def encode(self, sentence):
        """
        Converts a sentence string to a list of integer IDs.
        Ignores words not in the training vocabulary.
        """
        tokens = self.clean_sentence(sentence.split())
        return [self.word2id[t] for t in tokens if t in self.word2id]
