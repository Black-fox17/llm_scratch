import re
import json
from collections import defaultdict
from .constants import GPT_REGEXP

class BPETokenizer:
    def __init__(self, vocab=None, merges=None):
        self.vocab = vocab if vocab is not None else {}
        self.merges = merges if merges is not None else {}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def _pre_tokenize(self, text):
        return re.findall(GPT_REGEXP, text)

    def _compute_pair_freqs(self, splits, word_freqs):
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    def _merge_pair(self, a, b, splits, word_freqs):
        for word in word_freqs:
            split = splits[word]
            if len(split) == 1:
                continue

            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2 :]
                else:
                    i += 1
            splits[word] = split
        return splits

    def train(self, train_corpus, vocab_size):
        word_freqs = defaultdict(int)
        for text in train_corpus:
            new_words = self._pre_tokenize(text)
            for word in new_words:
                word_freqs[word] += 1

        alphabet = set()
        for word in word_freqs.keys():
            for letter in word:
                alphabet.add(letter)
        
        sorted_alphabet = sorted(list(alphabet))
        self.vocab = {"<|endoftext|>": 0}
        for i, char in enumerate(sorted_alphabet):
            self.vocab[char] = i + 1
        
        splits = {word: [c for c in word] for word in word_freqs.keys()}
        self.merges = {}

        while len(self.vocab) < vocab_size:
            pair_freqs = self._compute_pair_freqs(splits, word_freqs)
            if not pair_freqs:
                break
            
            best_pair = max(pair_freqs, key=pair_freqs.get)
            splits = self._merge_pair(*best_pair, splits, word_freqs)
            
            new_token = best_pair[0] + best_pair[1]
            self.merges[best_pair] = new_token
            self.vocab[new_token] = len(self.vocab)

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text):
        pre_tokenized_text = self._pre_tokenize(text)
        # We need to handle the case where characters in text were not in training alphabet
        # For simplicity in this demo, we'll just skip unknown characters or treat them as singletons
        
        final_tokens = []
        for word in pre_tokenized_text:
            split = [c for c in word]
            for pair, merge in self.merges.items():
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2 :]
                    else:
                        i += 1
            final_tokens.extend(split)
        
        return final_tokens

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.vocab.get(t, self.vocab.get("<|endoftext|>")) for t in tokens]

    def decode(self, ids):
        return "".join([self.inverse_vocab.get(i, "") for i in ids])

    def save(self, vocab_path, merges_path):
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=4, ensure_ascii=False)
        
        with open(merges_path, "w", encoding="utf-8") as f:
            f.write("#version: 0.2\n")
            for pair, merge in self.merges.items():
                f.write(f"{pair[0]} {pair[1]}\n")

    def load(self, vocab_path, merges_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        
        self.merges = {}
        with open(merges_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[1:]: # Skip version header
                parts = line.strip().split(" ")
                if len(parts) == 2:
                    self.merges[tuple(parts)] = "".join(parts)
        
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
