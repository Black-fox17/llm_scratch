import torch
import torch.nn as nn
import torch.nn.functional as F

import math
class BPELMDataLoader:
    def __init__(self, corpus, tokenizer, batch_size, seq_len, vocab_size, split=(0.95, 0.05)):
        # Load text
        # text = open(path, 'r', encoding='utf-8').read()

        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        text = ""
        for tx in corpus:
            text += "\n" + tx

        # Encode entire corpus as token IDs
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        self.data = data

        n = len(data)
        n_train = int(n * split[0])
        n_val = int(n * split[1])

        self.train = data[:n_train]
        self.val = data[n_train:n_train + n_val]

        self.batch_size = batch_size
        self.seq_len = seq_len

        self.train_ptr = 0
        self.val_ptr = 0

    def _next_batch(self, split):
        data = self.train if split == "train" else self.val
        ptr = self.train_ptr if split == "train" else self.val_ptr

        required_size = self.batch_size * self.seq_len + self.seq_len
        if len(data) < required_size:
            raise ValueError(
                f"Not enough tokens. Need {required_size}, got {len(data)}"
            )

        x = torch.zeros(self.batch_size, self.seq_len, dtype=torch.long)
        y = torch.zeros(self.batch_size, self.seq_len, dtype=torch.long)

        for i in range(self.batch_size):
            start = ptr + i * self.seq_len
            end = start + self.seq_len

            if end + 1 >= len(data):
                start = i * self.seq_len
                end = start + self.seq_len

            x[i] = data[start:end]
            y[i] = data[start + 1:end + 1]

        ptr += self.batch_size * self.seq_len
        max_ptr = len(data) - self.seq_len - 1

        if split == "train":
            self.train_ptr = ptr % max_ptr
        else:
            self.val_ptr = ptr % max_ptr

        return x, y

    def next_train_batch(self):
        return self._next_batch("train")

    def next_val_batch(self):
        return self._next_batch("val")
