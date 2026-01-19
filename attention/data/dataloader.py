import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CharLMDataLoader:
    def __init__(self, path, batch_size, seq_len, split=(0.95, 0.05)):
        text = open(path, 'r', encoding='utf-8').read()
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {c:i for i,c in enumerate(chars)}
        self.itos = {i:c for c,i in self.stoi.items()}

        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

        n = len(data)
        n_train = int(n * split[0])
        n_val = int(n * split[1])

        self.train = data[:n_train]
        self.val = data[n_train:n_train+n_val]

        self.batch_size = batch_size
        self.seq_len = seq_len

        self.train_ptr = 0
        self.val_ptr = 0

    def _next_batch(self, split):
        data = self.train if split == 'train' else self.val
        ptr = self.train_ptr if split == 'train' else self.val_ptr

        # Check if we have enough data
        required_size = self.batch_size * self.seq_len + self.seq_len
        if len(data) < required_size:
            raise ValueError(
                f"Not enough data! Need at least {required_size} characters, "
                f"but only have {len(data)}. Try reducing batch_size or seq_len, "
                f"or add more training data."
            )

        x = torch.zeros(self.batch_size, self.seq_len, dtype=torch.long)
        y = torch.zeros(self.batch_size, self.seq_len, dtype=torch.long)

        for i in range(self.batch_size):
            start = ptr + i * self.seq_len
            end = start + self.seq_len

            # Ensure we don't go past the end of data
            if end >= len(data):
                # Wrap around to beginning
                start = i * self.seq_len
                end = start + self.seq_len

            x[i] = data[start:end]
            y[i] = data[start+1:end+1]

        ptr += self.batch_size * self.seq_len
        if split == 'train':
            self.train_ptr = ptr % (len(self.train) - self.seq_len - 1)
        else:
            self.val_ptr = ptr % (len(self.val) - self.seq_len - 1)

        return x, y

    def next_train_batch(self):
        return self._next_batch('train')

    def next_val_batch(self):
        return self._next_batch('val')