from cell import LSTMCell   
import torch

class StackedLSTM:
    def __init__(self, vocab_size, hidden_size, num_layers):
        self.embed = torch.randn(vocab_size, hidden_size) * 0.1
        self.embed.requires_grad_()

        self.layers = []
        for i in range(num_layers):
            inp = hidden_size if i > 0 else hidden_size
            self.layers.append(LSTMCell(inp, hidden_size))

        self.Wy = torch.randn(hidden_size, vocab_size) * 0.1
        self.by = torch.zeros(vocab_size)

        self.Wy.requires_grad_()
        self.by.requires_grad_()

    def parameters(self):
        params = [self.embed, self.Wy, self.by]
        for l in self.layers:
            params += [l.W, l.b]
        return params

    def forward(self, x, state):
        x = self.embed[x]
        for l in self.layers:
            x, _ = l(x, state)
        return x @ self.Wy + self.by
        
