from .cell import LSTMCell
import torch


class StackedLSTM:
    def __init__(self, vocab_size, hidden_size, num_layers, device):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.embed = torch.randn(
            vocab_size,
            hidden_size,
            device=device
        ) * 0.1
        self.embed.requires_grad_()

        self.layers = []
        for _ in range(num_layers):
            self.layers.append(
                LSTMCell(hidden_size, hidden_size, device)
            )

        self.Wy = torch.randn(
            hidden_size,
            vocab_size,
            device=device
        ) * 0.1

        self.by = torch.zeros(
            vocab_size,
            device=device
        )

        self.Wy.requires_grad_()
        self.by.requires_grad_()
    def parameters(self):
        params = [self.embed, self.Wy, self.by]
        for l in self.layers:
            params += [l.W, l.b]
        return params

    def __call__(self, x, state):
        return self.forward(x, state)

    def forward(self, x, state):
        h_list, c_list = state
        x = self.embed[x]
        
        new_h_list = []
        new_c_list = []
        for i, layer in enumerate(self.layers):
            h, c = layer(x, h_list[i], c_list[i])
            new_h_list.append(h)
            new_c_list.append(c)
            x = h
        
        logits = x @ self.Wy + self.by
        
        return logits, (new_h_list, new_c_list)