class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        self.W = torch.randn(input_size + hidden_size, 4 * hidden_size) * 0.1
        self.b = torch.zeros(4 * hidden_size)

        self.W.requires_grad_()
        self.b.requires_grad_()

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = combined @ self.W + self.b
        i, f, g, o = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_next = f * c + i * g  
        h_next = o * torch.tanh(c_next)
        return h_next, c_next
