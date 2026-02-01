class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_ratio=4, dropout=0.1):
        super().__init__()
        hidden_dim = d_model * hidden_ratio
        self.w1 = nn.Linear(d_model, hidden_dim * 2)  # split for SwiGLU
        self.w2 = nn.Linear(hidden_dim, d_model)      # final projection
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()  # Swish activation
        # self.net = nn.Sequential(
        #     self.W1,
        #     nn.SiLU(),
        #     self.W2,
        #     nn.Dropout(dropout)
        # )


    def forward(self, x):
        # x: (B, T, d_model)
        x_proj = self.w1(x)           # (B, T, 2*hidden)
        x1, x2 = x_proj.chunk(2, dim=-1)  # split for gate
        return self.dropout(self.w2(x1 * self.act(x2)))
