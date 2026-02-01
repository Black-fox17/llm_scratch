import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_ratio = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * hidden_ratio),
            nn.GELU(),
            nn.Linear(d_model * hidden_ratio, d_model),
            # nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.net(x)