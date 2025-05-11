import torch
import torch.nn as nn

class ValueEstimator(nn.Module):
    def __init__(self, input_dim, hidden_layers, device="cpu"):
        super().__init__()
        assert len(hidden_layers) != 0, "AT LEAST ONE LAYER MUST BE SPECIFIED TO BUILD THE NEURAL NETWORK!"
        layers = []
        last_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)
        self.device = device
        self.to(device)

    def forward(self, x):
        return self.net(x).squeeze(-1)
