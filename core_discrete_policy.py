import torch
import torch.nn as nn

class DiscreteFF(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, device="cpu"):
        super().__init__()
        assert len(hidden_layers) != 0, "AT LEAST ONE LAYER MUST BE SPECIFIED TO BUILD THE NEURAL NETWORK!"
        layers = []
        last_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.device = device
        self.to(device)

    def forward(self, x):
        return self.net(x)
