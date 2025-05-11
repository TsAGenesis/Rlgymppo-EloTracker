
import torch
import torch.nn as nn
import numpy as np

class DiscreteFF(nn.Module):
    def __init__(self, input_dim, layer_sizes, output_dim, device="cpu"):
        super().__init__()
        layers = []
        last = input_dim
        for sz in layer_sizes:
            layers += [nn.Linear(last, sz), nn.ReLU()]
            last = sz
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers).to(device)
        self.device = device

    def forward(self, x):
        return self.net(x)

class PPOPolicy(nn.Module):
    def __init__(self, state_dim, action_dim,
                 policy_layer_sizes, critic_layer_sizes,
                 device="cpu"):
        super().__init__()
        self.device = device
        self.actor  = DiscreteFF(state_dim, policy_layer_sizes, action_dim, device)
        self.critic = DiscreteFF(state_dim, critic_layer_sizes, 1, device)

    def get_action(self, obs: np.ndarray):
        x = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
        logits = self.actor(x)
        prob   = torch.softmax(logits, dim=-1)
        action = torch.multinomial(prob, num_samples=1)
        return action.item()
