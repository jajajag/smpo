from __future__ import annotations
import torch
import torch.nn as nn

class SafetyQCritic(nn.Module):
    """Q_c(s,a) ≥ 0 using softplus head."""
    def __init__(self, obs_dim, act_dim, hidden=(256,256)):
        super().__init__()
        self.net = MLP_like(obs_dim + act_dim, 1, hidden)
        self.softplus = nn.Softplus()

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        q = self.net(x)
        return self.softplus(q).squeeze(-1)

# Lightweight internal MLP to avoid import cycle
class MLP_like(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256,256)):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

@torch.no_grad()
def q_cost_target(next_obs, policy, gamma, critic: SafetyQCritic, n_action_samples=1):
    """E_{a'~pi}[Q_c(s',a')] Monte Carlo estimate."""
    vals = []
    for _ in range(n_action_samples):
        a_prime, _ = policy.sample(next_obs)
        q_next = critic(next_obs, a_prime)
        vals.append(q_next)
    return gamma * torch.stack(vals, dim=0).mean(0)
