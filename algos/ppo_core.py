from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch.distributions import Normal

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256,256), act=nn.Tanh, out_act=None):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), act()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        if out_act is not None:
            layers += [out_act()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256,256), log_std_init=-0.5, log_std_min=-5.0, log_std_max=2.0):
        super().__init__()
        self.net = MLP(obs_dim, act_dim, hidden)
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs):
        mu = self.net(obs)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        return mu, std

    def dist(self, obs):
        mu, std = self.forward(obs)
        return Normal(mu, std)

    def sample(self, obs):
        d = self.dist(obs)
        a = d.rsample()
        logp = d.log_prob(a).sum(-1)
        return a, logp

    def log_prob(self, obs, act):
        d = self.dist(obs)
        return d.log_prob(act).sum(-1)

class ValueFunction(nn.Module):
    def __init__(self, obs_dim, hidden=(256,256)):
        super().__init__()
        self.v = MLP(obs_dim, 1, hidden)
    def forward(self, obs):
        return self.v(obs).squeeze(-1)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256,256)):
        super().__init__()
        self.pi = GaussianPolicy(obs_dim, act_dim, hidden)
        self.v = ValueFunction(obs_dim, hidden)

    @torch.no_grad()
    def act(self, obs):
        a, logp = self.pi.sample(obs)
        v = self.v(obs)
        return a, v, logp

# PPO Loss utilities
class PPOLoss:
    def __init__(self, clip_ratio=0.2, vf_coef=0.5, ent_coef=0.0, max_grad_norm=0.5):
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

    def compute(self, ac: ActorCritic, obs, act, adv, logp_old, ret):
        # Policy loss
        logp = ac.pi.log_prob(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
        pg_loss = -(torch.min(ratio * adv, clip_adv)).mean()
        # Value loss
        v = ac.v(obs)
        v_loss = 0.5 * (ret - v).pow(2).mean()
        # Entropy bonus
        mu, std = ac.pi.forward(obs)
        ent = (0.5 + 0.5*math.log(2*math.pi) + torch.log(std)).sum(-1).mean()
        loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * ent
        approx_kl = (logp_old - logp).mean().item()
        clip_frac = (torch.abs(ratio - 1.0) > self.clip_ratio).float().mean().item()
        return loss, {
            'loss/pg': pg_loss.item(),
            'loss/v': v_loss.item(),
            'loss/ent': ent.item(),
            'policy/approx_kl': approx_kl,
            'policy/clip_frac': clip_frac,
        }
