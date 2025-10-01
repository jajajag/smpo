import torch

class RunningNorm:
    def __init__(self, eps=1e-5, device='cpu'):
        self.count = torch.tensor(0.0, device=device)
        self.mean = torch.tensor(0.0, device=device)
        self.M2 = torch.tensor(0.0, device=device)
        self.eps = eps
        self.device = device

    def update(self, x: torch.Tensor):
        x = x.detach()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch = x.numel()
        delta = x.mean() - self.mean
        total = self.count + batch
        new_mean = self.mean + delta * (batch / (total + 1e-8))
        self.M2 = self.M2 + ((x - self.mean)**2).mean() * batch + (delta**2) * self.count * batch / (total + 1e-8)
        self.mean = new_mean
        self.count = total

    def norm(self, x: torch.Tensor):
        var = self.M2 / (self.count + 1e-8)
        std = torch.sqrt(var + self.eps)
        return (x - self.mean) / std
