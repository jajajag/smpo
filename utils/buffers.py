# smpo/utils/buffers.py
import torch

def discount_cumsum(x: torch.Tensor, discount: float) -> torch.Tensor:
    y = torch.zeros_like(x)
    running = 0.0
    for t in range(x.shape[0] - 1, -1, -1):
        running = x[t] + discount * running
        y[t] = running
    return y

class RolloutBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, device='cpu'):
        self.obs = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.act = torch.zeros((size, act_dim), dtype=torch.float32, device=device)
        self.rew = torch.zeros(size, dtype=torch.float32, device=device)
        self.cost = torch.zeros(size, dtype=torch.float32, device=device)
        self.done = torch.zeros(size, dtype=torch.float32, device=device)
        self.logp = torch.zeros(size, dtype=torch.float32, device=device)
        self.val = torch.zeros(size, dtype=torch.float32, device=device)
        self.cum_cost = torch.zeros(size, dtype=torch.float32, device=device)

        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = size
        self.gamma, self.lam = gamma, lam
        self.device = device

        # will be filled in finish_path
        self.adv = torch.zeros(size, dtype=torch.float32, device=device)
        self.ret = torch.zeros(size, dtype=torch.float32, device=device)

    def add(self, obs, act, rew, cost, done, logp, val, cum_cost):
        assert self.ptr < self.max_size
        i = self.ptr
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.cost[i] = cost
        self.done[i] = done
        self.logp[i] = logp
        self.val[i] = val
        self.cum_cost[i] = cum_cost
        self.ptr += 1

    def finish_path(self, last_val=0.0):
        """Finish current trajectory segment [path_start_idx, ptr)."""
        if self.ptr == self.path_start_idx:
            # nothing to finish
            return
        path_slice = slice(self.path_start_idx, self.ptr)

        rews = self.rew[path_slice]
        dones = self.done[path_slice]
        vals_seg = self.val[path_slice]
        device = self.val.device

        # bootstrap value (episodic: caller可传0.0；截断：可传V(s_last))
        vals = torch.cat([vals_seg, torch.as_tensor([last_val], device=device)])

        deltas = rews + self.gamma * vals[1:] * (1.0 - dones) - vals[:-1]
        adv = discount_cumsum(deltas, self.gamma * self.lam)
        ret = discount_cumsum(rews, self.gamma)

        self.adv[path_slice] = adv
        self.ret[path_slice] = ret

        # move start to current ptr
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size, "Buffer not full"
        adv = self.adv[:self.ptr]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        data = dict(
            obs=self.obs[:self.ptr],
            act=self.act[:self.ptr],
            ret=self.ret[:self.ptr],
            adv=adv,
            logp=self.logp[:self.ptr],
            cum_cost=self.cum_cost[:self.ptr],
        )
        return data

    def reset(self):
        self.ptr = 0
        self.path_start_idx = 0
