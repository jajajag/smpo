from __future__ import annotations
import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from ..utils.buffers import RolloutBuffer
from ..utils.obs_norm import RunningNorm
from ..utils.weighting import safety_weight
from ..utils.schedule import dynamic_threshold
from ..envs.safety_gym_wrappers import make_safety_env
from .ppo_core import ActorCritic, PPOLoss
from ..critics.safety_critic import SafetyQCritic, q_cost_target

class SMPOTrainer:
    def __init__(self, env_id: str, seed=1, steps_per_epoch=30000, epochs=200, device='cpu',
                 gamma=0.99, lam=0.95, lr=3e-4, clip_ratio=0.2,
                 d=25.0, b=3.0, eta=2.0, e_max=50,
                 n_policy_updates=80, n_value_updates=80,
                 batch_size=4096):
        self.device = device
        self.env = make_safety_env(env_id)
        obs_dim = int(np.prod(self.env.observation_space.shape))
        act_dim = int(np.prod(self.env.action_space.shape))
        self.ac = ActorCritic(obs_dim, act_dim).to(device)
        self.qc = SafetyQCritic(obs_dim, act_dim).to(device)
        self.pi_optim = optim.Adam(self.ac.parameters(), lr=lr)
        self.qc_optim = optim.Adam(self.qc.parameters(), lr=lr)
        self.ppo_loss = PPOLoss(clip_ratio=clip_ratio)
        self.buf = RolloutBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam, device)
        self.gamma, self.lam = gamma, lam
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_pi_up = n_policy_updates
        self.n_qc_up = n_value_updates
        self.d, self.b, self.eta, self.e_max = d, b, eta, e_max
        self.obs_norm = RunningNorm(device=device)

    def collect_rollout(self):
        """Collect one epoch of on-policy data and return per-episode stats."""
        # --- reset buffer pointers (兼容有 reset() 或旧字段名的情况) ---
        if hasattr(self.buf, "reset"):
            self.buf.reset()
        else:
            self.buf.ptr = 0
            if hasattr(self.buf, "path_start_idx"):
                self.buf.path_start_idx = 0
            if hasattr(self.buf, "path_start"):
                self.buf.path_start = 0
    
        obs, info = self.env.reset()
        ep_ret, ep_cost, ep_len = 0.0, 0.0, 0
        train_returns, train_costs = [], []
    
        for t in range(self.steps_per_epoch):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
    
            # 如果你在 __init__ 里加了 obs_norm，就用它；否则直接用原观测
            if hasattr(self, "obs_norm"):
                self.obs_norm.update(obs_t)
                obs_in = self.obs_norm.norm(obs_t)
            else:
                obs_in = obs_t
    
            # 采样动作并交互
            act, val, logp = self.ac.act(obs_in)
            next_obs, rew, term, trunc, info = self.env.step(act.detach().cpu().numpy())
            done = float(term or trunc)
    
            cost = float(info.get("cost", 0.0))
            cum_cost = float(info.get("cum_cost", 0.0))
    
            # 注意：这里按照你当前 Buffer 的签名来 add（obs, act, rew, cost, done, logp, val, cum_cost）
            self.buf.add(obs_in, act, float(rew), cost, done, logp, val, cum_cost)
    
            ep_ret += float(rew)
            ep_cost += cost
            ep_len += 1
            obs = next_obs
    
            # 训练 episode 结束：累计到列表，并切换到下一局
            if term or trunc:
                self.buf.finish_path(last_val=0.0)
                train_returns.append(ep_ret)
                train_costs.append(ep_cost)
                ep_ret, ep_cost, ep_len = 0.0, 0.0, 0
                obs, info = self.env.reset()
    
        # epoch 截断时，用 V(s_last) 做 bootstrap，但不把“未完成的那段”当作一个完整 episode 打印/返回
        last_obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if hasattr(self, "obs_norm"):
            last_obs_t = self.obs_norm.norm(last_obs_t)
        self.buf.finish_path(last_val=self.ac.v(last_obs_t).item())
    
        # 把当轮训练中“完成的所有 episode”的 R/C 返回给脚本，脚本会逐条打印
        return {"ep_returns": train_returns, "ep_costs": train_costs}

    def update_qc(self, data, gamma):
        # Targets: c_t + gamma * E_a'[Q_c(s',a')]
        # Build next-state batch – here we shift obs by one and append last with itself (approx)
        obs = data['obs']; act = data['act']
        costs = data['cum_cost'] * 0.0  # placeholder not used here
        # We do a simple temporal shift to mimic (s_t, a_t) -> s_{t+1}
        obs_next = torch.roll(obs, shifts=-1, dims=0)
        with torch.no_grad():
            q_next = q_cost_target(obs_next, self.ac.pi, gamma, self.qc)
        # Approximate immediate cost: use differences of cum_cost (not stored per-step cost directly in buffer)
        # We didn't store per-step cost; reconstruct from cum_cost diff:
        cum = data['cum_cost']
        c_t = torch.zeros_like(cum)
        c_t[:-1] = (cum[1:] - cum[:-1]).clamp_min(0.0)
        c_t[-1] = 0.0
        target = c_t + q_next
        # MSE + small L2
        for _ in range(self.n_qc_up):
            q_pred = self.qc(obs, act)
            loss = ((q_pred - target.detach())**2).mean() + 0.1 * (q_pred**2).mean()
            self.qc_optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.qc.parameters(), 1.0)
            self.qc_optim.step()

    def update_policy(self, data, epoch):
        obs = data['obs']; act = data['act']; adv = data['adv']
        logp_old = data['logp']; ret = data['ret']
        cum_cost = data['cum_cost']
    
        # Use pure PPO for initial epochs to stabilize
        WARMUP_EPOCHS = 5
        use_safety = epoch > WARMUP_EPOCHS
    
        if use_safety:
            with torch.no_grad():
                Qc = self.qc(obs, act)
            dprime = dynamic_threshold(torch.tensor(self.d, device=obs.device), epoch, self.e_max, self.eta)
            f = safety_weight(Qc, cum_cost, dprime, self.b, clip_val=5.0)
            mod_adv = f * adv
        else:
            f = torch.ones_like(adv)
            mod_adv = adv
    
        # In case of NaN/Inf, fallback to original adv
        if not torch.isfinite(mod_adv).all():
            mod_adv = adv  # 回退
        if not torch.isfinite(ret).all():
            ret = torch.nan_to_num(ret, 0.0)
    
        ppo = self.ppo_loss
    
        # Close extra term in early epochs to stabilize training
        EXTRA_COEF = 0.1  # 额外项缩放
        if use_safety:
            with torch.no_grad():
                a_prime, _ = self.ac.pi.sample(obs)
                g_s = self.qc(obs, a_prime)
            # df/dQc = df/df_proxy * df_proxy/dQc
            Qc_req = self.qc(obs, act).detach().requires_grad_(True)
            f_proxy = safety_weight(Qc_req, cum_cost, dprime, self.b, clip_val=5.0)
            f_grad = torch.autograd.grad(f_proxy, Qc_req, grad_outputs=torch.ones_like(f_proxy), retain_graph=False, create_graph=False)[0]
            W = (ret * g_s * f_grad)
            W = torch.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            a_prime = None
            W = None
    
        for _ in range(self.n_pi_up):
            loss_main, stats = ppo.compute(self.ac, obs, act, mod_adv, logp_old, ret)
            if use_safety:
                logp_prime_now = self.ac.pi.log_prob(obs, a_prime)
                extra = - (EXTRA_COEF * W * logp_prime_now).mean()
                if not torch.isfinite(extra):
                    extra = torch.tensor(0.0, device=obs.device)
                loss = loss_main + extra
            else:
                extra = torch.tensor(0.0, device=obs.device)
                loss = loss_main
    
            self.pi_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac.parameters(), self.ppo_loss.max_grad_norm)
            self.pi_optim.step()
    
        stats.update({
            'loss/extra_term': float(extra.detach().cpu()),
            'adv/mean_mod': float(mod_adv.mean().detach().cpu()),
            'weight/mean_f': float(f.mean().detach().cpu()) if use_safety else 1.0,
            'safety/enabled': int(use_safety),
        })
        return stats

    def train(self):
        from ..utils.logging import make_logdir
        from torch.utils.tensorboard import SummaryWriter
        logdir = make_logdir()
        writer = SummaryWriter(logdir)
        global_step = 0
        for ep in range(1, self.epochs+1):
            self.collect_rollout()
            data = self.buf.get()
            # 1) Update Q_c
            self.update_qc(data, self.gamma)
            # 2) Update policy with safety mod
            stats = self.update_policy(data, ep)
            # Logging
            for k,v in stats.items():
                writer.add_scalar(k, v, global_step)
            # Simple prints
            print(f"[Epoch {ep}] "+ ", ".join(f"{k}:{v:.3f}" for k,v in stats.items() if isinstance(v, (int,float))))
            global_step += self.steps_per_epoch
        writer.close()
