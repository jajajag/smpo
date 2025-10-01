import argparse, yaml, torch, sys, numpy as np, os, time
from pathlib import Path

from smpo.algos.smpo import SMPOTrainer
from smpo.utils.seeding import seed_all

def _cast_val(s):
    # 简单类型推断：int/float/bool/None/字符串
    if isinstance(s, (int, float, bool)) or s is None:
        return s
    sv = str(s)
    if sv.lower() in ("true", "false"):
        return sv.lower() == "true"
    if sv.lower() == "none":
        return None
    try:
        if "." in sv:
            return float(sv)
        return int(sv)
    except ValueError:
        return sv

def _apply_overrides(cfg, overrides):
    """支持 --set a.b=3 c=0.1 name=foo"""
    if not overrides:
        return cfg
    for kv in overrides:
        if "=" not in kv:
            raise ValueError("Bad override (need key=value): %s" % kv)
        key, val = kv.split("=", 1)
        val = _cast_val(val)
        cur = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = val
    return cfg

def evaluate(trainer, episodes=5):
    """对当前策略做简单评测，返回 (R_mean, C_mean)。"""
    import numpy as np
    env = trainer.env  # 直接复用训练环境的封装（含 cost）
    ac = trainer.ac
    R, C = [], []
    for _ in range(int(episodes)):
        obs, info = env.reset()
        ep_r, ep_c = 0.0, 0.0
        done = False
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=trainer.device)
            if hasattr(trainer, "obs_norm"):
                obs_t = trainer.obs_norm.norm(obs_t)
            with torch.no_grad():
                a, _, _ = ac.act(obs_t)
            obs, r, term, trunc, info = env.step(a.detach().cpu().numpy())
            done = term or trunc
            ep_r += float(r)
            ep_c += float(info.get("cost", 0.0))
        R.append(ep_r); C.append(ep_c)
    r_mean = float(np.mean(R)); c_mean = float(np.mean(C))
    print(f"[Test]  episodes={episodes}  R_mean={r_mean:.2f}  C_mean={c_mean:.2f}")
    return r_mean, c_mean

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='smpo/configs/config.yaml',
                   help='通用配置文件（默认使用通用 config.yaml）')
    p.add_argument('--env_id', type=str, default=None,
                   help='覆盖配置中的 env_id，比如 SafetyButton1-v0 / SafetyCarGoal1-v0 等')
    p.add_argument('--set', nargs='*', default=[],
                   help='额外覆盖任意键：--set seed=2 steps_per_epoch=10000 lr=1e-4')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # 新增：打印 train/test 的频率控制
    p.add_argument('--eval_every', type=int, default=1, help='每多少个 epoch 做一次评测打印')
    p.add_argument('--eval_episodes', type=int, default=5, help='评测使用的 episode 数')

    args = p.parse_args()

    # 读取配置
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[ERROR] config file not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 覆盖 env_id 与其他键
    if args.env_id is not None:
        cfg["env_id"] = args.env_id
    cfg = _apply_overrides(cfg, args.set)

    seed_all(int(cfg.get("seed", 1)))

    # 构建 Trainer
    trainer = SMPOTrainer(
        env_id=cfg["env_id"],
        seed=cfg.get("seed", 1),
        steps_per_epoch=cfg.get("steps_per_epoch", 30000),
        epochs=cfg.get("epochs", 200),
        device=args.device,
        gamma=cfg.get("gamma", 0.99),
        lam=cfg.get("gae_lambda", 0.95),
        lr=cfg.get("lr", 3e-4),
        clip_ratio=cfg.get("clip_ratio", 0.2),
        d=cfg.get("d", 25.0),
        b=cfg.get("b", 3.0),
        eta=cfg.get("eta", 2.0),
        e_max=cfg.get("e_max", 50),
        n_policy_updates=cfg.get("n_policy_updates", 80),
        n_value_updates=cfg.get("n_value_updates", 80),
        batch_size=cfg.get("batch_size", 4096),
    )

    # —— 训练主循环（边训练边打印）——
    global_step = 0
    for ep in range(1, trainer.epochs + 1):
        # 收集一轮数据；如果 collect_rollout 有返回每个 episode 的统计，逐条打印
        roll_stats = None
        try:
            roll_stats = trainer.collect_rollout()
        except TypeError:
            # 兼容旧版本没有返回值
            trainer.collect_rollout()

        # 打印训练集 episodic R/C
        if isinstance(roll_stats, dict) and "ep_returns" in roll_stats and "ep_costs" in roll_stats:
            ep_returns = roll_stats["ep_returns"] or []
            ep_costs = roll_stats["ep_costs"] or []
            for i, (r, c) in enumerate(zip(ep_returns, ep_costs), 1):
                print(f"[Train] ep#{i:02d}  R={float(r):8.2f}  C={float(c):6.2f}")
        else:
            # 若没有返回值，也至少打印一个 epoch 的占位信息
            print(f"[Train] epoch {ep}: collected {trainer.steps_per_epoch} steps.")

        # 更新 Q_c 与 Policy
        data = trainer.buf.get()
        trainer.update_qc(data, trainer.gamma)
        stats = trainer.update_policy(data, ep)

        # 打印本 epoch 聚合信息（可选）
        msg = f"[Epoch {ep}] "
        if isinstance(stats, dict):
            kvs = []
            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    kvs.append(f"{k}:{v:.3f}")
            msg += ", ".join(kvs)
        print(msg)

        # 按频率评测并打印 Test R/C
        if args.eval_every > 0 and (ep % args.eval_every) == 0:
            evaluate(trainer, episodes=int(args.eval_episodes))

        global_step += trainer.steps_per_epoch

    # 训练结束，评测一次（可选）
    if args.eval_every == 0:
        evaluate(trainer, episodes=int(args.eval_episodes))

if __name__ == '__main__':
    main()

