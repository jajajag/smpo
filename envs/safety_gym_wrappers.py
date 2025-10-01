import safety_gymnasium
import gymnasium as gym
import numpy as np

class SafetyGymWrap(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.cum_cost = 0.0

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}
        self.cum_cost = 0.0
        return obs, info

    def step(self, action):
        out = self.env.step(action)
        if isinstance(out, tuple) and len(out) == 5:
            obs, rew, term, trunc, info = out
            cost = float(info.get('cost', info.get('costs', 0.0)))
        elif isinstance(out, tuple) and len(out) == 6:
            obs, rew, cost, term, trunc, info = out
            cost = float(cost)
        else:
            raise ValueError(f"Unsupported env.step() return format: len={len(out)}")

        self.cum_cost += cost
        info = dict(info)
        info['cost'] = cost
        info['cum_cost'] = self.cum_cost
        return obs, rew, term, trunc, info


def make_safety_env(env_id: str, nonneg_reward=True, **kwargs):
    env = safety_gymnasium.make(env_id, **kwargs)
    env = SafetyGymWrap(env)
    if nonneg_reward:
        from .reward_shaping import NonNegativeReward
        env = NonNegativeReward(env)
    return env
