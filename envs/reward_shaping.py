class NonNegativeReward:
    """Shift rewards so that min reward >= 0 at runtime (per-episode baseline)."""
    def __init__(self, env, shift=0.0):
        self.env = env
        self.shift = shift
        self._ep_min = 0.0

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}
        self._ep_min = 0.0
        return obs, info

    def step(self, action):
        out = self.env.step(action)
        if isinstance(out, tuple) and len(out) == 5:
            obs, rew, term, trunc, info = out
        elif isinstance(out, tuple) and len(out) == 6:
            obs, rew, _cost, term, trunc, info = out
        else:
            raise ValueError(f"Unsupported wrapped env.step() return: len={len(out)}")

        self._ep_min = min(self._ep_min, rew)
        shift = -self._ep_min if self._ep_min < 0 else 0.0
        return obs, rew + shift, term, trunc, info

    def __getattr__(self, name):
        return getattr(self.env, name)
