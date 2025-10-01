import torch

def safety_weight(Qc, cum_cost, d_prime, b=3.0, clip_val=5.0):
    if not torch.is_tensor(d_prime):
        d_prime = torch.tensor(d_prime, device=Qc.device, dtype=Qc.dtype)

    Qc = torch.clamp(Qc, 0.0, d_prime)
    logb = torch.log(torch.tensor(b, device=Qc.device, dtype=Qc.dtype))

    x = torch.clamp(cum_cost + Qc - d_prime, min=-100.0, max=100.0)
    bd = torch.exp(logb * d_prime)
    coef = bd / (bd - 1.0 + 1e-8)
    exp_term = torch.exp(logb * x)
    f = coef * (1.0 - exp_term)

    f = torch.clamp(f, -clip_val, clip_val)
    f = torch.nan_to_num(f, nan=0.0, posinf=clip_val, neginf=-clip_val)
    return f
