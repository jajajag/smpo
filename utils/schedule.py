import torch

def dynamic_threshold(d, epoch, e_max=50, eta=2.0):
    """d' = (eta - (eta-1) * min(e_max, epoch)/e_max) * d"""
    ratio = min(e_max, epoch) / float(e_max)
    dprime = (eta - (eta - 1.0) * ratio) * d
    return dprime
