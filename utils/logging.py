import time, json, os

def log_scalar(writer, tag, val, step):
    if writer is not None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer.add_scalar(tag, val, step)
        except Exception:
            pass

def make_logdir(root="runs"):
    t = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(root, t)
    os.makedirs(path, exist_ok=True)
    return path
