import random
import numpy as np
import torch
import lightning as L


def set_seed(seed=42):
    """Set seeds for maximum reproducibility across all random number generators.

    Args:
        seed: Integer seed value for reproducibility
    """
    # Python's built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # PyTorch backend settings for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Lightning's seed setting (includes worker seeds)
    L.seed_everything(seed, workers=True)


def linear_normalize(x: torch.Tensor, lower_bound: float, upper_bound:float, lower_th:float|None=None, upper_th:float|None=None) -> torch.Tensor:
    """
    linearly map the variable x to the range of (lower_bound, upper_bound).
    :param x: 1-D tensor
    :param th: cutting edges of upper & lower limits.
    """
    if lower_th is not None:
        lower_th = lower_bound + lower_th * (upper_bound - lower_bound)
        x[x < lower_th] = lower_bound
    if upper_th is not None:
        upper_th = lower_bound + upper_th * (upper_bound - lower_bound)
        x[x > upper_th] = upper_bound

    x = (x - x.min()) / (x.max() - x.min() + 1.0e-8)
    x = lower_bound + x * (upper_bound - lower_bound)

    return x