import numpy as np
import torch


def set_seed(seed_value, pytorch=True):
    """
    Set seed for deterministic behavior

    Parameters
    ----------
    seed_value : int
        Seed value.
    pytorch : bool
        Whether the torch seed should also be set. The default is True.

    Returns
    -------
    None.
    """
    import random
    random.seed(seed_value)
    np.random.seed(seed_value)
    if pytorch:
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
