import numpy as np
import random
import torch

def set_random_seed(seed):
    """
    Set random seed for whole program

    :param seed: random seed
    :type seed: int
    """
    assert isinstance(seed, int)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def seed_worker(worker_id):
    """
    Set seed for dataloader worker for reproducibility.
    See: https://pytorch.org/docs/stable/notes/randomness.html 

    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
