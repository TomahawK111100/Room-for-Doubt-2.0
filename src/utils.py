import os
import random
import numpy as np
import torch
import pytorch_lightning as pl

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed, workers=True)

import torchvision
torchvision.datasets.CIFAR10._check_integrity = lambda self: True
torchvision.datasets.CIFAR100._check_integrity = lambda self: True
