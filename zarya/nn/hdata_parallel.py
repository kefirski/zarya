import torch.nn as nn

from .hmodule import HModule


class HDataParallel(HModule, nn.DataParallel):
    pass
