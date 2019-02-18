import torch.nn as nn

from .hmodule import HModule


class HDataParallel(nn.DataParallel, HModule):
    pass
