import torch.nn as nn


class Parameter(nn.Parameter):
    r"""zarya.nn.Parameters are supposed to be separated from usuall torch.nn.Parameters"""
    pass
