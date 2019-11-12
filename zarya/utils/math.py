import torch
from torch.autograd import Function


class Atanh(Function):
    eps = 1e-5

    @staticmethod
    def forward(ctx, x):
        x = torch.clamp(x, min=-1 + Atanh.eps, max=1 - Atanh.eps)
        ctx.save_for_backward(x)

        return 0.5 * torch.log((1 + x) / (1 - x))

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output / (1 - (x ** 2)) if ctx.needs_input_grad[0] else None


class Asinh(Function):
    eps = 1e-5

    @staticmethod
    def forward(ctx, x):
        _x = torch.sqrt(x * x + 1)

        ctx.save_for_backward(_x)
        return torch.log(torch.clamp(_x + x, min=Asinh.eps))

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output / x if ctx.needs_input_grad[0] else None


class Acosh(Function):
    eps = 1e-5

    @staticmethod
    def forward(ctx, x):
        _x = torch.sqrt(torch.clamp(x * x - 1, min=Acosh.eps))
        ctx.save_for_backward(_x)
        return torch.log(torch.clamp(x + _x, min=Acosh.eps))

    @staticmethod
    def backward(ctx, grad_output):
        (_x,) = ctx.saved_tensors
        return grad_output / _x if ctx.needs_input_grad[0] else None
