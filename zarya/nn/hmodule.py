from collections import OrderedDict

import torch

from .hyperbolic_parameter import HParameter


class HModule(torch.nn.Module):
    def __init__(self):
        super(HModule, self).__init__()
        self._hparameters = OrderedDict()

    def register_hparameter(self, name, param: HParameter):
        if "_hparameters" not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before HModule.__init__() call"
            )

        elif not isinstance(name, torch._six.string_classes):
            raise TypeError(
                "parameter name should be a string. "
                "Got {}".format(torch.typename(name))
            )
        elif "." in name:
            raise KeyError('parameter name can\'t contain "."')
        elif name == "":
            raise KeyError('parameter name can\'t be empty string ""')
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError("attribute '{}' already exists".format(name))

        if param is None:
            self._hparameters[name] = None
        elif not isinstance(param, HParameter):
            raise TypeError(
                "cannot assign '{}' object to parameter '{}' "
                "(zarya.nn.HParameter or None required)".format(type(param), name)
            )
        elif param.tensor.grad_fn:
            raise ValueError(
                "Cannot assign non-leaf Tensor to parameter '{0}'. Model "
                "parameters must be created explicitly. To express '{0}' "
                "as a function of another Tensor, compute the value in "
                "the forward() method.".format(name)
            )
        else:
            self._hparameters[name] = param

    def __setattr__(self, name, value):
        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]

        if isinstance(value, HParameter):
            hparams = self.__dict__.get("_parameters")
            if hparams is not None and name in hparams:
                if value is not None:
                    raise TypeError(
                        "cannot assign '{}' as parameter '{}' "
                        "(zarya.nn.HParameter or None expected)".format(
                            type(value), name
                        )
                    )
            else:
                if hparams is None:
                    raise AttributeError(
                        "cannot assign parameters before Module.__init__() call"
                    )
                remove_from(self.__dict__, self._buffers, self._modules)

            self.register_hparameter(name, value)
        else:
            super(HModule, self).__setattr__(name, value)

    def __getattr__(self, name):

        if "_hparameters" in self.__dict__:
            if name in self._hparameters:
                return self._hparameters[name]

        return super(HModule, self).__getattr__(name)

    def hparameters(self):
        res = []
        for param in self._hparameters.items():
            res += [param]

        for module in self.children():
            if isinstance(module, HModule):
                res += module.hparameters()

        return (val for val in res)

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for param in self._hparameters.values():
            if param is not None:
                param.tensor.data = fn(param.tensor.data)
                if param.tensor._grad is not None:
                    param.tensor._grad.data = fn(param.tensor._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        return self
