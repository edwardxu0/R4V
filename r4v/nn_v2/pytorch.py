import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import logging


class PytorchReshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.contiguous().view(*self.shape)


class PytorchFlatten(nn.Module):
    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        new_shape = (
            (1, -1) if self.axis == 0 else (int(np.prod(x.shape[: self.axis])), -1)
        )
        return x.reshape(*new_shape)


class PytorchTranspose(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class PytorchAtan(nn.Module):
    def forward(self, x):
        return torch.atan(x)


class PytorchMultiply(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.register_buffer("value", torch.from_numpy(value))

    def forward(self, x):
        return x * self.value
