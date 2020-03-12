import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, Iterable, List, Optional

from .. import logging


class PytorchReshape(nn.Module):
    def __init__(self, *shape: int):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(*self.shape)


class PytorchFlatten(nn.Module):
    def __init__(self, axis: int):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        new_shape = (
            (1, -1) if self.axis == 0 else (int(np.prod(x.shape[: self.axis])), -1)
        )
        return x.reshape(*new_shape)


class PytorchTranspose(nn.Module):
    def __init__(self, *dims: int):
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


class PytorchMultipath(nn.Module):
    def __init__(self, *paths: List[nn.Module], agg: str = None):
        super().__init__()
        self.agg = agg
        self.paths = [nn.ModuleList(path) for path in paths]
        for path_idx, path in enumerate(self.paths):
            self.add_module(str(path_idx), path)

    def forward(self, x):
        ys = []
        for path in self.paths:
            y = x
            for operation in path:
                y = operation(y)
            ys.append(y)
        if self.agg is None:
            return ys
        elif self.agg == "sum":
            return sum(ys)
        raise ValueError(f"Unsupported aggregation operation: {self.agg}")

