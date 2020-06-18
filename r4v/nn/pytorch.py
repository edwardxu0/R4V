import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, Iterable, List, Optional

from .. import logging


class Relu(nn.ModuleList):
    def forward(self, x):
        return F.relu(x)


class Reshape(nn.Module):
    def __init__(self, *shape: int):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(*self.shape)


class Flatten(nn.Module):
    def __init__(self, axis: int):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        new_shape = (
            (1, -1) if self.axis == 0 else (int(np.prod(x.shape[: self.axis])), -1)
        )
        return x.reshape(*new_shape)


class Transpose(nn.Module):
    def __init__(self, *dims: int):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class Atan(nn.Module):
    def forward(self, x):
        return torch.atan(x)


class Multiply(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.register_buffer("value", torch.from_numpy(value))

    def forward(self, x):
        return x * self.value


class MultiPath(nn.Module):
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


class Sequential(nn.Module):
    def __init__(self, *path: nn.Module):
        super().__init__()
        self.path = nn.ModuleList(path)

    def forward(self, x):
        y = x
        for operation in self.path:
            y = operation(y)
        return y
