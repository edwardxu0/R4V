import numpy as np
import torch
import torch.nn as nn

from dnnv.nn.operations import MaxPool as MaxPoolOp
from dnnv.nn.operations import Operation
from typing import Tuple, Union

from .base import Droppable
from .utils import single
from ..pytorch import Sequential


class MaxPool(Droppable):
    OP_PATTERN = MaxPoolOp

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        kernel_shape: Tuple[int, int],
        strides: Union[int, Tuple[int, int]],
        padding: Tuple[int, ...],
    ):
        super().__init__(input_shape, output_shape)
        self.kernel_shape = kernel_shape
        self.strides = strides
        self.padding = padding

    def __repr__(self):
        return (
            f"MaxPool({self.kernel_shape}, "
            f"strides={self.strides}, "
            f"padding={self.padding})"
        )

    @classmethod
    def from_operation_graph(cls, op_graph):
        input_shape = op_graph.input_shape
        assert len(input_shape) == 1
        output_shape = op_graph.output_shape
        assert len(output_shape) == 1

        op: MaxPoolOp = single(op_graph.output_operations)
        assert not op.ceil_mode
        assert op.dilations == 1
        assert op.storage_order == MaxPoolOp.ROW_MAJOR_STORAGE
        kernel_shape = tuple(op.kernel_shape)
        pads = tuple(op.pads)
        strides = tuple(op.strides)

        return cls(input_shape[0], output_shape[0], kernel_shape, strides, pads)

    def as_pytorch(self, maintain_weights: bool = False) -> nn.Module:
        pad_top, pad_left, pad_bottom, pad_right = self.padding

        pad_layer = nn.ZeroPad2d((pad_left, pad_right, pad_top, pad_bottom))
        pool_layer = nn.MaxPool2d(self.kernel_shape, self.strides)
        return Sequential(pad_layer, pool_layer)


__all__ = ["MaxPool"]
