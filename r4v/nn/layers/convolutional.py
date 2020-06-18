import numpy as np
import torch
import torch.nn as nn

from dnnv.nn.graph import OperationGraph
from dnnv.nn.operations import Conv, Relu
from typing import Optional, Tuple

from .base import Droppable, Scalable
from .utils import single
from ..pytorch import Relu as PytorchRelu, Sequential
from ...errors import R4VError


class Convolutional(Droppable, Scalable):
    OP_PATTERN = Conv >> (Relu | None)

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        weights: np.ndarray,
        bias: np.ndarray,
        kernel_shape: Tuple[int, int],
        strides: Tuple[int, int],
        padding: Tuple[int, int, int, int],
        activation: Optional[str],
    ):
        super().__init__(input_shape, output_shape)
        self.weights = weights
        self.bias = bias
        self.kernel_shape = kernel_shape
        self.strides = strides
        self.padding = padding
        self.activation = activation

    def __repr__(self):
        return (
            f"Convolutional({self.input_shape[1]}, "
            f"{self.output_shape[1]}, "
            f"kernel_shape={self.kernel_shape}, "
            f"strides={self.strides}, "
            f"padding={self.padding}, "
            f"activation={self.activation})"
        )

    @classmethod
    def from_operation_graph(cls, op_graph: OperationGraph):
        input_shape = op_graph.input_shape
        assert len(input_shape) == 1
        output_shape = op_graph.output_shape
        assert len(output_shape) == 1

        op = single(op_graph.output_operations)
        activation = None
        if isinstance(op, Relu):
            activation = "relu"
            op = op.x

        if not isinstance(op, Conv):
            raise ValueError(
                f"Unexpected operation in convolutional layer: {type(op).__name__}"
            )
        weights = op.w
        bias = op.b
        if bias is None:
            bias = np.zeros((weights.shape[0],), dtype=weights.dtype)

        assert np.all(op.dilations == 1)
        assert np.all(op.group == 1)

        kernel_shape = op.kernel_shape
        if kernel_shape is None:
            kernel_shape = weights.shape[2:4]
        pads = op.pads
        strides = op.strides

        return cls(
            input_shape[0],
            output_shape[0],
            weights,
            bias,
            kernel_shape,
            strides,
            pads,
            activation,
        )

    def scale(self, factor: float, attribute=None):
        assert attribute is None
        super().scale(factor)
        new_size = int(self.output_shape[1] * factor)
        if new_size < 1:
            raise R4VError("Cannot scale convolutional layers to 0 neurons")
        self.output_shape = (
            self.output_shape[0],
            new_size,
            self.output_shape[2],
            self.output_shape[3],
        )

    def as_pytorch(self, maintain_weights: bool = False) -> nn.Module:
        in_c = self.input_shape[1]
        out_c = self.output_shape[1]
        pad_top, pad_left, pad_bottom, pad_right = self.padding

        pad_layer = nn.ZeroPad2d((pad_left, pad_right, pad_top, pad_bottom))
        conv_layer = nn.Conv2d(in_c, out_c, self.kernel_shape, self.strides)
        if maintain_weights and self.modified:
            raise ValueError("Cannot maintain weights of modified layer.")
        elif maintain_weights:
            conv_layer.weight.data = torch.from_numpy(self.weights)
            conv_layer.bias.data = torch.from_numpy(self.bias)

        if self.activation == "relu":
            return Sequential(pad_layer, conv_layer, PytorchRelu())
        elif self.activation is not None:
            raise ValueError(
                f"Unsupported activation for convolutional layers: {self.activation}"
            )
        return Sequential(pad_layer, conv_layer)


__all__ = ["Convolutional"]
