import numpy as np
import torch
import torch.nn as nn

from dnnv.nn.operations import Add, Gemm, MatMul, Relu
from dnnv.nn.operations import Operation
from typing import Optional, Tuple

from .base import Droppable, Scalable
from .utils import single


class FullyConnected(Droppable, Scalable):
    OP_PATTERN = ((MatMul >> (Add | None)) | Gemm) >> (Relu | None)

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        weights: np.ndarray,
        bias: np.ndarray,
        activation: Optional[str],
    ):
        super().__init__(input_shape, output_shape)
        self.weights = weights
        self.bias = bias
        self.activation = activation

    def __repr__(self):
        return (
            f"FullyConnected({self.input_shape[1]}, "
            f"{self.output_shape[1]}, "
            f"activation={self.activation})"
        )

    @classmethod
    def from_operation_graph(cls, op_graph):
        input_shape = op_graph.input_shape
        assert len(input_shape) == 1
        output_shape = op_graph.output_shape
        assert len(output_shape) == 1

        op = single(op_graph.output_operations)
        activation = None
        if isinstance(op, Relu):
            activation = "relu"
            op = op.x

        if isinstance(op, Gemm):
            assert isinstance(op.a, Operation)
            assert isinstance(op.b, np.ndarray)
            assert isinstance(op.c, np.ndarray)
            assert op.alpha == op.beta == 1.0
            assert not op.transpose_a
            weights = op.b
            if op.transpose_b:
                weights = weights.T
            bias = op.c
        elif isinstance(op, Add):
            raise NotImplementedError()
        elif isinstance(op, MatMul):
            raise NotImplementedError()
        else:
            raise ValueError(
                f"Unexpected operation in fully connected layer: {type(op).__name__}"
            )

        return cls(input_shape[0], output_shape[0], weights, bias, activation)

    def as_pytorch(self, maintain_weights: bool = False) -> nn.Module:
        in_features = self.input_shape[1]
        out_features = self.output_shape[1]
        layer: nn.Module = nn.Linear(in_features, out_features)
        if maintain_weights and self.modified:
            raise ValueError("Cannot maintain weights of modified layer.")
        elif maintain_weights:
            layer.weight.data = torch.from_numpy(self.weights.T)
            layer.bias.data = torch.from_numpy(self.bias)

        if self.activation == "relu":
            layer = nn.Sequential(layer, nn.ReLU())
        elif self.activation is not None:
            raise ValueError(
                f"Unsupported activation for convolutional layers: {self.activation}"
            )
        return layer


__all__ = ["FullyConnected"]
