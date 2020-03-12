import numpy as np
import torch
import torch.nn as nn

from dnnv.nn.graph import OperationGraph
from dnnv.nn.operations import Input, Conv, Relu, Add
from dnnv.nn.operations import Operation
from typing import Optional, Tuple

from .base import Droppable, Linearizable
from .convolutional import Convolutional
from .utils import single
from ..pytorch import PytorchMultipath


class TestResidual(Droppable, Linearizable):
    OP_PATTERN = (None & (Conv >> Relu >> Conv)) | (
        (Conv >> Relu >> Conv) & None
    ) >> Add >> (Relu | None)

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        conv1: Convolutional,
        conv2: Convolutional,
        activation: Optional[str],
    ):
        super().__init__(input_shape, output_shape)
        self.conv1 = conv1
        self.conv2 = conv2
        self.activation = activation
        self.linear = False

    def __repr__(self):
        return f"TestResidual(\n\t{self.conv1},\n\t{self.conv2},\n\tresidual=Identity,\n\tactivation{self.activation})"

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

        if not isinstance(op, Add):
            raise ValueError(
                f"Unexpected operation in residual layer: {type(op).__name__}"
            )

        if isinstance(op.a, Input):
            computation_path = op.b
        else:
            computation_path = op.a
        if not isinstance(computation_path, Conv):
            raise ValueError(
                f"Unexpected operation in residual layer: {type(computation_path).__name__}"
            )

        conv2 = Convolutional.from_operation_graph(OperationGraph([computation_path]))
        conv1 = Convolutional.from_operation_graph(OperationGraph([computation_path.x]))

        return cls(input_shape[0], output_shape[0], conv1, conv2, activation)

    def update_shape(self, input_shape):
        self.conv1.update_shape(input_shape)
        self.conv2.update_shape(self.conv1.output_shape)
        super().update_shape(input_shape)

    def linearize(self):
        self.linear = True

    def as_pytorch(self, maintain_weights: bool = False) -> nn.Module:
        if not self.linear:
            layer = PytorchMultipath(
                [
                    self.conv1.as_pytorch(maintain_weights=maintain_weights),
                    self.conv2.as_pytorch(maintain_weights=maintain_weights),
                ],
                [],
                agg="sum",
            )
        else:
            layer = PytorchMultipath(
                [self.conv1.as_pytorch(maintain_weights=maintain_weights)], agg="sum"
            )
        if self.activation == "relu":
            return nn.Sequential(layer, nn.ReLU())
        elif self.activation is not None:
            raise ValueError(
                f"Unsupported activation for convolutional layers: {self.activation}"
            )
        return layer


__all__ = ["TestResidual"]
