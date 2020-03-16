import torch.nn as nn

from dnnv.nn.graph import OperationGraph
from dnnv.nn.operations import Identity as IdentityOp
from dnnv.nn.operations import Relu as ReluOp

from .base import SizePreserving
from ..pytorch import Relu as PytorchRelu


class Activation(SizePreserving):
    OP_PATTERN = NotImplemented

    @classmethod
    def from_operation_graph(cls, op_graph: OperationGraph):
        input_shape = op_graph.input_shape
        assert len(input_shape) == 1
        return cls(input_shape[0], input_shape[0])


class Identity(Activation):
    OP_PATTERN = IdentityOp

    def __repr__(self):
        return "Identity()"

    def as_pytorch(self, maintain_weights: bool = False) -> nn.Module:
        return nn.Sequential()


class Relu(Activation):
    OP_PATTERN = ReluOp

    def __repr__(self):
        return "Relu()"

    def as_pytorch(self, maintain_weights: bool = False) -> nn.Module:
        return PytorchRelu()


__all__ = ["Identity", "Relu"]
