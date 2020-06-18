import torch.nn as nn

from dnnv.nn.graph import OperationGraph
from dnnv.nn.operations import Atan as AtanOp
from dnnv.nn.operations import LogSoftmax as LogSoftmaxOp
from dnnv.nn.operations import Mul as MulOp
from dnnv.nn.operations import Operation
from numbers import Number
from typing import Tuple

from .base import Layer, SizePreserving
from .utils import single
from ..pytorch import Atan as PytorchAtan, Multiply as PytorchMultiply


class Atan(SizePreserving):
    OP_PATTERN = AtanOp

    def __repr__(self):
        return "Atan()"

    @classmethod
    def from_operation_graph(cls, op_graph: OperationGraph):
        input_shape = op_graph.input_shape
        assert len(input_shape) == 1
        return cls(input_shape[0], input_shape[0])

    def as_pytorch(self, maintain_weights: bool = False) -> nn.Module:
        return PytorchAtan()


class LogSoftmax(SizePreserving):
    OP_PATTERN = LogSoftmaxOp

    def __init__(
        self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...], axis: int = 1
    ):
        super().__init__(input_shape, output_shape)
        self.axis = axis

    def __repr__(self):
        return f"LogSoftmax(axis={self.axis})"

    @classmethod
    def from_operation_graph(cls, op_graph: OperationGraph):
        input_shape = op_graph.input_shape
        assert len(input_shape) == 1

        op: LogSoftmaxOp = single(op_graph.output_operations)
        axis = op.axis

        return cls(input_shape[0], input_shape[0], axis)

    def as_pytorch(self, maintain_weights: bool = False) -> nn.Module:
        return nn.LogSoftmax(dim=self.axis)


class Mul(SizePreserving):
    OP_PATTERN = MulOp

    def __init__(
        self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...], const: Number
    ):
        super().__init__(input_shape, output_shape)
        self.const = const

    def __repr__(self):
        return f"Multiply({self.const})"

    @classmethod
    def from_operation_graph(cls, op_graph: OperationGraph):
        input_shape = op_graph.input_shape
        assert len(input_shape) == 1

        op: MulOp = single(op_graph.output_operations)
        a = op.a
        b = op.b
        if isinstance(a, Operation) and isinstance(b, Operation):
            raise ValueError("At least one input to Mul must be concrete.")
        elif not isinstance(a, Operation) and not isinstance(b, Operation):
            raise ValueError("At most one input to Mul may be concrete.")
        elif isinstance(a, Operation):
            c = b
        else:
            c = a
        return cls(input_shape[0], input_shape[0], c)

    def as_pytorch(self, maintain_weights: bool = False) -> nn.Module:
        return PytorchMultiply(self.const)


__all__ = ["Atan", "LogSoftmax", "Mul"]
