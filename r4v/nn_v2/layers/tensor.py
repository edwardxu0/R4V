import torch.nn as nn

from dnnv.nn.graph import OperationGraph
from dnnv.nn.operations import Flatten as FlattenOp
from dnnv.nn.operations import Input as InputOp
from dnnv.nn.operations import Operation
from dnnv.nn.operations import Transpose as TransposeOp
from typing import Optional, Tuple

from .base import Layer
from .utils import single
from ..pytorch import PytorchFlatten, PytorchTranspose


class Flatten(Layer):
    OP_PATTERN = FlattenOp

    def __init__(
        self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...], axis: int
    ):
        super().__init__(input_shape, output_shape)
        self.axis = axis

    def __repr__(self):
        return f"Flatten({self.axis})"

    @classmethod
    def from_operation_graph(cls, op_graph: OperationGraph):
        input_shape = op_graph.input_shape
        assert len(input_shape) == 1
        output_shape = op_graph.output_shape
        assert len(output_shape) == 1

        op: FlattenOp = single(op_graph.output_operations)
        axis = op.axis
        if isinstance(axis, Operation):
            raise ValueError("Axis for Flatten must be concrete")

        return cls(input_shape[0], output_shape[0], axis)

    def as_pytorch(self, maintain_weights: bool = False) -> nn.Module:
        return PytorchFlatten(self.axis)


class Input(Layer):
    OP_PATTERN = InputOp

    def __repr__(self):
        return f"Input({self.output_shape})"

    @classmethod
    def from_operation_graph(cls, op_graph: OperationGraph):
        input_shape = op_graph.input_shape
        assert len(input_shape) == 1
        output_shape = op_graph.output_shape
        assert len(output_shape) == 1
        return cls(input_shape[0], output_shape[0])


class Transpose(Layer):
    OP_PATTERN = TransposeOp

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        permutation: Tuple[int, ...],
    ):
        super().__init__(input_shape, output_shape)
        self.permutation = permutation

    def __repr__(self):
        perm_s = ", ".join(str(d) for d in self.permutation)
        return f"Transpose({perm_s})"

    @classmethod
    def from_operation_graph(cls, op_graph: OperationGraph):
        input_shape = op_graph.input_shape
        assert len(input_shape) == 1
        output_shape = op_graph.output_shape
        assert len(output_shape) == 1

        op: TransposeOp = single(op_graph.output_operations)
        perm = op.permutation
        if isinstance(perm, Operation):
            raise ValueError("Permutation for Transpose must be concrete")

        return cls(input_shape[0], output_shape[0], tuple(perm))

    def as_pytorch(self, maintain_weights: bool = False) -> nn.Module:
        return PytorchTranspose(self.permutation)


__all__ = ["Flatten", "Input", "Transpose"]
