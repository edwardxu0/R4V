import numpy as np
import torch
import torch.nn as nn

from abc import abstractmethod
from dnnv.nn.graph import OperationGraph
from dnnv.nn.operations import Operation, OperationPattern
from dnnv.nn.transformers import DropPrefix
from dnnv.nn.visitors import OperationCounter
from typing import Optional, Sequence, Tuple, Type, Union

from ...utils import get_subclasses


class _Layer(type):
    def __new__(self, name, bases, namespace, **kwargs):
        if name == "LayerBase":
            return super().__new__(self, name, bases, namespace, **kwargs)
        if "OP_PATTERN" not in namespace:
            raise TypeError(f"Layer {name} must specify `OP_PATTERN`")
        op_pattern = namespace["OP_PATTERN"]
        if op_pattern is NotImplemented:
            namespace["OP_PATTERN"] = None
        if (
            op_pattern is not None
            and op_pattern is not NotImplemented
            and not isinstance(op_pattern, OperationPattern)
            and (
                not isinstance(op_pattern, type)
                or not issubclass(op_pattern, Operation)
            )
        ):
            raise TypeError("`OP_PATTERN` must be an operation pattern")
        return super().__new__(self, name, bases, namespace, **kwargs)

    @property
    def OP_PATTERN(self) -> Union[Type[Operation], OperationPattern, None]:
        return self.__dict__["OP_PATTERN"]


class LayerMatch:
    def __init__(self, layer, input_op_graph):
        self.layer = layer
        self.input_op_graph = input_op_graph


class LayerBase(metaclass=_Layer):
    @classmethod
    @abstractmethod
    def from_operation_graph(cls, operation_graph: OperationGraph) -> _Layer:
        raise NotImplementedError()

    @classmethod
    def match(
        cls: Type["LayerBase"],
        operation_graph: OperationGraph,
        layer_types: Optional[Sequence[Type["LayerBase"]]] = None,
    ) -> Optional[LayerMatch]:
        if layer_types is None:
            layer_types = list(get_subclasses(cls))

        best_match: Optional[Sequence[Operation]] = None
        best_op_count = float("inf")
        best_layer_type = LayerBase
        assert layer_types is not None
        for layer_type in layer_types:
            if layer_type.OP_PATTERN is None:
                continue
            matches = layer_type.OP_PATTERN.match(operation_graph.output_operations)
            for match in matches:
                op_count = 0
                visitor = OperationCounter()
                for op in match:
                    op_count = visitor.visit(op)
                if op_count < best_op_count:
                    best_match = match
                    best_op_count = op_count
                    best_layer_type = layer_type
        if best_match is None:
            return None
        input_op_graph = OperationGraph(best_match)
        op_graph = OperationGraph(operation_graph.walk(DropPrefix(input_op_graph)))
        return LayerMatch(
            best_layer_type.from_operation_graph(op_graph), input_op_graph
        )


class Layer(LayerBase):
    OP_PATTERN = NotImplemented

    def __init__(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.dropped = False
        self.modified = False

    def update_shape(self, input_shape):
        self.input_shape = input_shape
        dummy_input = torch.ones(self.input_shape)
        self.output_shape = tuple(self.as_pytorch()(dummy_input).shape)

    @abstractmethod
    def as_pytorch(self, maintain_weights: bool = False) -> nn.Module:
        raise NotImplementedError()

    def num_neurons(self):
        if self.dropped:
            return 0
        return np.product(self.output_shape)

    def num_parameters(self):
        if self.dropped:
            return 0
        params = self.as_pytorch(False).parameters()
        return sum(np.product(p.shape) for p in params)


class Droppable(Layer):
    OP_PATTERN = NotImplemented

    def drop(self):
        self.dropped = True
        self.modified = True


class Scalable(Layer):
    OP_PATTERN = NotImplemented

    def scale(self, factor: float, attribute=None):
        self.modified = True


class SizePreserving(Layer):
    OP_PATTERN = NotImplemented


class DroppableOperations(Layer):
    OP_PATTERN = NotImplemented

    @abstractmethod
    def drop_operation(self, op_type):
        raise NotImplementedError()


class Linearizable(Layer):
    OP_PATTERN = NotImplemented

    @abstractmethod
    def linearize(self):
        self.modified = True


__all__ = ["Layer", "Droppable", "Linearizable", "Scalable", "SizePreserving"]
