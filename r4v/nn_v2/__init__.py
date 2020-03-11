import numpy as np
import torch
import torch.nn as nn

from copy import deepcopy
from dnnv.nn import parse
from pathlib import Path
from typing import List, Optional, Sequence, Type, Union

from .layers import Layer
from .layers import Droppable, Scalable, SizePreserving
from .. import logging
from ..config import Configuration
from ..errors import NetworkParseError, R4VError
from ..utils import get_subclasses


def load_network(config: Configuration):
    op_graph = parse(Path(config["model"])).simplify()
    if config["input_format"] == "NHWC":
        op_graph = op_graph[2:]  # TODO : Double check this. Make more robust
    layer_types: List[Type[Layer]] = list(get_subclasses(Layer))
    layers: List[Layer] = []
    while True:
        layer_match = Layer.match(op_graph, layer_types=layer_types)
        if layer_match is None:
            break
        layers.insert(0, layer_match.layer)
        op_graph = layer_match.input_op_graph
    if len(op_graph.output_operations) > 0:
        raise NetworkParseError("Unsupported computation graph detected")
    return DNN(layers)


class DNN:
    def __init__(self, layers: Sequence[Layer]):
        self._layers = layers
        self.input_layer = layers[0]
        self.layers = layers[1:]
        self.final_layer_index = 0
        for i, layer in enumerate(self.layers):
            if not isinstance(layer, SizePreserving):
                self.final_layer_index = i

    def clone(self):
        return DNN(deepcopy(self._layers))

    def num_neurons(self, *args):
        num_neurons = 0
        for layer in self.layers:
            num_neurons += layer.num_neurons()
        return num_neurons

    @property
    def num_parameters(self):
        num_parameters = 0
        for layer in self.layers:
            num_parameters += layer.num_parameters()
        return num_parameters

    def as_pytorch(self, maintain_weights: Optional[bool] = None):
        if any(layer.modified for layer in self.layers) and maintain_weights:
            raise ValueError("Cannot maintain weights. Network has been modified.")
        if maintain_weights is None:
            maintain_weights = not any(layer.modified for layer in self.layers)
        layers = []
        for layer in self.layers:
            if not layer.dropped:
                layers.append(layer.as_pytorch(maintain_weights=maintain_weights))
        return Net(layers, self.input_layer.output_shape)

    def update_layer_shapes(self):
        input_shape = self.input_layer.output_shape
        for layer in self.layers:
            if layer.dropped:
                continue
            layer.update_shape(input_shape)
            input_shape = layer.output_shape
        return self

    def drop_layer(
        self, layer_id: int, layer_type: Optional[Union[str, Type[Droppable]]] = None
    ):
        logger = logging.getLogger(__name__)
        logger.debug("Dropping layer: %d", layer_id)
        if layer_type is None:
            layer_type = Droppable
        elif isinstance(layer_type, str):
            layer_types = {t.__name__: t for t in get_subclasses(Droppable)}
            if layer_type not in layer_types:
                raise ValueError("Unknown layer type: %s" % layer_type)
            layer_type = layer_types[layer_type]
        if layer_id >= self.final_layer_index:
            raise R4VError("Cannot remove final layer.")
        layer = self.layers[layer_id]
        if not isinstance(layer, layer_type):
            raise R4VError(f"Layer {layer} is not of type {layer_type.__name__}")
        layer.drop()
        self.update_layer_shapes()
        return self


class Net(nn.Module):
    def __init__(self, layers, input_shape):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.input_shape = input_shape
        self.cache = {True: {}, False: {}}

    def forward(self, x, *, cache_ids=None, validation=False):
        if cache_ids is not None and all(
            [cache_id.item() in self.cache[validation] for cache_id in cache_ids]
        ):
            return torch.stack(
                [self.cache[validation][cache_id.item()] for cache_id in cache_ids]
            ).to(next(self.parameters()).device)
        y = x
        for layer in self.layers:
            y = layer(y)
        if cache_ids is not None:
            y_ = y.cpu().detach()
            for i, cache_id in enumerate(cache_ids):
                self.cache[validation][cache_id.item()] = y_[i]
        return y

    def export_onnx(self, path):
        dummy_input = torch.ones(self.input_shape).to(next(self.parameters()).device)
        torch.onnx.export(self, dummy_input, path)
