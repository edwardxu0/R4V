import numpy as np
import torch
import torch.nn as nn

from copy import deepcopy
from dnnv.nn import parse
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Type, Union
from typing_extensions import TypedDict

from .layers import Layer
from .layers import Droppable, Linearizable, Scalable, SizePreserving
from .. import logging
from ..config import Configuration
from ..errors import NetworkParseError, R4VError
from ..utils import get_subclasses

LayerExclusions = TypedDict(
    "LayerExclusions",
    {"layer_type": Union[str, List[str]], "layer_id": Union[int, List[int]]},
)


def load_network(config: Configuration):
    logger = logging.getLogger(__name__)
    op_graph = parse(Path(config["model"])).simplify()
    if config.get("input_format", "NCHW") == "NHWC":
        op_graph = op_graph[2:]  # TODO : Double check this. Make more robust
    layer_types: List[Type[Layer]] = list(get_subclasses(Layer))
    layers: List[Layer] = []
    while True:
        layer_match = Layer.match(op_graph, layer_types=layer_types)
        if layer_match is None:
            break
        layers.insert(0, layer_match.layer)
        op_graph = layer_match.input_op_graph
    print(op_graph.output_operations)
    if len(op_graph.output_operations) > 0:
        raise NetworkParseError("Unsupported computation graph detected")
    for layer in layers:
        logger.debug(layer)
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

    def forall_layers(
        self,
        layer_type: Optional[Union[str, Type[Layer]]],
        strategy: Callable,
        excluding: LayerExclusions,
    ):
        layer_types = {t.__name__: t for t in get_subclasses(Layer)}
        if layer_type is None:
            layer_type = Layer
        if isinstance(layer_type, str):
            if layer_type not in layer_types:
                raise ValueError(f"Unknown droppable layer type: {layer_type}")
            layer_type = layer_types[layer_type]
        excluded_layer_types: List[Type[Layer]] = []
        if "layer_type" in excluding:
            exclude_layer_type = excluding["layer_type"]
            if isinstance(exclude_layer_type, str):
                exclude_layer_type = [exclude_layer_type]
            excluded_layer_types = [layer_types[t] for t in exclude_layer_type]
        excluded_layer_ids: List[int] = []
        if "layer_id" in excluding:
            exclude_layer_id = excluding["layer_id"]
            if isinstance(exclude_layer_id, int):
                excluded_layer_ids = [exclude_layer_id]
            else:
                excluded_layer_ids = exclude_layer_id
        for i, layer in enumerate(self.layers):
            if (
                not isinstance(layer, layer_type)
                or any(isinstance(layer, t) for t in excluded_layer_types)
                or i in excluded_layer_ids
            ):
                continue
            strategy(i)
        return self

    def drop_layer(
        self, layer_id: int, layer_type: Optional[Union[str, Type[Droppable]]] = None
    ):
        logger = logging.getLogger(__name__)
        logger.debug("Dropping layer: %d", layer_id)
        if layer_type is None:
            layer_type = Droppable
        if isinstance(layer_type, str):
            layer_types = {t.__name__: t for t in get_subclasses(Droppable)}
            if layer_type not in layer_types:
                raise ValueError(f"Unknown droppable layer type: {layer_type}")
            layer_type = layer_types[layer_type]
        elif not issubclass(layer_type, Droppable):
            raise ValueError(f"Layer type {layer_type.__name__} is not droppable.")
        if layer_id >= self.final_layer_index:
            raise R4VError("Cannot remove final layer.")
        layer = self.layers[layer_id]
        if not isinstance(layer, layer_type):
            raise R4VError(f"Layer {layer} is not of type {layer_type.__name__}")
        layer.drop()
        self.update_layer_shapes()
        return self

    def linearize(
        self, layer_id, layer_type: Optional[Union[str, Type[Linearizable]]] = None
    ):
        logger = logging.getLogger(__name__)
        logger.debug("Linearizing layer: %d", layer_id)
        if layer_type is None:
            layer_type = Linearizable
        if isinstance(layer_type, str):
            layer_types = {t.__name__: t for t in get_subclasses(Linearizable)}
            if layer_type not in layer_types:
                raise ValueError(f"Unknown linearizable layer type: {layer_type}")
            layer_type = layer_types[layer_type]
        elif not issubclass(layer_type, Linearizable):
            raise ValueError(f"Layer type {layer_type.__name__} is not linearizable.")
        layer = self.layers[layer_id]
        if not isinstance(layer, layer_type):
            raise R4VError(f"Layer {layer} is not of type {layer_type.__name__}")
        layer.linearize()
        self.update_layer_shapes()
        return self

    def scale_layer(
        self,
        layer_id: int,
        factor: float,
        layer_type: Optional[Union[str, Type[Scalable]]] = None,
    ):
        logger = logging.getLogger(__name__)
        logger.debug("Scaling layer (factor %f): %d", factor, layer_id)
        if layer_type is None:
            layer_type = Scalable
        if isinstance(layer_type, str):
            layer_types = {t.__name__: t for t in get_subclasses(Scalable)}
            if layer_type not in layer_types:
                raise ValueError(f"Unknown scalable layer type: {layer_type}")
            layer_type = layer_types[layer_type]
        elif not issubclass(layer_type, Scalable):
            raise ValueError(f"Layer type {layer_type.__name__} is not scalable.")
        if layer_id >= self.final_layer_index:
            raise R4VError("Cannot scale final layer.")
        layer = self.layers[layer_id]
        if not isinstance(layer, layer_type):
            raise R4VError(f"Layer {layer} is not of type {layer_type.__name__}")
        layer.scale(factor)
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
        with torch.no_grad():
            dummy_input = torch.ones(self.input_shape).to(
                next(self.parameters()).device
            )
            torch.onnx.export(self, dummy_input, path)

    def relu_loss(self, config):
        from .pytorch import Relu

        print(config)
        print(config.input_region_path)
        print(config["input_region_path"])
        exit()

        pre_relu_values = torch.zeros(x.size(0))

        orig_relu_forward = Relu.forward

        def relu_forward(self, x):
            pre_relu_values = torch.min(
                pre_relu_values, torch.abs(x).flatten().min()[0]
            )
            return orig_relu_forward(self, x)

        Relu.forward = relu_forward
        # TODO : do multiple passes of random samples
        # forward pass on center, then a random sample
        # high loss if sample has different sign then center
        # high loss if sample is far from center
        _ = self(x)
        return torch.exp(-pre_relu_values).sum()

    def relu_loss_(self):
        from .pytorch import Relu

        if len(Relu.value) == 0:
            return 0

        def weights(intermediate_loss_values):
            import os

            n = len(intermediate_loss_values)
            method = os.getenv("R4V_RELULOSS_REDUCTION", "max").lower()
            if method == "max":
                i_ = np.argmax(intermediate_loss_values)
                for i in range(n):
                    if i == i_:
                        yield 1.0
                    else:
                        yield 0.0
            elif method == "mean":
                for i in range(n):
                    yield i / n
            elif method == "sum":
                for i in range(n):
                    yield 1.0
            elif method.startswith("explicit="):
                W = [float(w) for w in method.split("=")[1].split(",")]
                assert len(W) == n
                for w in W:
                    yield w
            else:
                raise ValueError(f"Unknown reduction method: {method}.")

        loss = 0
        for w, v in zip(
            weights(Relu.intermediate_loss_values), Relu.intermediate_loss_values
        ):
            loss = loss + w * v
        Relu.value = []

        return loss
