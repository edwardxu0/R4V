from functools import partial
from typing import List, Optional, Union


def forall(model, layer_type, strategy, excluding=None, **params):
    action = partial(getattr(model, strategy), **params)
    if excluding is None:
        excluding = {}
    model.forall_layers(layer_type, action, excluding=excluding)


def drop_layer(
    model, layer_id: Union[List[int], int], layer_type: Optional[str] = None
):
    if isinstance(layer_id, int):
        layer_id = [layer_id]
    for lid in layer_id:
        model.drop_layer(lid, layer_type=layer_type)


def scale_layer(
    model,
    layer_id: Union[List[int], int],
    factor: float,
    layer_type: Optional[str] = None,
):
    if isinstance(layer_id, int):
        layer_id = [layer_id]
    for lid in layer_id:
        model.scale_layer(lid, factor=factor, layer_type=layer_type)


def linearize(model, layer_id: Union[List[int], int], layer_type: Optional[str] = None):
    if isinstance(layer_id, int):
        layer_id = [layer_id]
    for lid in layer_id:
        model.linearize(lid, layer_type=layer_type)


# def drop_operation(model, layer_id, op_type, layer_type=None):
#     if layer_type is not None:
#         model.drop_operation(layer_id, op_type, layer_type=layer_type)
#     else:
#         model.drop_operation(layer_id, op_type)


# def scale_input(model, factor):
#     model.scale_input(factor)


# def scale_layer(model, layer_id, factor, layer_type=None):
#     if layer_type is not None:
#         model.scale_layer(layer_id, factor, layer_type=layer_type)
#     else:
#         model.scale_layer(layer_id, factor)


# def scale_convolution_stride(model, layer_id, factor, layer_type=None):
#     if layer_type is not None:
#         model.scale_convolution_stride(layer_id, factor, layer_type=layer_type)
#     else:
#         model.scale_convolution_stride(layer_id, factor)


# def replace_convolution_padding(model, layer_id, padding, layer_type=None):
#     if layer_type is not None:
#         model.replace_convolution_padding(layer_id, padding, layer_type=layer_type)
#     else:
#         model.replace_convolution_padding(layer_id, padding)
