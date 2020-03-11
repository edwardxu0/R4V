import numpy as np
import onnx

from onnx import numpy_helper

ONNX_TO_NUMPY_DTYPE = {
    onnx.TensorProto.DOUBLE: np.dtype("float64"),
    onnx.TensorProto.FLOAT16: np.dtype("float16"),
    onnx.TensorProto.FLOAT: np.dtype("float32"),
    onnx.TensorProto.INT16: np.dtype("int16"),
    onnx.TensorProto.INT32: np.dtype("int32"),
    onnx.TensorProto.INT64: np.dtype("int64"),
}


def as_numpy(node):
    if isinstance(node, onnx.TensorProto):
        return numpy_helper.to_array(node)
    elif isinstance(node, onnx.NodeProto):
        return numpy_helper.to_array(node.attribute[0].t)
    elif isinstance(node, onnx.AttributeProto):
        if node.type == onnx.AttributeProto.FLOAT:
            return np.float(node.f)
        elif node.type == onnx.AttributeProto.INT:
            return np.int(node.i)
        elif node.type == onnx.AttributeProto.INTS:
            return np.asarray(node.ints)
        elif node.type == onnx.AttributeProto.STRING:
            return node.s.decode("utf-8")
        raise ValueError("Unknown attribute type: %s" % (node,))
    else:
        raise ValueError("Unknown node type: %s" % type(node))


def fix_padding(pads):
    # return padding as left, right, top, bottom
    logger = logging.getLogger(__name__)
    if len(pads) == 2:
        pads = (int(pads[0]), int(pads[0]), int(pads[1]), int(pads[1]))
    elif len(pads) == 4:
        pads = (int(pads[0]), int(pads[2]), int(pads[1]), int(pads[3]))
    elif len(pads) == 8:
        assert pads[0] == 0 and pads[1] == 0
        assert pads[4] == 0 and pads[5] == 0
        pads = (int(pads[2]), int(pads[6]), int(pads[3]), int(pads[7]))
    else:
        raise AssertionError(
            "Unsupported length for padding values (%s): %s" % (pads, len(pads))
        )
    return pads


def get_tf_pads(in_height, in_width, kernel_shape, strides):
    out_height = np.ceil(float(in_height) / float(strides[0]))
    out_width = np.ceil(float(in_width) / float(strides[1]))
    pad_along_height = max(
        (out_height - 1) * strides[0] + kernel_shape[0] - in_height, 0
    )
    pad_along_width = max((out_width - 1) * strides[1] + kernel_shape[1] - in_width, 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    return int(pad_top), int(pad_bottom), int(pad_left), int(pad_right)


def as_implicit_padding(pads):
    if tuple(pads) == (0, 0, 0, 0):
        return "VALID"
    else:
        return "SAME"


def get_conv_parameters(conv_op, pad_op=None):
    logger = logging.getLogger(__name__)

    conv_weight = as_numpy(conv_op[2])
    if len(conv_op) > 3:
        conv_bias = as_numpy(conv_op[3])
    else:
        conv_bias = np.zeros((conv_weight.shape[0],), dtype=np.float32)

    conv_attributes = {a.name: as_numpy(a) for a in conv_op[0].attribute}
    if "auto_pad" in conv_attributes:
        logger.warning("auto_pad is deprecated: (%s)", conv_attributes["auto_pad"])
    if "dilations" in conv_attributes:
        assert np.all(
            conv_attributes["dilations"] == 1
        ), "Only 1-dilation convolutions are currently supported."
    if "group" in conv_attributes:
        assert conv_attributes["group"] == 1

    kernel_shape = tuple(conv_attributes.get("kernel_shape", conv_weight.shape[2:]))
    assert kernel_shape == tuple(conv_weight.shape[2:])
    strides = tuple(conv_attributes.get("strides", (1, 1)))
    pads = fix_padding(tuple(conv_attributes.get("pads", (0, 0, 0, 0))))
    if pad_op is None:
        return conv_weight, conv_bias, kernel_shape, strides, pads
    assert pads == (0, 0, 0, 0)
    padding_attributes = {a.name: as_numpy(a) for a in pad_op[0].attribute}
    pads = fix_padding(tuple(padding_attributes["pads"]))
    assert padding_attributes.get("mode", "constant") == "constant"
    assert padding_attributes.get("value", 0.0) == 0.0

    return conv_weight, conv_bias, kernel_shape, strides, pads


def get_batchnorm_parameters(bn_op):
    attributes = {a.name: as_numpy(a) for a in bn_op[0].attribute}
    assert "epsilon" in attributes
    assert "momentum" in attributes
    scale = as_numpy(bn_op[2])
    bias = as_numpy(bn_op[3])
    mean = as_numpy(bn_op[4])
    var = as_numpy(bn_op[5])
    return attributes, scale, bias, mean, var
