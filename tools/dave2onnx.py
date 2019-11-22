#!/usr/bin/env python
import argparse
import importlib.util
import numpy as np
import onnxmltools  # uses onnxmltools==1.3
import onnx
import os
import sys

from keras.models import load_model
from onnx import helper, numpy_helper


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Convert the dave network to an onnx model."
    )

    parser.add_argument(
        "dave_path",
        type=str,
        help="path to file defining dave model (keras model accessible by call to function DAVE)",
    )
    parser.add_argument(
        "output_path", type=str, help="where to write the resulting onnx model"
    )
    return parser.parse_args()


def add_atan_output(model):
    nodes = list(model.graph.node)
    nodes.append(
        helper.make_node("Atan", inputs=[nodes[-1].output[0]], outputs=["atan_output"])
    )
    nodes.append(
        helper.make_node(
            "Mul", inputs=["atan_output", "atan_mul_val"], outputs=["atan_output_2"]
        )
    )

    initializers = list(model.graph.initializer)
    initializers.append(
        numpy_helper.from_array(np.array(2.0, dtype=np.float32), name="atan_mul_val")
    )

    print("\n===================")
    for node in nodes:
        print(node.op_type, node.input, node.output)

    new_output = [
        helper.make_tensor_value_info("atan_output_2", onnx.TensorProto.FLOAT, (1, 1))
    ]
    graph = helper.make_graph(
        nodes, "dave", model.graph.input, new_output, initializers
    )
    model = helper.make_model(graph)

    return model


def main(args):
    spec = importlib.util.spec_from_file_location("model", args.dave_path)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)

    onnx_model = onnxmltools.convert_keras(model.DAVE())
    onnx_model = add_atan_output(onnx_model)

    onnxmltools.utils.save_model(onnx_model, args.output_path)


if __name__ == "__main__":
    main(_parse_args())
