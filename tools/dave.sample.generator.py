#!/usr/bin/env python
import argparse
import logging

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

from pathlib import Path
from r4v.nn import load_network, FullyConnected, Convolutional

TEMPLATE = """[distillation]
maxmemory="32G"
threshold=1e-9
cuda=true
type="regression"
precompute_teacher=true

[distillation.parameters]
epochs=10
optimizer="adadelta"
rho=0.95
loss="MSE"
learning_rate=1.0

[distillation.data]
format="udacity-driving"
batchsize=256
presized=true

[distillation.data.transform]
bgr=true
mean=[103.939, 116.779, 123.68]
max_value=255.0

[distillation.data.train]
shuffle=true

[distillation.data.train.teacher]
path="artifacts/udacity.sdc.100/training"

[distillation.data.train.student]
path="artifacts/udacity.sdc.100/training"

[distillation.data.validation]
shuffle=false

[distillation.data.validation.teacher]
path="artifacts/udacity.sdc.100/validation"

[distillation.data.validation.student]
path="artifacts/udacity.sdc.100/validation"

[distillation.teacher]
framework="onnx"
input_shape=[1, 100, 100, 3]
input_format="NHWC"
model="networks/dave/model.onnx"

"""


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", type=Path, help="path to save generated configs")
    return parser.parse_args()


def get_num_neurons(network, scale, layer_ids=None):
    print(f"  Scaling by {scale}")
    layer_ids = layer_ids or [0, 1, 2, 3, 4, 7, 8, 9, 10]
    layers = [network.layers[l_id] for l_id in layer_ids]
    drop_layers = []
    scale_layers = []
    for i, l in zip(layer_ids, layers):
        if l.out_features * scale < 1:
            print(f"    Dropping: {i}", l.out_features, scale)
            drop_layers.append(i)
            network.drop_layer(i)
        else:
            scale_layers.append(i)
    if len(drop_layers) > 0:
        new_scale = (scale * 82669) / network.as_pytorch().num_neurons()
        print(f"    Old scale = {scale}; New scale = {new_scale}")
        num_neurons, new_drop_layers, scale_layers, scale = get_num_neurons(
            network, new_scale, layer_ids=scale_layers
        )
        return num_neurons, drop_layers + new_drop_layers, scale_layers, scale
    network = network.scale_layer(scale_layers, scale)
    model = network.as_pytorch()
    return model.num_neurons(), drop_layers, scale_layers, scale


def main(args):
    args.output_path.mkdir(parents=True, exist_ok=True)
    for i in range(20):
        print("TEST", i)
        network = load_network(
            {
                "model": "networks/dave/model.onnx",
                "input_shape": [1, 100, 100, 3],
                "input_format": "NHWC",
            }
        )
        scale = 1 / 24 + (i / 20) * (100 / 456)
        num_neurons, drop_layers, scale_layers, scale = get_num_neurons(network, scale)
        filename = args.output_path / (
            "dave.D.%s.S.%e.%s.toml"
            % (
                ".".join(str(l) for l in sorted(set(drop_layers))),
                scale,
                ".".join(str(l) for l in scale_layers),
            )
        )
        with open(filename, "w+") as f:
            f.write(TEMPLATE)
            if len(drop_layers) > 0:
                f.write("[[distillation.strategies.drop_layer]]\n")
                f.write("layer_id=%s\n\n" % list(drop_layers))
            if len(scale_layers) > 0:
                f.write("[[distillation.strategies.scale_layer]]\n")
                f.write(f"factor={scale}\n")
                f.write("layer_id=%s\n\n" % list(scale_layers))


if __name__ == "__main__":
    main(_parse_args())
