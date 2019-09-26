#!/usr/bin/env python3
import logging
import multiprocessing as mp
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

from itertools import combinations
from multiprocessing.pool import ThreadPool
from r4v.nn import load_network

TEMPLATE = """[distillation]
maxmemory="32G"
threshold=1e-9
cuda=true
type="regression"
precompute_teacher=true

[distillation.parameters]
epochs=50
optimizer="adadelta"
rho=0.95
loss="MSE"

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

[[distillation.strategies.drop_layer]]
"""


def get_num_neurons(drop_layers):
    # print("dropping layers (0):", drop_layers)
    dave = load_network(
        {
            "model": "networks/dave/model.onnx",
            "input_shape": [1, 100, 100, 3],
            "input_format": "NHWC",
        }
    )
    # print("dropping layers (1):", drop_layers)
    dave.drop_layer(list(drop_layers))
    # print("dropping layers (2):", drop_layers)
    network = dave.as_pytorch()
    # print("dropping layers (3):", drop_layers)
    return drop_layers, network.num_neurons()


def main():
    neuron_count_limit = (
        load_network(
            {
                "model": "networks/dave/model.onnx",
                "input_shape": [1, 100, 100, 3],
                "input_format": "NHWC",
            }
        )
        .as_pytorch()
        .num_neurons()
    )
    droppable_layers = [0, 1, 2, 3, 4, 7, 8, 9, 10]
    drop_layers = []
    for k in range(1, len(droppable_layers)):
        for layers in combinations(droppable_layers, k):
            print(layers)
            drop_layers.append(layers)

    print("Creating worker pool.")
    pool = ThreadPool(2)
    print("Measuring neuron counts.")
    neuron_counts = pool.imap_unordered(get_num_neurons, drop_layers)

    print("Collecting results.")
    configs = {}
    for layers, num_neurons in neuron_counts:
        configs[layers] = num_neurons
        print(layers, num_neurons)
    neuron_counts = list(configs.values())
    print(
        len(configs),
        np.min(neuron_counts),
        np.max(neuron_counts),
        np.mean(neuron_counts),
        np.median(neuron_counts),
    )
    min_neuron_count = np.min(neuron_counts)
    nbins = 10
    bin_width = (neuron_count_limit - min_neuron_count) / nbins
    bin_limits = [min_neuron_count + bin_width * i for i in range(nbins + 1)]
    bins = [[] for _ in range(nbins)]
    for layers, num_neurons in sorted(configs.items(), key=lambda kv: kv[1]):
        for i, (bin_min, bin_max) in enumerate(zip(bin_limits, bin_limits[1:])):
            if num_neurons >= bin_min and num_neurons < bin_max:
                bins[i].append(layers)
    for i, b in enumerate(bins):
        print(i, (bin_limits[i], bin_limits[i + 1]), len(b))
    print()
    selected = set()
    while len(selected) < 20:
        for i, b in enumerate(bins):
            if len(b) == 0:
                continue
            while True:
                c = np.random.choice(b)
                if c not in selected:
                    break
            selected.add(c)
            print(len(selected), (bin_limits[i], bin_limits[i + 1]), c)
    for layers in selected:
        filename = "configs/scenario.3/dave.tse/dave.tse.D.%s.toml" % ".".join(
            str(l) for l in layers
        )
        with open(filename, "w+") as f:
            f.write(TEMPLATE)
            f.write("layer_id=%s" % list(layers))


if __name__ == "__main__":
    main()
