#!/usr/bin/env python
import argparse
import numpy as np
import torch

from pathlib import Path

from dnnv.nn import parse as parse_network, OperationGraph
from dnnv.nn.operations import Relu, Operation
from dnnv.properties import parse as parse_property
from dnnv.verifiers.common import ConvexPolytopeExtractor
from r4v.nn_v2 import load_network
from r4v.nn_v2.pytorch import Relu


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Approximate the difficulty of a DNN property"
    )

    parser.add_argument("model_path", type=Path, help="path to model")
    parser.add_argument("property_path", type=Path, help="path to property")

    return parser.parse_args()


class Sentinal(Operation):
    def __init__(self, x):
        self.x = x


def abs_min(y):
    return abs(y).min()


def abs_max(y):
    return abs(y).max()


def abs_mean(y):
    return abs(y).mean()


def measure(model, phi):
    difficulties = []
    for prop in ConvexPolytopeExtractor().extract_from(phi):
        input_constraint = prop.input_constraint.as_hyperrectangle()
        lb = input_constraint.lower_bound
        ub = input_constraint.upper_bound
        center = ((ub + lb) / 2).astype(np.float32)
        model(torch.from_numpy(center).to(torch.device("cuda")))
        relu_loss = model.relu_loss().item()
        difficulties.append(relu_loss)
    return max(difficulties)


def main(args):
    op_graph = parse_network(args.model_path)
    op_graph.pprint()

    phi = parse_property(args.property_path)
    print(phi)
    phi.networks[0].concretize(op_graph)

    model = load_network({"model": args.model_path}).as_pytorch(maintain_weights=True)
    model.to(torch.device("cuda"))

    difficulty = measure(model, phi)
    print(difficulty)


if __name__ == "__main__":
    main(_parse_args())
