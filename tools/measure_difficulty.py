#!/usr/bin/env python
import argparse
import numpy as np

from pathlib import Path

from dnnv.nn import parse as parse_network, OperationGraph
from dnnv.nn.operations import Relu, Operation
from dnnv.properties import parse as parse_property
from dnnv.verifiers.common import ConvexPolytopeExtractor


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


def measure(op_graph, phi, difficulty=None):
    if difficulty is None:
        difficulty = abs_min
    difficulties = []
    for prop in ConvexPolytopeExtractor().extract_from(phi):
        input_constraint = prop.input_constraint.as_hyperrectangle()
        lb = input_constraint.lower_bound
        ub = input_constraint.upper_bound
        center = ((ub + lb) / 2).astype(np.float32)

        partial_difficulties = []
        op_graph_ = OperationGraph([Sentinal(op_graph.output_operations[0])])
        current_op = op_graph_[:1]
        for i in range(2, 100):
            next_op = op_graph_[:i]
            assert len(next_op.output_operations) == 1
            if isinstance(next_op.output_operations[0], Sentinal):
                break
            if isinstance(next_op.output_operations[0], Relu):
                y = current_op(center)
                partial_difficulties.append(difficulty(y))
            current_op = next_op
        difficulties.append(partial_difficulties)
    return difficulties


def main(args):
    op_graph = parse_network(args.model_path)[2:]
    op_graph.pprint()

    phi = parse_property(args.property_path)
    print(phi)
    phi.networks[0].concretize(op_graph)

    for diff_measure in [abs_min, abs_max, abs_mean]:
        print(diff_measure.__name__)
        difficulties = measure(op_graph, phi, diff_measure)
        print(difficulties)
        print(
            f"min={np.min(difficulties):f} mean={np.mean(difficulties):f} max={np.max(difficulties):f}"
        )
        print()


if __name__ == "__main__":
    main(_parse_args())
