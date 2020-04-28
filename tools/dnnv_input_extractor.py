import argparse
import numpy as np

from collections import namedtuple
from pathlib import Path

from dnnv.properties import parse as parse_property
from dnnv.verifiers.common import ConvexPolytopeExtractor

InputDetails = namedtuple("InputDetails", ["shape", "dtype"])


def _parse_args():
    parser = argparse.ArgumentParser(
        description="dnnv property input precondition extractor"
    )
    parser.add_argument("properties", nargs="+", type=Path)
    parser.add_argument("-o", "--output_path", type=Path, required=True)
    return parser.parse_known_args()


def main(args, extra_args):
    def mock_network(x):
        return np.zeros((1, 10))

    mock_network.input_details = (InputDetails((1, 1, 28, 28), np.float32),)
    mock_network.output_shape = ((1, 10),)
    mock_network.output_operations = [None]

    input_preconditions = []
    for property_path in args.properties:
        prop = parse_property(property_path, extra_args)
        prop.concretize(**{"N": mock_network})
        for constraints in ConvexPolytopeExtractor().extract_from(prop):
            new_precondition = constraints.input_constraint.as_hyperrectangle()
            for pre in input_preconditions:
                if (new_precondition.lower_bound == pre[0]).all() and (
                    new_precondition.upper_bound == pre[1]
                ).all():
                    break
            else:
                input_preconditions.append(
                    (new_precondition.lower_bound, new_precondition.upper_bound)
                )
    print(len(input_preconditions))
    np.save(args.output_path, input_preconditions)
    if extra_args:
        print("Unused args:", extra_args)


if __name__ == "__main__":
    main(*_parse_args())
