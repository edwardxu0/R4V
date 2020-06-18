"""
"""
import argparse

from collections import defaultdict
from typing import Callable, Dict

from .. import repair
from ..cli.utils import (
    bool_type,
    float_type,
    int_type,
    literal_type,
    str_type,
    SetParameter,
)
from ..config import parse as parse_config


class SetRepairParameter(SetParameter):
    pass


def add_subparser(subparsers: argparse._SubParsersAction):
    from ..cli import utils as cli_utils

    parser = subparsers.add_parser(
        "repair",
        description="Repair a DNN model",
        help="repair a DNN model",
        formatter_class=cli_utils.HelpFormatter,
        parents=[cli_utils.common_parser()],
    )

    parser.add_argument("config", help="configuration for repair")

    parser.add_argument(
        "--set",
        nargs="+",
        metavar=("NAME", "VALUE"),
        action=SetRepairParameter,
        dest="override",
        default={},
        help="set or override parameter options",
    )

    parser.set_defaults(func=repair.run)
