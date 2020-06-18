"""
"""
import argparse

from collections import defaultdict
from typing import Callable, Dict

from .. import dispatcher
from ..cli.utils import (
    bool_type,
    float_type,
    int_type,
    literal_type,
    str_type,
    SetParameter,
)
from ..config import parse as parse_config


def dispatch(args: argparse.Namespace):
    from . import distill

    overrides = args.override
    plugins = [{"name": p[0], "path": p[1]} for p in args.plugins]
    overrides["plugins"] = plugins
    config = parse_config(args.config, override=overrides)

    distillation_config = config.distillation
    dispatcher.dispatch(
        target=distill,
        args=(distillation_config,),
        max_memory=config.distillation.max_memory,
        timeout=config.distillation.timeout,
    )


class SetDistillationParameter(SetParameter):
    qualified_names = {
        # distillation parameters
        "a": "distillation.parameters.alpha",
        "alpha": "distillation.parameters.alpha",
        "b": "distillation.parameters.batchsize",
        "batchsize": "distillation.parameters.batchsize",
        "batch_size": "distillation.parameters.batchsize",
        "epochs": "distillation.parameters.epochs",
        "learning_rate": "distillation.parameters.learning_rate",
        "lr": "distillation.parameters.learning_rate",
        "m": "distillation.parameters.momentum",
        "momentum": "distillation.parameters.momentum",
        "novalidation": "distillation.novalidation",
        "precompute_teacher": "distillation.precompute_teacher",
        "T": "distillation.parameters.T",
        "temperature": "distillation.parameters.T",
        "wd": "distillation.parameters.weight_decay",
        "weight_decay": "distillation.parameters.weight_decay",
        # network configuration
        "teacher.input_layer": "distillation.teacher.input_layer",
        "teacher.output_layer": "distillation.teacher.output_layer",
    }

    parameter_type: Dict[str, Callable] = defaultdict(lambda: literal_type)
    # distillation parameters
    parameter_type["distillation.novalidation"] = bool_type
    parameter_type["distillation.parameters.alpha"] = float_type
    parameter_type["distillation.parameters.batch_size"] = int_type
    parameter_type["distillation.parameters.epochs"] = int_type
    parameter_type["distillation.parameters.learning_rate"] = float_type
    parameter_type["distillation.parameters.momentum"] = float_type
    parameter_type["distillation.parameters.T"] = float_type
    parameter_type["distillation.parameters.weight_decay"] = float_type
    parameter_type["distillation.precompute_teacher"] = bool_type
    # network configuration
    parameter_type["distillation.teacher.input_layer"] = int_type
    parameter_type["distillation.teacher.output_layer"] = int_type


def add_subparser(subparsers: argparse._SubParsersAction):
    from ..cli import utils as cli_utils

    parser = subparsers.add_parser(
        "distill",
        description="Distill a teacher model to a verifiable student model",
        help="distill a teacher model to a verifiable student model",
        formatter_class=cli_utils.HelpFormatter,
        parents=[cli_utils.common_parser()],
    )

    parser.add_argument("config", help="configuration for distillation")

    parser.add_argument(
        "--set",
        nargs="+",
        metavar=("NAME", "VALUE"),
        action=SetDistillationParameter,
        dest="override",
        default={},
        help="set or override parameter options",
    )

    parser.set_defaults(func=dispatch)
