"""
"""
import argparse

from collections import defaultdict

from .. import dispatcher
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


def float_type(parser, name, x):
    if len(x) > 1:
        raise parser.error(f"Too many values for parameter {name}.")
    return float(x[0])


def int_type(parser, name, x):
    if len(x) > 1:
        raise parser.error(f"Too many values for parameter {name}.")
    return int(x[0])


def str_type(parser, name, x):
    if len(x) > 1:
        raise parser.error(f"Too many values for parameter {name}.")
    return str(x[0])


def bool_type(parser, name, x):
    if len(x) > 1:
        raise parser.error(f"Too many values for parameter {name}.")
    if len(x) == 1:
        return x[0].lower() in ["true", "1"]
    return True


def literal_type(parser, name, x):
    import ast

    if len(x) > 1:
        raise parser.error(f"Too many values for parameter {name}.")
    try:
        return ast.literal_eval(x[0])
    except:
        return x[0]


class SetParameter(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings=option_strings, dest=dest, **kwargs)
        self.parameters = {}
        self.qualified_names = {
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
        self.parameter_type = defaultdict(lambda: literal_type)
        # distillation parameters
        self.parameter_type["distillation.novalidation"] = bool_type
        self.parameter_type["distillation.parameters.alpha"] = float_type
        self.parameter_type["distillation.parameters.batch_size"] = int_type
        self.parameter_type["distillation.parameters.epochs"] = int_type
        self.parameter_type["distillation.parameters.learning_rate"] = float_type
        self.parameter_type["distillation.parameters.momentum"] = float_type
        self.parameter_type["distillation.parameters.T"] = float_type
        self.parameter_type["distillation.parameters.weight_decay"] = float_type
        self.parameter_type["distillation.precompute_teacher"] = bool_type
        # network configuration
        self.parameter_type["distillation.teacher.input_layer"] = int_type
        self.parameter_type["distillation.teacher.output_layer"] = int_type

    def __call__(self, parser, namespace, values, option_string=None):
        name, *val = values
        qualified_name = name
        if name in self.qualified_names:
            qualified_name = self.qualified_names[name]
        if qualified_name in self.parameters:
            raise parser.error(f"Multiple values specified for parameter {name}")
        value = self.parameter_type[qualified_name](parser, name, val)
        self.parameters[qualified_name] = value
        items = (getattr(namespace, self.dest) or {}).copy()
        items[qualified_name] = value
        setattr(namespace, self.dest, items)


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
        "--plugin",
        nargs=2,
        metavar=("NAME", "PATH"),
        action="append",
        dest="plugins",
        default=[],
        help="distillation plugins",
    )
    parser.add_argument(
        "--set",
        nargs="+",
        metavar=("NAME", "VALUE"),
        action=SetParameter,
        dest="override",
        default={},
        help="set or override parameter options",
    )

    parser.set_defaults(func=dispatch)
