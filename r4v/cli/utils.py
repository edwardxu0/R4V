"""
"""
import argparse

from collections import defaultdict
from typing import Callable, Dict

from .. import __version__
from .. import logging


class HelpFormatter(argparse.HelpFormatter):
    def _format_action_invocation(self, action):
        if not action.option_strings:
            (metavar,) = self._metavar_formatter(action, action.dest)(1)
            return metavar
        else:
            parts = []
            # if the Optional doesn't take a value, format is:
            #    -s, --long
            if action.nargs == 0:
                parts.extend(action.option_strings)

            # if the Optional takes a value, format is:
            #    -s ARGS, --long ARGS
            # change to
            #    -s, --long ARGS
            else:
                default = action.dest.upper()
                args_string = self._format_args(action, default)
                for option_string in action.option_strings:
                    parts.append("%s" % option_string)
                parts[-1] += " %s" % args_string
            return ", ".join(parts)


def common_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-V", "--version", action="version", version=__version__)
    parser.add_argument("--seed", type=int, default=None, help="the random seed to use")
    parser.add_argument(
        "--plugin",
        nargs=2,
        metavar=("NAME", "PATH"),
        action="append",
        dest="plugins",
        default=[],
        help="the name and path of a plugin module to load",
    )
    logging.add_arguments(parser)
    return parser


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
    qualified_names: Dict[str, str] = {}
    parameter_type: Dict[str, Callable] = defaultdict(lambda: literal_type)

    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings=option_strings, dest=dest, **kwargs)
        self.parameters = {}

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
