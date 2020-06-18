from .cli import add_subparser
from .config import RepairConfiguration
from ..config import parse as parse_config
from ..data import get_data_loader
from ..nn import load_network
from .. import logging


def run(args):
    overrides = args.override
    plugins = [{"name": p[0], "path": p[1]} for p in args.plugins]
    overrides["plugins"] = plugins
    config = parse_config(args.config, override=overrides)

    return repair(config.repair)


def repair(config: RepairConfiguration):
    logger = logging.getLogger(__name__)

    print(config.config)

    network = load_network(config.model)
    print(network)

    train_loader = get_data_loader(config.data.train)
    val_loader = get_data_loader(config.data.validation)

    print(train_loader.dataset)
