import importlib
import toml

from functools import partial
from types import ModuleType
from typing import Dict

from . import logging


def parse(config_filename, override={}):
    config_dict = toml.load(config_filename)
    tmp_dict = config_dict
    for key, value in override.items():
        if value is None:
            continue
        sections = key.split(".")
        for section in sections[:-1]:
            if section not in tmp_dict:
                tmp_dict[section] = {}
            tmp_dict = tmp_dict[section]
        tmp_dict[sections[-1]] = value
        tmp_dict = config_dict
    return Configuration(config_dict)


class Configuration:
    PLUGINS: Dict[str, ModuleType] = {}

    def __init__(self, config):
        self.config = config
        self._cache = {}
        if "plugins" in self.config:
            for plugin in self.config["plugins"]:
                if plugin["name"] not in self.__class__.PLUGINS:
                    spec = importlib.util.spec_from_file_location(
                        plugin["name"], plugin["path"]
                    )
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    self.__class__.PLUGINS[plugin["name"]] = mod

    def __getattr__(self, name):
        if name in self.config:
            return self.config[name]
        return object.__getattribute__(self, name)

    def __getitem__(self, name):
        if hasattr(self, name):
            return getattr(self, name)
        return self.config[name]

    def get(self, name, default=None):
        if hasattr(self, name):
            return getattr(self, name)
        return self.config.get(name, default)

    @property
    def distillation(self):
        from .distillation import DistillationConfiguration

        return DistillationConfiguration(self.config["distillation"])

    @property
    def max_memory(self):
        maxmem = self.config.get("maxmemory", -1)
        if isinstance(maxmem, int):
            return maxmem
        elif maxmem.endswith("G"):
            return int(maxmem[:-1]) * 1_000_000_000
        elif maxmem.endswith("M"):
            return int(maxmem[:-1]) * 1_000_000
        elif maxmem.endswith("k"):
            return int(maxmem[:-1]) * 1000
        else:
            return int(maxmem)

    @property
    def timeout(self):
        return int(self.config.get("timeout", -1))
