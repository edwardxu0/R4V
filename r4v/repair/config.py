from .. import logging
from ..config import Configuration
from ..data.config import DataConfiguration


class RepairConfiguration(Configuration):
    @property
    def data(self):
        data_config = self.config.get("data", None)
        if data_config is not None:
            return DataConfiguration(data_config)
        raise ValueError("No data configuration defined")
