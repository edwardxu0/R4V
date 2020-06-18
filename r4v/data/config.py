from ..config import Configuration


class DataConfiguration(Configuration):
    @property
    def batch_size(self):
        return self.batchsize

    @property
    def batchsize(self):
        batch_size = self.config.get("batchsize", None)
        if batch_size is not None:
            return batch_size
        raise ValueError("No batch size defined")

    @property
    def path(self):
        path = self.config.get("path", None)
        if path is not None:
            return path
        raise ValueError("No data path defined")

    @property
    def test(self):
        test_config = self.config.get("test", None)
        if test_config is not None:
            data_config = self.config.copy()
            data_config.update(test_config)
            data_config["_STAGE"] = "test"
            return DataConfiguration(data_config)
        raise ValueError("No test configuration defined")

    @property
    def train(self):
        train_config = self.config.get("train", None)
        if train_config is not None:
            data_config = self.config.copy()
            data_config.update(train_config)
            data_config["_STAGE"] = "train"
            return DataConfiguration(data_config)
        raise ValueError("No train configuration defined")

    @property
    def validation(self):
        validation_config = self.config.get("validation", None)
        if validation_config is not None:
            data_config = self.config.copy()
            data_config.update(validation_config)
            data_config["_STAGE"] = "validation"
            return DataConfiguration(data_config)
        raise ValueError("No validation configuration defined")
