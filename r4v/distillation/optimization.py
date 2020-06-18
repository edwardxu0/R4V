import numpy as np
import torch
import torch.optim as optim

from .. import logging
from ..utils import get_subclasses


class Loss:
    def __init__(self, config):
        self.config = config

    def initialize(self, model):
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.get("learning_rate", 0.001),
            betas=(self.config.get("beta1", 0.9), self.config.get("beta2", 0.999)),
            weight_decay=self.config.get("weight_decay", 0),
        )

    def step(self, epoch, model, device="cpu"):
        logger = logging.getLogger(__name__)
        if self.config.get("validation_only", False):
            return
        for i in range(self.config.get("iterations", 1)):
            self.optimizer.zero_grad()
            loss = self.compute_loss(model, device=device)
            loss.backward()
            self.optimizer.step()
        logger.info("%s: %f", self.__class__.__name__, loss.item())

    def compute_loss(self, model, device="cpu"):
        raise NotImplementedError()

    def compute_val_loss(self, model, device="cpu"):
        raise NotImplementedError()


class ReluLoss(Loss):
    def __init__(self, config):
        super().__init__(config)
        self.input_region = np.load(config["input_region"])

    def sample_input_region(self):
        x_ = []
        for region in self.input_region:
            x_.append(np.random.uniform(region[0], region[1]))
        return torch.from_numpy(np.concatenate(x_)).float()

    def _loss(self, model, x):
        from ..nn.pytorch import Relu

        Relu.pre_relu_values = torch.zeros(x.size(0), device=x.device) + float("inf")

        orig_relu_forward = Relu.forward

        def relu_forward(self, x):
            Relu.pre_relu_values = torch.min(
                Relu.pre_relu_values, torch.abs(x).flatten(1).min(1)[0]
            )
            return orig_relu_forward(self, x)

        Relu.forward = relu_forward
        # TODO : do multiple passes of random samples
        # forward pass on center, then a random sample
        # high loss if sample has different sign then center
        # high loss if sample is far from center
        _ = model(x)
        Relu.forward = orig_relu_forward
        return torch.exp(-Relu.pre_relu_values).sum() / len(x)

    def compute_loss(self, model, device="cpu"):
        x = self.sample_input_region().to(device)
        return self._loss(model, x)

    def compute_val_loss(self, model, device="cpu"):
        x_ = []
        for region in self.input_region:
            x_.append((region[0] + region[1]) / 2)
        x = torch.from_numpy(np.concatenate(x_)).float().to(device)
        return self._loss(model, x)


def build_loss(config):
    subclasses = {c.__name__: c for c in get_subclasses(Loss)}
    loss = subclasses[config["type"]](config)
    return loss
