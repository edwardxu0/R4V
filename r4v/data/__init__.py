import numpy as np
import os
import torch.utils.data as data

from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader

from .config import DataConfiguration
from ..utils import get_subclasses


class Dataset(data.Dataset):
    def __init__(self, *roots, transforms=None, target_transforms=None):
        self.samples = []
        for root in roots:
            samples = self._process_root_dir(root)
            self.samples.append(samples)
            if len(self.samples[0]) != len(samples):
                raise ValueError(
                    "Dataset folders must have the same number of samples."
                )
        self.transforms = transforms
        self.target_transforms = target_transforms

    def _process_root_dir(self, root_dir):
        raise NotImplementedError()

    def process_sample(self, sample):
        return sample

    def assert_same_targets(self, target0, target1):
        assert target0 == target1

    def __getitem__(self, index):
        samples = []
        targets = []
        for i, dataset in enumerate(self.samples):
            raw_sample, target = dataset[index]
            sample = self.process_sample(raw_sample)
            if self.transforms is not None:
                sample = self.transforms[i](sample)
            if self.target_transforms is not None:
                target = self.target_transforms[i](target)
            samples.append(sample)
            targets.append(target)
            self.assert_same_targets(target, targets[0])
        return (index, tuple(samples), target)

    def __len__(self):
        return len(self.samples[0])

    def __str__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(len(self))
        fmt_str += "    Root Locations: {}\n".format(self.roots)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, repr(self.transforms).replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp, repr(self.target_transforms).replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str


class DataTransform:
    def __init__(self, config):
        pass

    def __call__(self, x):
        return x


class UdacityDriving(Dataset):
    def __init__(
        self, *roots, transforms=None, target_transforms=None, loader=default_loader
    ):
        super().__init__(
            *roots, transforms=transforms, target_transforms=target_transforms
        )
        self.loader = loader

    def _process_root_dir(self, root_dir):
        samples = []
        interpolated_csv = os.path.join(root_dir, "interpolated.csv")
        with open(interpolated_csv) as f:
            _ = f.readline()
            for line in f:
                split_line = line.split(",")
                target = np.float32(split_line[6])
                samples.append((os.path.join(root_dir, split_line[5]), target))
        return samples

    def process_sample(self, sample):
        return self.loader(sample)


def get_data_loader(config: DataConfiguration) -> DataLoader:
    datasets = {c.__name__: c for c in get_subclasses(Dataset)}
    transforms = [DataTransform(config.transform)]

    return DataLoader(
        datasets[config.dataset](config.path, transforms=transforms),
        batch_size=config.get("batch_size", 1),
        shuffle=config.get("shuffle", False),
    )
