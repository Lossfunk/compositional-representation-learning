from .v0_dataset import v0Dataset
from .cifar import CIFAR10Dataset


def get_dataset(config):
    dataset_type = config["data"]["train"]["type"]
    if dataset_type == "v0Dataset":
        return v0Dataset(config["data"]["train"]["config"])
    elif dataset_type == "CIFAR10Dataset":
        return CIFAR10Dataset(config["data"]["train"]["config"])
    else:
        raise ValueError(f"Dataset type {dataset_type} not implemented.")
