import os
from dotenv import load_dotenv

load_dotenv()

from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np


class CIFAR10Dataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.image_size = config["image_size"]
        self.transform = transforms.ToTensor()
        data_root_dir = os.getenv("DATA_ROOT_DIR")
        train = config["train"]
        self.dataset = torchvision.datasets.CIFAR10(
            root=data_root_dir, train=train, transform=self.transform, download=True
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        mask = np.ones(self.image_size, dtype=np.uint8) * 255
        mask = self.transform(mask).squeeze(0)
        return {"images": image, "object_masks": mask, "metadata": {"label": label}}
