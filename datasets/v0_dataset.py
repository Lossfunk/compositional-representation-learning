import random

from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import cv2

class v0Dataset(Dataset):
    def __init__(self, config):
        
        self.config = config

        self.image_size = config['image_size']
        self.shapes = config['shapes']
        self.colors = config['colors']
        self.color_map = {
            "red": (255, 0, 0),
            "blue": (0, 0, 255)
        }

        self.num_samples = config['num_samples']

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image, mask = self.generate_image()
        image = self.transform(image)
        mask = self.transform(mask).squeeze(0)
        
        return {
            "images": image,
            "object_masks": mask
        }

    def generate_image(self):
        image = np.zeros([self.image_size[0], self.image_size[1], 3], dtype=np.uint8)
        mask = np.zeros(self.image_size, dtype=np.uint8)
        
        shape = random.choice(self.shapes)
        color = self.color_map[random.choice(self.colors)]
        
        if shape == "circle":
            radius = np.random.randint(10, 30)
            center = (np.random.randint(radius, self.image_size[1] - radius), np.random.randint(radius, self.image_size[0] - radius))
            cv2.circle(image, center, radius, color, -1)
            cv2.circle(mask, center, radius, 255, -1)
        elif shape == "square":
            side_length = np.random.randint(10, 30)
            top_left = (np.random.randint(0, self.image_size[1] - side_length), np.random.randint(0, self.image_size[0] - side_length))
            cv2.rectangle(image, top_left, (top_left[0] + side_length, top_left[1] + side_length), color, -1)
            cv2.rectangle(mask, top_left, (top_left[0] + side_length, top_left[1] + side_length), 255, -1)
        
        return image, mask