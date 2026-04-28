import random

from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import cv2


class v0Dataset(Dataset):
    def __init__(self, config):

        self.config = config
        self.return_metadata = config["return_metadata"]

        self.image_size = config["image_size"]
        self.shapes = config["shapes"]
        self.colors = config["colors"]
        self.center_range = config["center_range"]
        self.size_range = config["size_range"]
        self.color_map = {
            "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255),
            "yellow": (255, 255, 0), "cyan": (0, 255, 255),
        }
        self.excluded_combinations = config.get("excluded_combinations", [])

        self.num_samples = config["num_samples"]

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image, mask, metadata = self.generate_image()
        image = self.transform(image)
        mask = self.transform(mask).squeeze(0)
        if self.return_metadata:
            return {"images": image, "object_masks": mask, "metadata": metadata}
        else:
            return {"images": image, "object_masks": mask}

    def generate_image(self):
        image = np.zeros([self.image_size[0], self.image_size[1], 3], dtype=np.uint8)
        mask = np.zeros(self.image_size, dtype=np.uint8)

        while True:
            shape = random.choice(self.shapes)
            color_name = random.choice(self.colors)
            if f"{color_name} {shape}" not in self.excluded_combinations:
                break
        
        color = self.color_map[color_name]

        size = np.random.randint(self.size_range[0], self.size_range[1])
        center = (
            np.random.randint(self.center_range[0], self.center_range[1]),
            np.random.randint(self.center_range[0], self.center_range[1]),
        )

        if shape == "circle":
            radius = size // 2
            cv2.circle(image, center, radius, color, -1)
            cv2.circle(mask, center, radius, 255, -1)
        elif shape == "square":
            side_length = size
            cv2.rectangle(
                image,
                (center[0] - side_length // 2, center[1] - side_length // 2),
                (center[0] + side_length // 2, center[1] + side_length // 2),
                color,
                -1,
            )
            cv2.rectangle(
                mask,
                (center[0] - side_length // 2, center[1] - side_length // 2),
                (center[0] + side_length // 2, center[1] + side_length // 2),
                255,
                -1,
            )
        elif shape == "triangle":
            r = size // 2
            pts = np.array([
                [center[0], center[1] - r],
                [center[0] - int(r * np.sin(np.pi / 3)), center[1] + r // 2],
                [center[0] + int(r * np.sin(np.pi / 3)), center[1] + r // 2],
            ], dtype=np.int32)
            cv2.fillPoly(image, [pts], color)
            cv2.fillPoly(mask, [pts], 255)
        elif shape == "pentagon":
            r = size // 2
            angles = [np.pi / 2 + 2 * np.pi * k / 5 for k in range(5)]
            pts = np.array([
                [center[0] + int(r * np.cos(a)), center[1] - int(r * np.sin(a))]
                for a in angles
            ], dtype=np.int32)
            cv2.fillPoly(image, [pts], color)
            cv2.fillPoly(mask, [pts], 255)
        elif shape == "star":
            r_outer = size // 2
            r_inner = r_outer * 2 // 5
            pts = []
            for k in range(5):
                # Outer point
                a_out = np.pi / 2 + 2 * np.pi * k / 5
                pts.append([center[0] + int(r_outer * np.cos(a_out)),
                            center[1] - int(r_outer * np.sin(a_out))])
                # Inner point (rotated by pi/5)
                a_in = a_out + np.pi / 5
                pts.append([center[0] + int(r_inner * np.cos(a_in)),
                            center[1] - int(r_inner * np.sin(a_in))])
            pts = np.array(pts, dtype=np.int32)
            cv2.fillPoly(image, [pts], color)
            cv2.fillPoly(mask, [pts], 255)

        metadata = {
            "shape": shape,
            "color": color,
        }

        return image, mask, metadata
