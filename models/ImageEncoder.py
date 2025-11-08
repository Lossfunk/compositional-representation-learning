from torch import nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.conv_1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding="same"
        )
        self.conv_2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding="same"
        )

        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding="same"
        )
        self.conv_4 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding="same"
        )

        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_5 = nn.Conv2d(
            in_channels=128,
            out_channels=embed_dim,
            kernel_size=1,
            stride=1,
            padding="same",
        )

    def forward(self, x):

        x = self.conv_1(x)
        x = F.relu(x)
        x = self.conv_2(x)
        x = F.relu(x)

        x = self.maxpool_1(x)
        x = F.relu(x)

        x = self.conv_3(x)
        x = F.relu(x)
        x = self.conv_4(x)
        x = F.relu(x)

        x = self.maxpool_2(x)
        x = F.relu(x)

        x = self.conv_5(x)

        return x
