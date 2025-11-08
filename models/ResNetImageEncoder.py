import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import v2 as transforms  # Using v2 for modern transforms


class ResNetImageEncoder(nn.Module):
    """
    A PyTorch module that uses a frozen, pre-trained ResNet-18 as a
    feature extractor and passes the features through a small MLP
    to produce a final embedding of a specified dimension.
    """

    def __init__(self, embed_dim: int):
        """
        Initializes the model.

        Args:
            embed_dim (int): The desired output dimension of the embedding.
        """
        super(ResNetImageEncoder, self).__init__()

        # 1. Load pre-trained ResNet-18
        # We use the modern 'weights' argument for clarity and correctness
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        resnet18 = models.resnet18(weights=weights)

        # 2. Freeze all parameters in the ResNet
        # We don't want to update these weights during training
        for param in resnet18.parameters():
            param.requires_grad = False

        # 3. Get the number of input features for the original classifier
        # For ResNet-18, this is 512
        num_ftrs = resnet18.fc.in_features

        # 4. Remove the final fully connected layer (classifier)
        # We'll replace it with an Identity layer, so the model's 'forward'
        # method returns the 512-dim features from the avgpool layer.
        resnet18.fc = nn.Identity()

        self.resnet_features = resnet18

        # 5. Define the small MLP
        # This takes the 512 features from ResNet and maps them to 'embed_dim'
        self.mlp = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs // 2),  # 512 -> 256
            nn.ReLU(inplace=True),
            nn.Linear(num_ftrs // 2, embed_dim),  # 256 -> embed_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input image tensor, expected to be
                              preprocessed and normalized.
                              Shape: (batch_size, 3, 224, 224)

        Returns:
            torch.Tensor: The final embedding. Shape: (batch_size, embed_dim)
        """
        # Get features from ResNet
        # Input: (N, 3, 224, 224)
        # Output: (N, 512)
        features = self.resnet_features(x)

        # Pass features through the MLP
        # Input: (N, 512)
        # Output: (N, embed_dim)
        embedding = self.mlp(features)

        return embedding
