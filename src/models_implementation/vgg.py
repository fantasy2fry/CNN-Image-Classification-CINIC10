import torch
import torch.nn as nn


class VGG11(nn.Module):
    """
    VGG-11 architecture implemented from scratch, adapted for 32x32 images (CINIC-10).
    Includes configurable Dropout for regularization experiments.
    """

    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(VGG11, self).__init__()

        # ==========================================
        # 1. Feature Extractor (Convolutional Blocks)
        # ==========================================
        # Image spatial dimensions: 32x32 -> 16x16 -> 8x8 -> 4x4 -> 2x2 -> 1x1
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Size becomes 16x16

            # Block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Size becomes 8x8

            # Block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Size becomes 4x4

            # Block 4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Size becomes 2x2

            # Block 5
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Size becomes 1x1
        )

        # ==========================================
        # 2. Classifier (Fully Connected Layers)
        # ==========================================
        # After the last MaxPool, we have 512 channels of size 1x1.
        # Flattened size = 512 * 1 * 1 = 512.
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),  # Configurable dropout for experiments

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),  # Configurable dropout for experiments

            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        """
        Defines the data flow through the network.
        """
        # 1. Extract features using convolutional layers
        x = self.features(x)

        # 2. Flatten the 4D tensor [batch_size, channels, height, width]
        # into a 2D tensor [batch_size, flat_features]
        x = torch.flatten(x, 1)

        # 3. Output predictions using the classifier
        x = self.classifier(x)

        return x
