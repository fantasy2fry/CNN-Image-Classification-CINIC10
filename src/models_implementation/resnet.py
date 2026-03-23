import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights


class FinetunedResNet34(nn.Module):
    """
    Pre-trained ResNet-34 adapted for CINIC-10 classification (10 classes).
    Supports freezing the feature extraction layers for Transfer Learning experiments.
    """

    def __init__(self, num_classes=10, freeze_features=True):
        super(FinetunedResNet34, self).__init__()

        # 1. Load the pre-trained ResNet-34 model with optimal ImageNet weights
        self.resnet = resnet34(weights=ResNet34_Weights.DEFAULT)

        # 2. Freeze the parameters of the feature extractor if requested
        # This means these weights won't be updated during training (saves memory and time)
        if freeze_features:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # 3. Replace the final classification layer (the "head")
        # ResNet's final layer is stored in an attribute called 'fc' (fully connected).
        # We read how many input features it expects (512 for ResNet-34)...
        num_ftrs = self.resnet.fc.in_features

        # ...and replace it with a brand new, UNFFROZEN linear layer for our 10 classes.
        # By default, newly created layers in PyTorch have requires_grad=True.
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        """
        Defines the forward pass.
        """
        # Just pass the input through the modified ResNet
        return self.resnet(x)
