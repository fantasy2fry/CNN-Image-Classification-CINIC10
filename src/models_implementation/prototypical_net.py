import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


# ==========================================
# 1. Prototypical Loss Function
# ==========================================
class PrototypicalLoss(nn.Module):
    """
    Computes the Prototypical Network Loss.
    It groups embeddings by class, calculates the mean (prototype) for each class,
    and then calculates the Euclidean distance from each embedding to the prototypes.
    """

    def __init__(self):
        super(PrototypicalLoss, self).__init__()

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: Raw feature vectors from the CNN (Shape: [batch_size, feature_dim])
            labels: True labels for the batch (Shape: [batch_size])
        """
        # Find which classes are present in the current batch
        unique_classes = torch.unique(labels)

        prototypes = []

        # Calculate the center of gravity (Prototype) for each class
        for c in unique_classes:
            class_mask = (labels == c)
            class_embeddings = embeddings[class_mask]

            # Mean across the batch dimension
            class_prototype = class_embeddings.mean(dim=0)
            prototypes.append(class_prototype)

        # Stack prototypes: [num_unique_classes, feature_dim]
        prototypes = torch.stack(prototypes)

        # Measure Squared Euclidean Distances from every embedding to every prototype
        distances = torch.cdist(embeddings, prototypes, p=2.0) ** 2

        # Convert distances to logits (Network wants to minimize loss, so we negate distances)
        # Shortest distance becomes the largest logit value
        logits = -distances

        # Map original labels (e.g., [3, 7, 8]) to prototype indices (e.g., [0, 1, 2])
        target_indices = torch.zeros_like(labels)
        for i, c in enumerate(unique_classes):
            target_indices[labels == c] = i

        # Calculate standard Cross Entropy on the negative distances
        loss = F.cross_entropy(logits, target_indices)

        return loss, logits


# ==========================================
# 2. "Headless" Feature Extractor Model
# ==========================================
class PrototypicalResNet34(nn.Module):
    """
    ResNet-34 prepared for Prototypical Networks.
    Outputs raw embeddings (512 dimensions) instead of class probabilities.
    """

    def __init__(self, freeze_features=False, embedding_dim=128, dropout=0.3):
        super(PrototypicalResNet34, self).__init__()

        # Load pre-trained ResNet
        self.resnet = resnet34(weights=ResNet34_Weights.DEFAULT)

        # Freeze strategy for few-shot:
        # - freeze_features=True  → freeze all except layer4 (last residual block) + fc
        #   This is better than freezing everything: the model can still adapt
        #   the highest-level features without destroying pretrained low-level ones.
        # - freeze_features=False → full fine-tuning (use with higher weight_decay)
        if freeze_features:
            for name, param in self.resnet.named_parameters():
                if not (name.startswith('layer4') or name.startswith('fc')):
                    param.requires_grad = False

        # Projection head: 512 → embedding_dim with Dropout.
        # Smaller embedding forces compact, discriminative representations.
        # Dropout regularizes against overfitting on tiny few-shot sets.
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x):
        embeddings = self.resnet(x)
        # L2 normalization: keeps embeddings on a unit hypersphere,
        # stabilizes Euclidean distances and prevents loss from exploding
        return F.normalize(embeddings, p=2, dim=1)