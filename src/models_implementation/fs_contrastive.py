import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveModel(nn.Module):
    """
    A wrapper for any backbone (pretrained model) that adapts it for
    Contrastive Learning. It bypasses standard 10-class classification
    and instead projects features into a smaller space (embeddings),
    where distances can be computed.
    """
    def __init__(self, backbone, in_features, embedding_dim=128):
        super(ContrastiveModel, self).__init__()

        # Base model (e.g., MobileNet without the final FC layer)
        self.backbone = backbone
        # Projection Head! This layer will learn even if the backbone is fully frozen
        self.projection = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        # Extract embedding from the backbone
        features = self.backbone(x)
        # Pass features through the trainable projection head
        embeddings = self.projection(features)
        
        # Normalize the embedding (L2 normalization, p=2), which is critical for
        # Contrastive Learning (and calculating Cosine/Euclidean distances)     
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

class TripletLoss(nn.Module):
    """
    Triplet Loss - uses "triplets" of examples:
    1. Anchor - a random example
    2. Positive - another example from the SAME class
    3. Negative - an example from a DIFFERENT class
    
    It pulls the Anchor closer to the Positive and pushes the Negative away
    (to prevent representation collapse to one point).
    """
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Distance between anchor and positive (should approach 0)
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        # Distance between anchor and negative (should be large, at least 'margin')
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        
        # Loss: max(0, dist_pos - dist_neg + margin)
        # The error is 0 only if the distance to the negative is greater 
        # than the distance to the positive by at least 'margin'.
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


class ContrastiveLoss(nn.Module):
    """
    Standard Contrastive Loss operating on pairs (A, B) and a label
    indicating if they are from the same class (Label=1) or different (Label=0).
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        # Label=1 (same class): penalize large distances
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                      (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
