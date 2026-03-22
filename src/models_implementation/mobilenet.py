import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

def get_finetuned_mobilenet(num_classes=10, freeze_features=True):
    """
    Loads a pre-trained MobileNetV2 and modifies the final layer for CINIC-10.
    
    Args:
        num_classes (int): Number of output classes (10 for CINIC-10).
        freeze_features (bool): If True, freezes the convolutional base so only 
                                the new classifier head is trained. 
                                Perfect for pure fine-tuning / Transfer Learning.
    """
    
    # 1. Download the pre-trained MobileNetV2 from PyTorch
    # The weights are downloaded automatically and cached by PyTorch.
    # No need to manually download anything to your 'models/' folder!
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    
    # 2. Freeze the feature extractor (optional but standard for fine-tuning)
    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False
            
    # 3. Replace the 'classifier' head
    # MobileNetV2's classifier is a Sequential block. The original was trained on 
    # ImageNet (1000 classes). We need to replace the last Linear layer to output 10.
    
    # Let's see what the original classifier looks like inside:
    # (0): Dropout(p=0.2, inplace=False)
    # (1): Linear(in_features=1280, out_features=1000, bias=True)
    
    # We grab the 'in_features' from the original last layer (which is 1280)
    in_features = model.classifier[1].in_features
    
    # We replace the entire classifier block with a new one for CINIC-10
    model.classifier = nn.Sequential(
        #  nn.Dropout(p=0.2), # Standard dropout for MobileNet, it should be more like baseline
        nn.Linear(in_features, num_classes) # Map 1280 features to our 10 classes
    )
    
    return model

# Quick test if run directly
if __name__ == "__main__":
    model = get_finetuned_mobilenet(num_classes=10)
    print("MobileNetV2 modified for CINIC-10 successfully!")
    print(f"Final layer: {model.classifier[1]}")
