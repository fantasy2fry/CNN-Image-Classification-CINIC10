import os
import random
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


# ==========================================
# 1. Reproducibility
# ==========================================
def set_seed(seed=42):
    """
    Sets the seed for all random number generators to ensure reproducible results.
    """
    # 1. Python & NumPy
    random.seed(seed)
    np.random.seed(seed)

    # 2. PyTorch Base
    torch.manual_seed(seed)

    # 3. PyTorch CUDA (NVIDIA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 4. PyTorch MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# ==========================================
# 2. Contrastive Learning Helper
# ==========================================
class TwoCropTransform:
    """
    Generates two differently augmented views of the same image.
    Essential for Contrastive Learning.
    """

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]


# ==========================================
# 3. Main DataLoader Function
# ==========================================
def get_cinic10_dataloaders(
        data_dir,
        batch_size=128,
        num_workers=0,
        samples_per_class=None,
        use_crop=True,
        use_horizontal_flip=True,
        use_rotation=True,
        use_cutout=False,
        is_contrastive=False,
        pretrained=False
):
    """
    Creates and returns DataLoaders for the CINIC-10 dataset.

    Args:
    - data_dir (str): Path to the extracted CINIC-10 dataset.
    - batch_size (int): Size of the batch.
    - num_workers (int): Number of subprocesses to use for data loading (0 for macOS).
    - samples_per_class (int): Exact number of images per class for balanced Few-Shot Learning.
    - use_crop (bool): Whether to apply Random Crop augmentation.
    - use_horizontal_flip (bool): Whether to apply Random Horizontal Flip augmentation.
    - use_rotation (bool): Whether to apply Random Rotation augmentation.
    - use_cutout (bool): Whether to apply the Cutout (Random Erasing) augmentation.
    - is_contrastive (bool): Whether to return two augmented views for contrastive learning.
    - pretrained (bool): If True, resizes images to 224x224 and applies ImageNet normalization.
    """

    # --- Configuration based on model type ---
    if pretrained:
        # Standard ImageNet normalization values
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
        target_size = 224
    else:
        # Official CINIC-10 normalization values
        norm_mean = [0.47889522, 0.4722784, 0.43047404]
        norm_std = [0.24205776, 0.23828046, 0.25874835]
        target_size = 32

    # --- Data Augmentation Pipeline ---
    train_transform_list = []
    test_transform_list = []

    # 1. Resizing & Cropping
    if pretrained:
        train_transform_list.append(transforms.Resize(256))
        test_transform_list.extend([
            transforms.Resize(256),
            transforms.CenterCrop(target_size)
        ])
        if use_crop:
            train_transform_list.append(transforms.RandomCrop(target_size))
        else:
            # Fallback if crop is disabled: Center crop to maintain 224x224 shape
            train_transform_list.append(transforms.CenterCrop(target_size))
    else:
        if use_crop:
            train_transform_list.append(transforms.RandomCrop(target_size, padding=4))
        # No resizing/cropping needed for 32x32 test set

    # 2. Standard Operations: Horizontal Flip, Rotation
    if use_horizontal_flip:
        train_transform_list.append(transforms.RandomHorizontalFlip())

    if use_rotation:
        train_transform_list.append(transforms.RandomRotation(15))

    # 3. Tensor Conversion & Normalization
    train_transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    test_transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    # 4. Advanced Technique: Cutout (RandomErasing in PyTorch)
    if use_cutout:
        # scale defines the proportion of image to erase, ratio defines aspect ratio of erased region
        train_transform_list.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)))

    base_train_transforms = transforms.Compose(train_transform_list)
    test_transforms = transforms.Compose(test_transform_list)

    # Wrap the base transform for Contrastive Learning if required
    if is_contrastive:
        train_transforms = TwoCropTransform(base_train_transforms)
    else:
        train_transforms = base_train_transforms

    # --- Load Datasets ---
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(root=valid_dir, transform=test_transforms)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)

    # --- Dataset Reduction (Few-Shot Learning) ---
    if samples_per_class is not None:
        targets = np.array(train_dataset.targets)
        classes = np.unique(targets)
        balanced_indices = []

        for c in classes:
            # Find all indices for class 'c'
            class_indices = np.where(targets == c)[0]
            # Safely sample without replacement
            n_samples = min(samples_per_class, len(class_indices))
            sampled_indices = np.random.choice(class_indices, n_samples, replace=False)
            balanced_indices.extend(sampled_indices)

        # Shuffle the final subset to mix classes in batches
        random.shuffle(balanced_indices)
        train_dataset = Subset(train_dataset, balanced_indices)
        train_dataset.classes = train_dataset.dataset.classes
        print(f"[*] Few-Shot Learning enabled: {len(balanced_indices)} total samples ({samples_per_class} per class).")

    # --- Create DataLoaders ---
    # Automatically disable pin_memory for MPS (Mac) to avoid warnings, enable for CUDA
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=use_pin_memory)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                              pin_memory=use_pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=use_pin_memory)

    return train_loader, valid_loader, test_loader
