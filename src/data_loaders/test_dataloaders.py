import os
import sys
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

# Add the project root directory to Python's path so the "Run" button works
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the dataloader function from your previous file
# (Make sure the previous code is saved as data_loader.py in the same folder)
from src.utils import get_cinic10_dataloaders, set_seed


def denormalize_image(tensor_img):
    """
    Reverses the normalization process so the image can be displayed correctly.
    """
    cinic_mean = np.array([0.47889522, 0.4722784, 0.43047404])
    cinic_std = np.array([0.24205776, 0.23828046, 0.25874835])

    # PyTorch tensors are [Channels, Height, Width]
    # Matplotlib expects [Height, Width, Channels], so we need to permute
    img = tensor_img.numpy().transpose((1, 2, 0))

    # Reverse the normalization: image = (image * std) + mean
    img = img * cinic_std + cinic_mean

    # Ensure all values are strictly between 0 and 1 for matplotlib
    img = np.clip(img, 0, 1)

    return img


def show_batch_images(dataloader, num_images=8):
    """
    Fetches a single batch from the dataloader and plots a few images.
    """
    # Get one batch of training data
    images, labels = next(iter(dataloader))

    # Get the class names (e.g., 'airplane', 'dog', etc.)
    class_names = dataloader.dataset.classes

    # Set up the matplotlib figure
    fig = plt.figure(figsize=(12, 6))

    for i in range(num_images):
        ax = fig.add_subplot(2, int(num_images / 2), i + 1, xticks=[], yticks=[])

        # Denormalize and format the image
        img_to_show = denormalize_image(images[i])

        # Display the image
        ax.imshow(img_to_show)

        # Set the title as the actual class name
        class_idx = labels[i].item()
        ax.set_title(class_names[class_idx])

    plt.tight_layout()
    plt.show()


# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":

    # Ensure reproducibility
    set_seed(42)

    # Path to your dataset
    # We use os.path.join with the file's current location to always find the data folder
    # no matter where we run the script from.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(current_dir, "..", "..", "data")

    if os.path.exists(DATA_PATH):
        print("Loading DataLoaders...")

        # We load the standard train dataloader
        # Note: We set shuffle=True in the dataloader, so every time you run this,
        # you will get a different random batch of images.
        train_loader, valid_loader, test_loader = get_cinic10_dataloaders(
            data_dir=DATA_PATH,
            batch_size=32,  # Smaller batch size just for visualization
            num_workers=2,
            use_cutout=False  # Turn off cutout just to see original images clearly
        )

        print("Displaying a batch of images...")
        show_batch_images(train_loader, num_images=8)

    else:
        print(f"Error: Dataset folder '{DATA_PATH}' not found.")