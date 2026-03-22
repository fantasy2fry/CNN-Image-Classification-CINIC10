import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomCNN, self).__init__()
        
        # --- FEATURE EXTRACTOR --------------------------------------------------------------
        # 1st Convolutional Block
        # Input: [Batch_Size, 3, 32, 32]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)  # 3 for rgb channels
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Size after pool1: 16x16
        
        # 2nd Convolutional Block
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Size after pool2: 8x8
        
        # 3rd Convolutional Block
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # windows won't overlap, so we don't need to worry about padding here
        # Size after pool3: 4x4
        


        # --- CLASSIFIER (Fully Connected) ---------------------------------------------------
        # We need to flatten the 128 channels of 4x4 images into a single 1D vector.
        # Math: 128 channels * 4 height * 4 width = 2048 features
        self.fc1 = nn.Linear(in_features=2048, out_features=256)  # 128*4*4 = 2048
        # (According to the research plan, regularization will be added/compared later)
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)




    def forward(self, x):
        # 1. Pass through Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # 2. Pass through Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # 3. Pass through Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # 4. Flatten the tensor
        x = x.view(x.size(0), -1) 
        
        # 5. Pass through Classifier (baseline without dropout)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
