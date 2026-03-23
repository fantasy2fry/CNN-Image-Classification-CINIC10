import torch.nn as nn

class CustomCNN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.0):
        super(CustomCNN, self).__init__()
        
        self.features = nn.Sequential(
            # --- FEATURE EXTRACTOR --------------------------------------------------------------
            # 1st Convolutional Block
            # Input: [Batch_Size, 3, 32, 32]
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),  # 3 for rgb channels
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Size after pool1: 16x16
            
            # 2nd Convolutional Block
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Size after pool2: 8x8
            
            # 3rd Convolutional Block
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # windows won't overlap, so we don't need to worry about padding here
            # Size after pool3: 4x4    
        )

        # --- CLASSIFIER (Fully Connected) ---------------------------------------------------
        # We need to flatten the 128 channels of 4x4 images into a single 1D vector.
        # Math: 128 channels * 4 height * 4 width = 2048 features
        self.classifier = nn.Sequential(
            nn.Linear(in_features=2048, out_features=256),  # 128*4*4 = 2048
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=256, out_features=num_classes)
        )



    def forward(self, x):
        x = self.features(x)  # Pass through the convolutional feature extractor
        # 4. Flatten the tensor
        x = x.view(x.size(0), -1) # Reshape to [Batch_Size, 2048]
        # 5. Pass through Classifier (baseline without dropout)
        x = self.classifier(x)

        return x
