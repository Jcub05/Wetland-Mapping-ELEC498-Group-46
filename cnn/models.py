import torch
import torch.nn as nn
import torchvision.models as models
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class PixelMLP(nn.Module):
    """
    Baseline deep model for per-pixel embeddings: input (B, 64) -> logits (B, 6)
    This is the correct starting point for your current NPZ dataset shape.
    """
    def __init__(self, in_features: int = 64, num_classes: int = 6, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class WetlandCNN15(nn.Module):
    """
    2D CNN optimized for 15x15 wetland patches.
    Expected input shape: (B, 64, 15, 15)
    """
    def __init__(self, in_channels: int = 64, num_classes: int = 6, dropout: float = 0.3):
        super().__init__()
        
        # Block 1 (15x15 -> 7x7)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        
        # Block 2 (7x7 -> 3x3)
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1), 
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        
        # Block 3 (3x3 -> 1x1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2), 
            nn.AdaptiveAvgPool2d((1, 1)) # Safely pools exactly to 1x1
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.classifier(x)

class ResNet18Wetland(nn.Module):
    """
    ResNet-18 architecture fine-tuned for 64-channel 15x15 wetland patches.
    Uses pretrained weights on subsequent layers to extract advanced spatial features.
    """
    def __init__(self, in_channels: int = 64, num_classes: int = 6, dropout: float = 0.3):
        super().__init__()
        
        # Load pre-trained ResNet-18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Modify the first Convolutional Layer to accept 64 channels instead of 3 (RGB)
        # We retain the original geometry parameters of ResNet's first layer.
        original_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            in_channels, 
            original_conv.out_channels, 
            kernel_size=original_conv.kernel_size, 
            stride=original_conv.stride, 
            padding=original_conv.padding, 
            bias=False
        )
        
        # Initialize the new 64-channel conv1 weights dynamically
        nn.init.kaiming_normal_(self.resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
        
        # Modify the fully connected output layer for 6 classes, adding regularization
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_ftrs, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)