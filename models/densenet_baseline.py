import torch
import torch.nn as nn
from torchvision.models import densenet121

class DenseNet121_CIFAR10(nn.Module):
    """
    DenseNet121 model optimized for CIFAR-10 (32x32) input.

    - Modifies the initial convolutional layer to better handle smaller image sizes.
    - Replaces the final classifier to match the number of CIFAR-10 classes.
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Load a DenseNet121 model pretrained on ImageNet
        self.densenet = densenet121(weights='IMAGENET1K_V1')
        
        # Modify the first conv layer for 32x32 input instead of the original 224x224
        self.densenet.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Replace the final classifier layer for the 10 classes of CIFAR-10
        self.densenet.classifier = nn.Linear(1024, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.
        Returns logits.
        """
        return self.densenet(x)