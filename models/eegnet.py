"""
EEGNet PyTorch Model Architecture
A Compact Convolutional Neural Network for EEG-based Signal Classification.
Reference: Lawhern et al., "EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces" (2018).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dWithConstraint(nn.Conv2d):
    """Custom 2D Convolution with max-norm constraint on weights."""
    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        if self.max_norm is not None:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)

class EEGNet(nn.Module):
    """
    EEGNet architecture for multi-channel EEG epoch classification.
    Inputs shape: (batch_size, 1, n_channels, n_samples)
    Outputs shape: (batch_size, n_classes)
    """
    def __init__(
        self,
        n_classes: int = 4,
        n_channels: int = 14,
        n_samples: int = 1024,
        fs: int = 256,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout_rate: float = 0.25
    ):
        super(EEGNet, self).__init__()
        
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_samples = n_samples

        # Block 1: Temporal Conv -> Depthwise Spatial Conv
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Spatial Depthwise Conv (filters across channels)
        self.depthwiseConv = Conv2dWithConstraint(
            F1, F1 * D, (n_channels, 1),
            groups=F1,
            bias=False,
            max_norm=1.0
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pooling1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # Block 2: Separable Convolution
        self.separableConv = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.pooling2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)

        # Calculate flattened linear size dynamically
        out_samples = n_samples // 32
        self.flatten_size = F2 * out_samples

        # Final Dense Classifier
        self.fc = nn.Linear(self.flatten_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x input shape: (batch, n_channels, n_samples) -> reshape to (batch, 1, n_channels, n_samples)
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwiseConv(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pooling1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.separableConv(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pooling2(x)
        x = self.dropout2(x)

        # Flatten & Linear Classification
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
