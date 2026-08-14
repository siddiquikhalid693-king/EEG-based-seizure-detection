"""
CNN-LSTM Hybrid PyTorch Model Architecture
Combines 1D Temporal Convolutions (Spatial-Temporal Feature Extraction) with Bidirectional LSTM (Long-term Temporal Dynamics).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM(nn.Module):
    """
    CNN-LSTM Hybrid Neural Network for EEG seizure detection.
    Input shape: (batch_size, n_channels, n_samples)
    Output shape: (batch_size, n_classes)
    """
    def __init__(
        self,
        n_classes: int = 4,
        n_channels: int = 14,
        n_samples: int = 1024,
        lstm_hidden_dim: int = 64,
        lstm_layers: int = 2
    ):
        super(CNN_LSTM, self).__init__()

        # 1D Convolutional Blocks
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.dropout = nn.Dropout(0.3)

        # Bidirectional LSTM Sequence Encoder
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        # FC Classifier
        self.fc1 = nn.Linear(lstm_hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x input shape: (batch, n_channels, n_samples)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)

        # Reshape for LSTM: (batch, channels, seq_len) -> (batch, seq_len, channels)
        x = x.transpose(1, 2)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch, seq_len, lstm_hidden_dim * 2)
        
        # Take mean over sequence length dimension
        pooled = torch.mean(lstm_out, dim=1)
        
        # Dense classification
        out = F.relu(self.fc1(pooled))
        out = self.fc2(out)
        return out
