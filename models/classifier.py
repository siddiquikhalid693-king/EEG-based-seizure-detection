"""
Unified EEG Epilepsy Classifier & Inference Wrapper
Loads trained Deep Learning PyTorch models (EEGNet / CNN-LSTM) and executes real-time sliding epoch inference.
"""

import os
import torch
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataset.generator import CLASS_NAMES, CHANNELS
from models.eegnet import EEGNet
from models.cnn_lstm import CNN_LSTM
from dsp.features import EEGFeatureExtractor

class SeizureClassifierWrapper:
    """Inference wrapper for EEG epoch classification and seizure localization."""
    
    def __init__(self, model_type: str = 'eegnet', weights_path: Optional[str] = None):
        self.model_type = model_type.lower()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.n_classes = 4
        self.n_channels = len(CHANNELS)
        self.n_samples = 1024  # 4 seconds at 256Hz
        
        if self.model_type == 'eegnet':
            self.model = EEGNet(n_classes=self.n_classes, n_channels=self.n_channels, n_samples=self.n_samples)
        else:
            self.model = CNN_LSTM(n_classes=self.n_classes, n_channels=self.n_channels, n_samples=self.n_samples)
            
        self.model.to(self.device)
        self.model.eval()

        # Load weights if present
        if weights_path and os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"Successfully loaded PyTorch weights from {weights_path}")
            except Exception as e:
                print(f"Warning: Could not load weights from {weights_path}: {e}")

    def predict_epoch(self, epoch_data: np.ndarray) -> Dict:
        """
        Classifies a single EEG epoch.
        epoch_data: Shape (n_channels, n_samples)
        Returns dictionary with predicted state, probabilities, confidence, and focal channels.
        """
        # Ensure exact shape (n_channels, n_samples)
        if epoch_data.shape[1] > self.n_samples:
            epoch_data = epoch_data[:, :self.n_samples]
        elif epoch_data.shape[1] < self.n_samples:
            # Pad with zeros if necessary
            pad = np.zeros((self.n_channels, self.n_samples - epoch_data.shape[1]), dtype=np.float32)
            epoch_data = np.hstack([epoch_data, pad])

        # PyTorch Tensor conversion
        tensor_data = torch.from_numpy(epoch_data.astype(np.float32)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_data)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx] * 100.0)

        # Spatial localization calculation (channel-wise line length & amplitude energy)
        line_lengths = EEGFeatureExtractor.calculate_line_length(epoch_data)
        top_channels_idx = np.argsort(line_lengths)[::-1][:3]
        focal_channels = [CHANNELS[i] for i in top_channels_idx if i < len(CHANNELS)]

        return {
            'predicted_class_idx': pred_idx,
            'predicted_label': CLASS_NAMES[pred_idx],
            'confidence': confidence,
            'probabilities': {CLASS_NAMES[i]: float(probs[i] * 100.0) for i in range(self.n_classes)},
            'focal_channels': focal_channels,
            'is_seizure': (pred_idx == 3)
        }
