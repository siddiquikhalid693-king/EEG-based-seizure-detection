"""
EEG Feature Extraction Module
Extracts spectral band powers, Hjorth parameters, line length, and energy features from EEG signals.
"""

import numpy as np
from scipy.signal import welch
from typing import Dict, List

FREQ_BANDS = {
    'Delta': (0.5, 4.0),
    'Theta': (4.0, 8.0),
    'Alpha': (8.0, 13.0),
    'Beta':  (13.0, 30.0),
    'Gamma': (30.0, 50.0)
}

class EEGFeatureExtractor:
    """Extracts clinical and diagnostic EEG features from multi-channel signal epochs."""
    
    @staticmethod
    def calculate_band_powers(data: np.ndarray, fs: int = 256) -> Dict[str, np.ndarray]:
        """
        Computes absolute and relative band power using Welch's Method for each channel.
        data: Shape (n_channels, n_samples)
        Returns: Dict mapping band name -> array of band powers per channel (shape n_channels)
        """
        n_channels = data.shape[0] if data.ndim == 2 else 1
        data_2d = data if data.ndim == 2 else data.reshape(1, -1)
        
        band_powers = {band: np.zeros(n_channels) for band in FREQ_BANDS}
        total_powers = np.zeros(n_channels)
        
        for ch in range(n_channels):
            freqs, psd = welch(data_2d[ch], fs=fs, nperseg=min(data_2d.shape[1], fs * 2))
            total_powers[ch] = np.sum(psd)
            
            for band, (f_min, f_max) in FREQ_BANDS.items():
                idx = np.logical_and(freqs >= f_min, freqs <= f_max)
                band_powers[band][ch] = np.sum(psd[idx])
                
        # Calculate relative band powers (normalized percentage)
        relative_powers = {}
        for band in FREQ_BANDS:
            # Avoid divide by zero
            safe_total = np.where(total_powers == 0, 1.0, total_powers)
            relative_powers[band] = (band_powers[band] / safe_total) * 100.0
            
        return relative_powers

    @staticmethod
    def calculate_hjorth_parameters(data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculates Hjorth Parameters (Activity, Mobility, Complexity) for each channel.
        Activity = var(y(t))
        Mobility = sqrt(var(dy/dt) / var(y(t)))
        Complexity = Mobility(dy/dt) / Mobility(y(t))
        """
        data_2d = data if data.ndim == 2 else data.reshape(1, -1)
        n_channels = data_2d.shape[0]
        
        activity = np.zeros(n_channels)
        mobility = np.zeros(n_channels)
        complexity = np.zeros(n_channels)
        
        for ch in range(n_channels):
            y = data_2d[ch]
            dy = np.diff(y)
            ddy = np.diff(dy)
            
            var_y = np.var(y)
            var_dy = np.var(dy)
            var_ddy = np.var(ddy)
            
            activity[ch] = var_y
            
            mob_y = np.sqrt(var_dy / var_y) if var_y > 0 else 0.0
            mobility[ch] = mob_y
            
            mob_dy = np.sqrt(var_ddy / var_dy) if var_dy > 0 else 0.0
            complexity[ch] = (mob_dy / mob_y) if mob_y > 0 else 0.0
            
        return {
            'Activity': activity,
            'Mobility': mobility,
            'Complexity': complexity
        }

    @staticmethod
    def calculate_line_length(data: np.ndarray) -> np.ndarray:
        """
        Calculates Line Length feature (useful indicator of epileptic spike discharge energy).
        Line Length = sum(|y(t+1) - y(t)|)
        """
        data_2d = data if data.ndim == 2 else data.reshape(1, -1)
        return np.sum(np.abs(np.diff(data_2d, axis=1)), axis=1)

    @classmethod
    def extract_epoch_features(cls, data: np.ndarray, fs: int = 256) -> np.ndarray:
        """
        Combines all DSP features into a single feature vector per channel.
        data: Shape (n_channels, n_samples)
        Returns: Flattened feature vector (n_channels * num_features)
        """
        rel_powers = cls.calculate_band_powers(data, fs)
        hjorth = cls.calculate_hjorth_parameters(data)
        line_len = cls.calculate_line_length(data)
        
        features_list = []
        for band in FREQ_BANDS:
            features_list.append(rel_powers[band])
        features_list.append(hjorth['Activity'])
        features_list.append(hjorth['Mobility'])
        features_list.append(hjorth['Complexity'])
        features_list.append(line_len)
        
        # Stack all channel features: shape (num_features, n_channels) -> flatten
        stacked = np.stack(features_list, axis=0).T  # shape: (n_channels, num_features)
        return stacked.flatten()
