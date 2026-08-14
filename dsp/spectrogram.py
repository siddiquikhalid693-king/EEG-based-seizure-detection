"""
EEG Spectrogram Module
Computes Short-Time Fourier Transform (STFT) spectrograms for multi-channel EEG signals.
"""

import numpy as np
from scipy.signal import spectrogram
from typing import Tuple

class EEGSpectrogram:
    """Computes STFT time-frequency spectrograms for channel signal analysis."""
    
    @staticmethod
    def compute_spectrogram(data_channel: np.ndarray, fs: int = 256, nperseg: int = 128, noverlap: int = 112) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes STFT spectrogram for a single channel.
        data_channel: 1D numpy array of EEG samples
        Returns:
            f: Frequencies array (Hz)
            t: Segment times array (seconds)
            Sxx: Power Spectral Density 2D matrix (f_bins, t_bins)
        """
        f, t, Sxx = spectrogram(data_channel, fs=fs, nperseg=nperseg, noverlap=noverlap)
        # Filter frequencies up to 50 Hz for clean clinical display
        idx = f <= 50.0
        f_filtered = f[idx]
        Sxx_filtered = Sxx[idx, :]
        
        # Log scale conversion for better contrast
        Sxx_log = 10 * np.log10(Sxx_filtered + 1e-10)
        return f_filtered, t, Sxx_log
