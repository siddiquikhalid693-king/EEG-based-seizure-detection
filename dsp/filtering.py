"""
EEG Digital Filtering Module
Applies bandpass filtering (0.5 - 70 Hz) and Notch filtering (50/60 Hz mains powerline interference).
"""

import numpy as np
from scipy.signal import butter, iirnotch, filtfilt

class EEGFilter:
    """Applies Butterworth bandpass and IIR notch filters to multi-channel EEG signals."""
    
    @staticmethod
    def bandpass_filter(data: np.ndarray, lowcut: float = 0.5, highcut: float = 70.0, fs: int = 256, order: int = 4) -> np.ndarray:
        """
        Applies a zero-phase 4th order Butterworth bandpass filter.
        data: Shape (n_channels, n_samples) or (n_samples,)
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='bandpass')
        
        if data.ndim == 1:
            return filtfilt(b, a, data)
        else:
            filtered = np.zeros_like(data)
            for i in range(data.shape[0]):
                filtered[i] = filtfilt(b, a, data[i])
            return filtered

    @staticmethod
    def notch_filter(data: np.ndarray, notch_freq: float = 50.0, fs: int = 256, q: float = 30.0) -> np.ndarray:
        """Applies zero-phase IIR notch filter to remove 50Hz / 60Hz powerline noise."""
        nyq = 0.5 * fs
        freq = notch_freq / nyq
        b, a = iirnotch(freq, q)
        
        if data.ndim == 1:
            return filtfilt(b, a, data)
        else:
            filtered = np.zeros_like(data)
            for i in range(data.shape[0]):
                filtered[i] = filtfilt(b, a, data[i])
            return filtered

    @classmethod
    def process_eeg(cls, data: np.ndarray, fs: int = 256, lowcut: float = 0.5, highcut: float = 70.0, notch_freq: float = 50.0) -> np.ndarray:
        """Applies full preprocessing pipeline: Notch filter followed by Bandpass filter."""
        notched = cls.notch_filter(data, notch_freq=notch_freq, fs=fs)
        bandpassed = cls.bandpass_filter(notched, lowcut=lowcut, highcut=highcut, fs=fs)
        return bandpassed
