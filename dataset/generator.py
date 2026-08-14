"""
EEG Dataset & Signal Generator Module
Generates realistic multi-channel 10-20 montage EEG signals for normal and epileptic states:
- Normal Awake / Sleep Background
- Interictal Spike Discharges
- Pre-ictal Seizure Onset
- Absence Seizure (Generalized 3Hz Spike-Wave)
- Temporal Lobe Focal Seizure (Evolving Rhythmic Discharge)
"""

import numpy as np
from typing import Dict, List, Tuple

# Standard 10-20 Montage Channels
CHANNELS = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'F3-C3',  'C3-P3',  'F4-C4',  'C4-P4',
    'Fz-Cz',  'Cz-Pz'
]

# Sampling Rate (Hz)
FS = 256

# Class Labels
CLASS_NAMES = {
    0: 'NORMAL',
    1: 'INTERICTAL_SPIKE',
    2: 'PREICTAL',
    3: 'ICTAL_SEIZURE'
}

class EEGDataGenerator:
    def __init__(self, fs: int = FS, channels: List[str] = None):
        self.fs = fs
        self.channels = channels if channels is not None else CHANNELS
        self.n_channels = len(self.channels)

    def _generate_background_noise(self, n_samples: int) -> np.ndarray:
        """Pink/colored noise for baseline EEG background."""
        # White noise
        white = np.random.randn(self.n_channels, n_samples)
        # Apply 1/f spectral shaping via cumulative sum/filter
        b, a = [0.05], [1, -0.95]
        from scipy.signal import lfilter
        colored = lfilter(b, a, white, axis=1)
        return colored * 5.0  # ~5 microvolts baseline

    def generate_normal_eeg(self, duration_sec: float = 10.0) -> Tuple[np.ndarray, int]:
        """Generates normal awake EEG with dominant posterior alpha rhythm (8-13 Hz)."""
        n_samples = int(duration_sec * self.fs)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)
        eeg = self._generate_background_noise(n_samples)

        for i, ch in enumerate(self.channels):
            # Alpha rhythm (10 Hz) stronger in occipital channels (T5-O1, T6-O2, C3-P3, C4-P4)
            alpha_amp = 25.0 if ('O1' in ch or 'O2' in ch or 'P' in ch) else 10.0
            alpha_freq = 10.0 + np.random.uniform(-0.5, 0.5)
            eeg[i] += alpha_amp * np.sin(2 * np.pi * alpha_freq * t + np.random.uniform(0, 2*np.pi))

            # Beta rhythm (18 Hz) in frontal channels
            if 'Fp' in ch or 'F' in ch:
                eeg[i] += 8.0 * np.sin(2 * np.pi * 18.0 * t + np.random.uniform(0, 2*np.pi))

        return eeg, 0  # Label 0: NORMAL

    def generate_interictal_spikes(self, duration_sec: float = 10.0) -> Tuple[np.ndarray, int]:
        """Generates background EEG with paroxysmal sharp spikes (Interictal)."""
        eeg, _ = self.generate_normal_eeg(duration_sec)
        n_samples = eeg.shape[1]

        # Insert 3-5 random spike-wave transients
        num_spikes = np.random.randint(3, 6)
        spike_focus_channels = [1, 2]  # F7-T3, T3-T5 temporal channels

        for _ in range(num_spikes):
            center_sample = np.random.randint(self.fs, n_samples - self.fs)
            spike_width = int(0.08 * self.fs)  # 80ms sharp spike
            t_spike = np.linspace(-np.pi, np.pi, spike_width)
            
            # Sharp spike waveform + slow wave overshoot
            spike_wave = 150.0 * np.cos(t_spike) - 50.0 * np.sin(0.5 * t_spike)

            for ch_idx in spike_focus_channels:
                start = center_sample - spike_width // 2
                end = start + spike_width
                if 0 <= start and end < n_samples:
                    eeg[ch_idx, start:end] += spike_wave

        return eeg, 1  # Label 1: INTERICTAL_SPIKE

    def generate_preictal_eeg(self, duration_sec: float = 10.0) -> Tuple[np.ndarray, int]:
        """Generates pre-ictal prodromal transition state (high frequency gamma/beta emergence)."""
        n_samples = int(duration_sec * self.fs)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)
        eeg = self._generate_background_noise(n_samples)

        for i, ch in enumerate(self.channels):
            # Gradual buildup of high frequency beta/gamma activity (25-40 Hz)
            ramp = np.linspace(0.5, 2.5, n_samples)
            gamma = 18.0 * ramp * np.sin(2 * np.pi * 32.0 * t)
            beta = 15.0 * ramp * np.sin(2 * np.pi * 22.0 * t)
            eeg[i] += gamma + beta

        return eeg, 2  # Label 2: PREICTAL

    def generate_absence_seizure(self, duration_sec: float = 10.0) -> Tuple[np.ndarray, int]:
        """Generates generalized 3 Hz spike-and-wave discharge (Absence Seizure)."""
        n_samples = int(duration_sec * self.fs)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)
        eeg = self._generate_background_noise(n_samples)

        # Rhythmic 3 Hz fundamental with sharp harmonics
        freq = 3.0
        spike_wave_base = (
            220.0 * np.sin(2 * np.pi * freq * t) +
            110.0 * np.sin(2 * np.pi * (2 * freq) * t) +
            70.0 * np.sin(2 * np.pi * (3 * freq) * t)
        )

        # Synchronous across ALL channels (Generalized)
        for i in range(self.n_channels):
            channel_factor = np.random.uniform(0.85, 1.15)
            eeg[i] += spike_wave_base * channel_factor

        return eeg, 3  # Label 3: ICTAL_SEIZURE

    def generate_focal_temporal_seizure(self, duration_sec: float = 10.0) -> Tuple[np.ndarray, int]:
        """Generates focal temporal lobe seizure (evolving rhythmic theta/delta 4-6 Hz)."""
        n_samples = int(duration_sec * self.fs)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)
        eeg = self._generate_background_noise(n_samples)

        # Primary focus: Left Temporal F7-T3, T3-T5
        focal_channels = [1, 2]
        
        # Frequency evolution: starts 6Hz fast theta, slows to 3.5Hz high amplitude delta
        freq_envelope = np.linspace(6.5, 3.5, n_samples)
        phase = 2 * np.pi * np.cumsum(freq_envelope) / self.fs
        
        amplitude_envelope = np.sin(np.pi * t / duration_sec) ** 0.5
        seizure_wave = 180.0 * amplitude_envelope * (np.sin(phase) + 0.4 * np.sin(2 * phase))

        for i in range(self.n_channels):
            if i in focal_channels:
                eeg[i] += seizure_wave
            elif i in [0, 3, 8]:  # Neighboring channels receive partial recruitment
                eeg[i] += seizure_wave * 0.4

        return eeg, 3  # Label 3: ICTAL_SEIZURE

    def generate_dataset(self, n_samples_per_class: int = 100, epoch_sec: float = 4.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a balanced dataset of multi-channel EEG epochs for model training.
        Returns:
            X: Shape (N, n_channels, n_time_points)
            y: Shape (N,) class indices
        """
        X_list = []
        y_list = []

        generators = [
            self.generate_normal_eeg,
            self.generate_interictal_spikes,
            self.generate_preictal_eeg,
            self.generate_absence_seizure,
            self.generate_focal_temporal_seizure
        ]

        for gen in generators:
            # Map generator function to class label
            for _ in range(n_samples_per_class // (2 if gen in [self.generate_absence_seizure, self.generate_focal_temporal_seizure] else 1)):
                data, label = gen(duration_sec=epoch_sec)
                X_list.append(data)
                y_list.append(label)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)

        # Shuffle dataset
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        return X[indices], y[indices]


if __name__ == '__main__':
    gen = EEGDataGenerator()
    X, y = gen.generate_dataset(n_samples_per_class=10, epoch_sec=4.0)
    print(f"Generated EEG Dataset X shape: {X.shape}, y shape: {y.shape}")
    print(f"Classes balance: {np.bincount(y)}")
