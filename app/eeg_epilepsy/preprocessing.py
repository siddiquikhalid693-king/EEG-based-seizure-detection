from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


@dataclass(frozen=True)
class PreprocessingConfig:
    sampling_rate_hz: int = 256
    low_cut_hz: float = 0.5
    high_cut_hz: float = 40.0
    epoch_seconds: float = 2.0


def bandpass_filter(signal: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    nyquist = 0.5 * config.sampling_rate_hz
    low = config.low_cut_hz / nyquist
    high = config.high_cut_hz / nyquist
    if not 0 < low < high < 1:
        raise ValueError("Invalid bandpass frequencies for the sampling rate")
    b, a = butter(4, [low, high], btype="band")
    return filtfilt(b, a, signal, axis=0)


def epoch_signal(signal: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    samples_per_epoch = int(config.sampling_rate_hz * config.epoch_seconds)
    if samples_per_epoch <= 0:
        raise ValueError("epoch_seconds must produce at least one sample")
    usable = (len(signal) // samples_per_epoch) * samples_per_epoch
    if usable == 0:
        raise ValueError("Signal is shorter than one epoch")
    trimmed = signal[:usable]
    return trimmed.reshape(-1, samples_per_epoch, *signal.shape[1:])


def zscore_normalize(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std == 0, 1, std)
    return (values - mean) / std


def extract_epoch_features(epochs: np.ndarray) -> pd.DataFrame:
    flat_epochs = epochs.reshape(epochs.shape[0], -1)
    features = {
        "mean": flat_epochs.mean(axis=1),
        "std": flat_epochs.std(axis=1),
        "min": flat_epochs.min(axis=1),
        "max": flat_epochs.max(axis=1),
        "median": np.median(flat_epochs, axis=1),
        "energy": np.square(flat_epochs).mean(axis=1),
    }
    return pd.DataFrame(features)


def load_numeric_csv(path: str | Path, label_column: str | None = None) -> tuple[pd.DataFrame, pd.Series | None]:
    frame = pd.read_csv(path)
    labels = None
    if label_column and label_column in frame.columns:
        labels = frame[label_column]
        frame = frame.drop(columns=[label_column])
    numeric = frame.select_dtypes(include=["number"])
    if numeric.empty:
        raise ValueError("No numeric EEG columns were found in the CSV file")
    return numeric, labels
