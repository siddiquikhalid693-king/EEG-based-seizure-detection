"""
EDF & CSV File Importer Module
Parses standard European Data Format (.edf) and CSV EEG recordings into structured multi-channel arrays.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import io

class EEGFileParser:
    """Parses EDF / CSV format EEG signals for processing by the epilepsy detection engine."""
    
    @staticmethod
    def parse_csv(file_obj_or_path, fs_default: int = 256) -> Tuple[np.ndarray, List[str], int]:
        """
        Parses a CSV file containing EEG channel columns.
        Expected format: columns as channel names, rows as time samples (or 'time' column).
        """
        df = pd.read_csv(file_obj_or_path)
        
        # Check if time column exists
        if 'time' in df.columns.str.lower():
            time_col = [c for c in df.columns if c.lower() == 'time'][0]
            t = df[time_col].values
            if len(t) > 1:
                fs = int(round(1.0 / np.mean(np.diff(t))))
            else:
                fs = fs_default
            df = df.drop(columns=[time_col])
        else:
            fs = fs_default
            
        channel_names = list(df.columns)
        data = df.values.T.astype(np.float32)  # Shape: (channels, samples)
        return data, channel_names, fs

    @staticmethod
    def parse_edf(file_path: str) -> Tuple[np.ndarray, List[str], int]:
        """Parses EDF file using PyEDFlib or MNE if available."""
        try:
            import mne
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            data = raw.get_data(units='uV').astype(np.float32)
            channel_names = raw.ch_names
            fs = int(raw.info['sfreq'])
            return data, channel_names, fs
        except Exception as e:
            raise RuntimeError(f"Error parsing EDF file: {e}")
