"""
Multi-Channel EEG Plotter Module
Generates clean stacked multi-channel EEG waveform interactive figures using Plotly.
"""

import numpy as np
import plotly.graph_objects as go
from typing import List, Optional, Dict

class EEGPlotter:
    """Generates stacked multi-channel EEG waveform plots with highlight overlays for seizure events."""
    
    @staticmethod
    def create_eeg_figure(
        eeg_data: np.ndarray,
        channel_names: List[str],
        fs: int = 256,
        time_window_sec: Optional[float] = None,
        start_sec: float = 0.0,
        gain: float = 1.0,
        highlights: Optional[List[Dict]] = None
    ) -> go.Figure:
        """
        Creates an interactive Plotly figure for multi-channel EEG signals.
        eeg_data: Shape (n_channels, n_samples)
        """
        n_channels, n_samples = eeg_data.shape
        t = np.arange(n_samples) / fs + start_sec

        fig = go.Figure()
        
        # Calculate channel offset spacing based on signal variance
        std_val = np.std(eeg_data) if np.std(eeg_data) > 0 else 50.0
        offset_step = 100.0 * (1.0 / gain)

        # Plot channels from top to bottom
        for i in range(n_channels):
            ch_name = channel_names[i] if i < len(channel_names) else f"Ch {i+1}"
            offset = (n_channels - 1 - i) * offset_step
            signal_offset = eeg_data[i] * gain + offset

            fig.add_trace(go.Scatter(
                x=t,
                y=signal_offset,
                mode='lines',
                name=ch_name,
                line=dict(width=1.2, color='#1E293B'),
                hoverinfo='x+name'
            ))

        # Y-axis ticks and labels
        yticks = [(n_channels - 1 - i) * offset_step for i in range(n_channels)]
        yticklabels = [channel_names[i] if i < len(channel_names) else f"Ch {i+1}" for i in range(n_channels)]

        # Highlight Seizure Epochs if provided
        if highlights:
            for h in highlights:
                t0, t1 = h.get('start_sec', 0), h.get('end_sec', 0)
                color = h.get('color', 'rgba(239, 68, 68, 0.25)')  # Default red alert
                label = h.get('label', 'Seizure Alert')
                fig.add_vrect(
                    x0=t0, x1=t1,
                    fillcolor=color, opacity=0.3,
                    layer="below", line_width=1,
                    annotation_text=label, annotation_position="top left"
                )

        fig.update_layout(
            title="<b>Multi-Channel EEG Signal Monitor (10-20 System)</b>",
            xaxis_title="Time (seconds)",
            yaxis=dict(
                tickmode='array',
                tickvals=yticks,
                ticktext=yticklabels,
                showgrid=True,
                gridcolor='#E2E8F0',
                zeroline=False
            ),
            xaxis=dict(showgrid=True, gridcolor='#E2E8F0'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=60, r=30, t=50, b=50),
            height=max(450, n_channels * 40),
            showlegend=False
        )

        return fig
