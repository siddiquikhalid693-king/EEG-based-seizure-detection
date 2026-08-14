"""
2D Scalp Brain Topography Module
Maps 10-20 system electrode potentials and band power onto a 2D scalp heatmap with head outline.
"""

import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import Rbf
from typing import Dict, List, Optional

# Standard 10-20 System 2D Normalized Coordinates (x, y)
ELECTRODE_POS_2D = {
    'Fp1': (-0.30,  0.80), 'Fp2': ( 0.30,  0.80),
    'F7':  (-0.70,  0.50), 'F3':  (-0.35,  0.45), 'Fz': ( 0.00,  0.50), 'F4': ( 0.35,  0.45), 'F8': ( 0.70,  0.50),
    'T3':  (-0.85,  0.00), 'C3':  (-0.40,  0.00), 'Cz': ( 0.00,  0.00), 'C4': ( 0.40,  0.00), 'T4': ( 0.85,  0.00),
    'T5':  (-0.70, -0.50), 'P3':  (-0.35, -0.45), 'Pz': ( 0.00, -0.50), 'P4': ( 0.35, -0.45), 'T6': ( 0.70, -0.50),
    'O1':  (-0.30, -0.80), 'O2':  ( 0.30, -0.80)
}

# Map differential montage channels to closest electrode center
MONTAGE_TO_ELECTRODE = {
    'Fp1-F7': 'F7',
    'F7-T3':  'T3',
    'T3-T5':  'T5',
    'T5-O1':  'O1',
    'Fp2-F8': 'F8',
    'F8-T4':  'T4',
    'T4-T6':  'T6',
    'T6-O2':  'O2',
    'F3-C3':  'F3',
    'C3-P3':  'P3',
    'F4-C4':  'F4',
    'C4-P4':  'P4',
    'Fz-Cz':  'Fz',
    'Cz-Pz':  'Cz'
}

class ScalpTopography:
    """Generates 2D Scalp Brain Heatmap visualizations from EEG electrode values."""
    
    @staticmethod
    def create_scalp_heatmap(channel_values: Dict[str, float], title: str = "2D Scalp Voltage Topography") -> go.Figure:
        """
        Creates a 2D scalp topography plot with interpolated potential heatmap and head outline.
        channel_values: Dict mapping channel name (or electrode name) to scalar amplitude/power value.
        """
        # Collect x, y coordinates and corresponding values
        elec_x, elec_y, vals, labels = [], [], [], []
        
        for name, val in channel_values.items():
            elec_name = MONTAGE_TO_ELECTRODE.get(name, name)
            if elec_name in ELECTRODE_POS_2D:
                x, y = ELECTRODE_POS_2D[elec_name]
                elec_x.append(x)
                elec_y.append(y)
                vals.append(val)
                labels.append(elec_name)

        if len(vals) < 3:
            # Fallback if insufficient channels mapped
            fig = go.Figure()
            fig.add_annotation(text="Insufficient 10-20 channels mapped", showarrow=False)
            return fig

        # 2D Grid Interpolation using Radial Basis Function (Rbf)
        grid_x, grid_y = np.mgrid[-1.0:1.0:100j, -1.0:1.0:100j]
        rbf = Rbf(elec_x, elec_y, vals, function='multiquadric', smooth=0.1)
        grid_z = rbf(grid_x, grid_y)

        # Mask points outside head circle (radius = 0.95)
        dist_from_center = np.sqrt(grid_x**2 + grid_y**2)
        grid_z[dist_from_center > 0.95] = np.nan

        fig = go.Figure()

        # 1. Scalp Topographic Heatmap Contour
        fig.add_trace(go.Contour(
            z=grid_z.T,
            x=np.linspace(-1.0, 1.0, 100),
            y=np.linspace(-1.0, 1.0, 100),
            colorscale='Jet',
            contours=dict(coloring='heatmap', showlines=False),
            colorbar=dict(title="uV / Power"),
            hoverinfo='skip'
        ))

        # 2. Outer Head Circle Outline
        theta = np.linspace(0, 2 * np.pi, 200)
        head_x = 0.95 * np.cos(theta)
        head_y = 0.95 * np.sin(theta)
        fig.add_trace(go.Scatter(
            x=head_x, y=head_y, mode='lines',
            line=dict(color='black', width=3), hoverinfo='skip', showlegend=False
        ))

        # 3. Nose Triangle Indicator (Top)
        nose_x = [-0.1, 0.0, 0.1]
        nose_y = [0.95, 1.08, 0.95]
        fig.add_trace(go.Scatter(
            x=nose_x, y=nose_y, mode='lines',
            line=dict(color='black', width=3), hoverinfo='skip', showlegend=False
        ))

        # 4. Left & Right Ears Indicators
        left_ear_x = [-0.95, -1.02, -0.95]
        left_ear_y = [0.15, 0.0, -0.15]
        right_ear_x = [0.95, 1.02, 0.95]
        right_ear_y = [0.15, 0.0, -0.15]
        fig.add_trace(go.Scatter(x=left_ear_x, y=left_ear_y, mode='lines', line=dict(color='black', width=2), hoverinfo='skip', showlegend=False))
        fig.add_trace(go.Scatter(x=right_ear_x, y=right_ear_y, mode='lines', line=dict(color='black', width=2), hoverinfo='skip', showlegend=False))

        # 5. Electrode Markers & Labels
        fig.add_trace(go.Scatter(
            x=elec_x, y=elec_y,
            mode='markers+text',
            text=labels,
            textposition="top center",
            marker=dict(size=10, color='white', line=dict(color='black', width=2)),
            showlegend=False
        ))

        fig.update_layout(
            title=f"<b>{title}</b>",
            xaxis=dict(range=[-1.2, 1.2], showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(range=[-1.2, 1.2], showgrid=False, zeroline=False, showticklabels=False),
            width=480,
            height=480,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=20, r=20, t=50, b=20)
        )

        return fig
