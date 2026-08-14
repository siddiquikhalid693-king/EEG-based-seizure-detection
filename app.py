"""
EEG-Based Epilepsy Detection System & Clinical Workbench
Interactive Streamlit Dashboard powered by PyTorch Deep Learning (EEGNet & CNN-LSTM), DSP signal processing, 2D Brain Topography, and PDF diagnostic report generation.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Append current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.generator import EEGDataGenerator, CHANNELS, CLASS_NAMES
from dataset.edf_parser import EEGFileParser
from dsp.filtering import EEGFilter
from dsp.features import EEGFeatureExtractor, FREQ_BANDS
from dsp.spectrogram import EEGSpectrogram
from models.classifier import SeizureClassifierWrapper
from visualization.eeg_plotter import EEGPlotter
from visualization.scalp_topography import ScalpTopography
from reports.pdf_generator import ClinicalPDFReportGenerator

# Streamlit Page Config
st.set_page_config(
    page_title="CEREVIGIL – An Intelligent EEG-Based Epileptic Seizure Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling for Premium Clinical Aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
    .status-normal {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.2rem;
    }
    .status-seizure {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.2rem;
        animation: pulse 1.5s infinite;
    }
    .status-interictal {
        background-color: #FEF3C7;
        color: #B45309;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_classifier(model_type: str):
    """Cached loader for PyTorch EEG Classifier."""
    weights_path = f"{model_type.lower()}_model.pth"
    if not os.path.exists(weights_path):
        weights_path = 'model.pth' if os.path.exists('model.pth') else None
    return SeizureClassifierWrapper(model_type=model_type, weights_path=weights_path)

def main():
    st.markdown('<div class="main-header">🧠 CEREVIGIL</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">An Intelligent EEG-Based Epileptic Seizure Detection System</div>', unsafe_allow_html=True)

    # Sidebar Controls
    st.sidebar.title("⚙️ Clinical Configuration")
    
    # 1. Model Selector
    model_choice = st.sidebar.selectbox(
        "Select Deep Learning Model Architecture",
        ["EEGNet (Compact Spatial-Temporal CNN)", "CNN-LSTM (BiLSTM Sequence Hybrid)"]
    )
    model_key = 'eegnet' if 'EEGNet' in model_choice else 'cnn_lstm'
    classifier = load_classifier(model_key)

    # 2. Data Source Selector
    data_source = st.sidebar.radio("Data Source", ["Clinical Case Presets", "Upload Custom CSV / EDF"])
    
    fs = 256
    eeg_data = None
    channels = CHANNELS
    case_label = "NORMAL"

    if data_source == "Clinical Case Presets":
        preset = st.sidebar.selectbox(
            "Select Clinical Preset",
            [
                "Case 1: Focal Temporal Lobe Seizure (Ictal)",
                "Case 2: Generalized 3Hz Spike-Wave (Absence Seizure)",
                "Case 3: Interictal Temporal Sharp Waves",
                "Case 4: Pre-Ictal Prodromal State",
                "Case 5: Normal Awake Alpha Background"
            ]
        )
        gen = EEGDataGenerator(fs=fs)
        if "Case 1" in preset:
            eeg_data, _ = gen.generate_focal_temporal_seizure(duration_sec=10.0)
            case_label = "Focal Temporal Seizure"
        elif "Case 2" in preset:
            eeg_data, _ = gen.generate_absence_seizure(duration_sec=10.0)
            case_label = "Absence Seizure"
        elif "Case 3" in preset:
            eeg_data, _ = gen.generate_interictal_spikes(duration_sec=10.0)
            case_label = "Interictal Spikes"
        elif "Case 4" in preset:
            eeg_data, _ = gen.generate_preictal_eeg(duration_sec=10.0)
            case_label = "Pre-Ictal State"
        else:
            eeg_data, _ = gen.generate_normal_eeg(duration_sec=10.0)
            case_label = "Normal Awake"
    else:
        uploaded_file = st.sidebar.file_uploader("Upload EEG File (.csv or .edf)", type=["csv", "edf"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                eeg_data, channels, fs = EEGFileParser.parse_csv(uploaded_file)
            elif uploaded_file.name.endswith('.edf'):
                # Save temporarily for EDF reader
                with open("temp.edf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                eeg_data, channels, fs = EEGFileParser.parse_edf("temp.edf")
        else:
            st.info("Please upload a CSV or EDF file to begin analysis, or switch to Clinical Case Presets.")
            return

    # Signal Processing Filters Toggle
    st.sidebar.subheader("🎛️ Digital Filters")
    apply_notch = st.sidebar.checkbox("50Hz Notch Filter", value=True)
    apply_bandpass = st.sidebar.checkbox("0.5 - 70Hz Bandpass Filter", value=True)

    if apply_notch or apply_bandpass:
        lowcut = 0.5 if apply_bandpass else 0.1
        highcut = 70.0 if apply_bandpass else (fs / 2 - 1)
        notch_freq = 50.0 if apply_notch else 0.0
        eeg_data = EEGFilter.process_eeg(eeg_data, fs=fs, lowcut=lowcut, highcut=highcut, notch_freq=notch_freq)

    # Display Controls
    st.sidebar.subheader("🔍 Visualization Display")
    gain = st.sidebar.slider("Signal Sensitivity Gain", min_value=0.2, max_value=4.0, value=1.0, step=0.1)
    duration_sec = eeg_data.shape[1] / fs

    # -------------------------------------------------------------
    # Deep Learning Inference Execution
    # -------------------------------------------------------------
    # Extract center 4-second epoch for classifier
    epoch_samples = min(1024, eeg_data.shape[1])
    epoch_data = eeg_data[:, :epoch_samples]
    prediction = classifier.predict_epoch(epoch_data)

    # Band Power Calculation
    rel_band_powers = EEGFeatureExtractor.calculate_band_powers(epoch_data, fs=fs)
    avg_band_powers = {band: float(np.mean(vals)) for band, vals in rel_band_powers.items()}

    # Header Alert Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("**Diagnostic Status**")
        p_label = prediction['predicted_label']
        if prediction['is_seizure']:
            st.markdown(f'<span class="status-seizure">🚨 {p_label}</span>', unsafe_allow_html=True)
        elif p_label == 'INTERICTAL_SPIKE':
            st.markdown(f'<span class="status-interictal">⚠️ {p_label}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="status-normal">🟢 {p_label}</span>', unsafe_allow_html=True)

    with m2:
        st.metric("AI Model Confidence", f"{prediction['confidence']:.1f}%")

    with m3:
        st.metric("Focal Channels", ", ".join(prediction['focal_channels'][:2]))

    with m4:
        st.metric("Sampling Rate / Duration", f"{fs} Hz / {duration_sec:.1f}s")

    st.markdown("---")

    # -------------------------------------------------------------
    # Main Application Tabs
    # -------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Multi-Channel Waveforms",
        "🗺️ 2D Brain Topography & Spectrogram",
        "📊 DSP Features & Spectral Bands",
        "📄 Clinical PDF Report"
    ])

    # -------------------------------------------------------------
    # Tab 1: Multi-Channel EEG Signal Visualizer
    # -------------------------------------------------------------
    with tab1:
        st.subheader("Multi-Channel 10-20 EEG Waveform Monitor")
        
        # Prepare Seizure Highlights overlay if seizure detected
        highlights = []
        if prediction['is_seizure']:
            highlights.append({
                'start_sec': 0.0,
                'end_sec': duration_sec,
                'color': 'rgba(239, 68, 68, 0.25)',
                'label': f"AI Seizure Alert ({prediction['confidence']:.1f}%)"
            })

        fig_eeg = EEGPlotter.create_eeg_figure(
            eeg_data=eeg_data,
            channel_names=channels,
            fs=fs,
            gain=gain,
            highlights=highlights
        )
        st.plotly_chart(fig_eeg, use_container_width=True)

    # -------------------------------------------------------------
    # Tab 2: 2D Brain Scalp Topography & Spectrogram
    # -------------------------------------------------------------
    with tab2:
        col_top, col_spec = st.columns([1, 1])
        
        with col_top:
            st.subheader("2D Scalp Brain Potential Heatmap")
            # Calculate instantaneous voltage amplitude across channels
            instant_voltages = {channels[i]: float(np.max(np.abs(epoch_data[i]))) for i in range(len(channels))}
            fig_scalp = ScalpTopography.create_scalp_heatmap(instant_voltages, title="Cortical Amplitude Topography (uV)")
            st.plotly_chart(fig_scalp, use_container_width=True)

        with col_spec:
            st.subheader("Time-Frequency Spectrogram (STFT)")
            selected_ch = st.selectbox("Select Channel for STFT Analysis", channels, index=1 if len(channels) > 1 else 0)
            ch_idx = channels.index(selected_ch)
            
            freqs, times, Sxx_log = EEGSpectrogram.compute_spectrogram(eeg_data[ch_idx], fs=fs)
            
            fig_spec = go.Figure(data=go.Heatmap(
                z=Sxx_log,
                x=times,
                y=freqs,
                colorscale='Viridis',
                colorbar=dict(title="dB/Hz")
            ))
            fig_spec.update_layout(
                title=f"<b>Spectrogram: {selected_ch}</b>",
                xaxis_title="Time (s)",
                yaxis_title="Frequency (Hz)",
                height=450,
                margin=dict(l=40, r=40, t=50, b=40)
            )
            st.plotly_chart(fig_spec, use_container_width=True)

    # -------------------------------------------------------------
    # Tab 3: Feature Workbench & Spectral Power
    # -------------------------------------------------------------
    with tab3:
        st.subheader("EEG Feature Extraction & Spectral Analysis")
        col_b1, col_b2 = st.columns([1, 1])

        with col_b1:
            st.markdown("#### Frequency Band Power Distribution")
            df_bands = pd.DataFrame({
                'Frequency Band': list(avg_band_powers.keys()),
                'Relative Power (%)': list(avg_band_powers.values())
            })
            fig_band = px.bar(
                df_bands, x='Frequency Band', y='Relative Power (%)',
                color='Frequency Band', text_auto='.1f',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_band.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_band, use_container_width=True)

        with col_b2:
            st.markdown("#### Hjorth Parameters per Channel")
            hjorth = EEGFeatureExtractor.calculate_hjorth_parameters(epoch_data)
            df_hjorth = pd.DataFrame({
                'Channel': channels[:len(hjorth['Activity'])],
                'Activity (uV²)': np.round(hjorth['Activity'], 2),
                'Mobility': np.round(hjorth['Mobility'], 3),
                'Complexity': np.round(hjorth['Complexity'], 3)
            })
            st.dataframe(df_hjorth, use_container_width=True, height=320)

    # -------------------------------------------------------------
    # Tab 4: Clinical PDF Diagnostic Report
    # -------------------------------------------------------------
    with tab4:
        st.subheader("Generate Clinical Diagnostic PDF Report")
        
        c1, c2 = st.columns(2)
        with c1:
            patient_name = st.text_input("Patient Name", value="John Doe")
            patient_id = st.text_input("Patient ID", value="PT-1042")
        with c2:
            patient_age = st.number_input("Patient Age", value=34, min_value=1, max_value=120)
            patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])

        if st.button("📄 Export Clinical Diagnostic Report (PDF)", type="primary"):
            patient_info = {
                'name': patient_name,
                'id': patient_id,
                'age': patient_age,
                'gender': patient_gender
            }
            recording_info = {
                'fs': fs,
                'duration_sec': duration_sec,
                'n_channels': len(channels),
                'montage': '10-20 Standard Bipolar'
            }
            
            pdf_path = "eeg_clinical_report.pdf"
            ClinicalPDFReportGenerator.generate_report(
                output_filename=pdf_path,
                patient_info=patient_info,
                recording_info=recording_info,
                detection_results=prediction,
                band_powers=avg_band_powers
            )
            
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=f,
                    file_name=f"EEG_Report_{patient_id}.pdf",
                    mime="application/pdf"
                )
            st.success(f"Diagnostic report generated successfully: {pdf_path}")

if __name__ == '__main__':
    main()
