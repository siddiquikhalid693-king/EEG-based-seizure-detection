# CEREVIGIL – An Intelligent EEG-Based Epileptic Seizure Detection System

**CEREVIGIL** is a deep learning-powered clinical workbench for multi-channel EEG signal visualization, automated epileptic seizure detection, 2D brain topography mapping, time-frequency spectral analysis, and automated clinical PDF report generation.

---

## 🌟 Key Features

- **Deep Learning Architectures**:
  - **EEGNet**: Compact Spatial-Temporal Convolutional Neural Network.
  - **CNN-LSTM**: Hybrid Deep Learning model combining spatial CNN feature maps with bidirectional LSTM temporal dynamics.
- **10-20 EEG Montage Visualizer**: Interactive multi-channel EEG signal monitor with customizable signal gain and automated seizure alert highlighting.
- **2D Scalp Brain Topography & Spectrograms**: Real-time cortical voltage potential heatmap (10-20 electrode system) & STFT time-frequency spectrogram analysis.
- **DSP Feature Extraction**: Hjorth parameters (Activity, Mobility, Complexity) and relative spectral band powers ($\delta, \theta, \alpha, \beta, \gamma$).
- **Clinical PDF Diagnostics**: Automated PDF export with patient details, confidence scores, focal channel metrics, and frequency distribution charts.

---

## 🚀 Launch CEREVIGIL

```bash
streamlit run app.py
```
