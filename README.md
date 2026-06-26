# EEG-Based Epileptic Seizure Detection

Software prototype for binary EEG seizure detection: `ictal/seizure` vs
`non-ictal/normal`.

The project includes:

- EEG preprocessing utilities for filtering, epoching, normalization, and feature extraction
- A trainable baseline classifier for tabular/CSV experiments
- Metric definitions focused on seizure-class recall, precision, F1, and confusion matrix
- Flask web app for upload, prediction, confidence display, and PDF report download
- CLI scripts for training, evaluation, and one-off predictions

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

Then open `http://127.0.0.1:5000`.

## Data Format

For the current baseline, use CSV files with numeric EEG-derived columns and one label
column. The default label column is `label`, where seizure samples are represented by
`1`, `seizure`, `ictal`, or `true`, and normal samples by `0`, `normal`, `non-ictal`,
or `false`.

Example:

```csv
channel_1,channel_2,channel_3,label
0.12,0.04,-0.18,0
1.42,0.91,1.11,1
```

Raw `.edf` support is scaffolded through `mne`, but training from EDF needs a
patient-wise dataset manifest and annotation policy before it should be trusted.

## Training

```bash
python scripts/train.py data/train.csv --label-column label --model-path models/seizure_model.joblib
```

## Evaluation

```bash
python scripts/evaluate.py data/test.csv --model-path models/seizure_model.joblib
```

Evaluate on a held-out test set that was not oversampled. If SMOTE is used later, it
must be applied to the training split only.

## Prediction

```bash
python scripts/predict.py sample.csv --model-path models/seizure_model.joblib
```

## Project Notes

The planned deep-learning direction is a hybrid CNN+LSTM model. This repository starts
with a reproducible baseline and a stable app contract so data handling, metrics, and
reporting can mature before adding heavier model training.
