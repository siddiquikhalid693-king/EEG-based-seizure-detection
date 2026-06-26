# Experiment Plan

## Baseline

Train the included Random Forest baseline on numeric EEG features exported to CSV.
This gives a reproducible comparison point before introducing a CNN+LSTM model.

## Deep Learning Direction

The intended production research model is a hybrid CNN+LSTM:

- CNN layers learn local waveform/spatial patterns.
- LSTM layers learn temporal dependencies across windows.
- Final dense layer emits seizure probability.

## Dataset Rules

CHB-MIT and UCI-style datasets should not be mixed blindly. CHB-MIT contains scalp EEG
recordings with patient and time context. The UCI epileptic seizure dataset is already
segmented/transformed. Keep dataset-specific preprocessing documented and report
results separately unless a harmonized protocol is defined.

## Minimum Experiment Record

Each experiment should record:

- dataset name and source
- train/validation/test split policy
- preprocessing settings
- model architecture or baseline name
- threshold
- confusion matrix
- accuracy, precision, recall, F1-score
- notes on class imbalance handling
