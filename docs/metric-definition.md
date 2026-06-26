# Metric Definition

## Classification Target

This project is a binary classifier.

- Positive class: `ictal` / `seizure`
- Negative class: `non-ictal` / `normal`

All reported seizure metrics must treat the seizure class as positive.

## Confusion Matrix Terms

- True Positive (TP): seizure epoch predicted as seizure
- False Positive (FP): normal epoch predicted as seizure
- True Negative (TN): normal epoch predicted as normal
- False Negative (FN): seizure epoch predicted as normal

## Metrics

Accuracy:

```text
(TP + TN) / (TP + TN + FP + FN)
```

Precision:

```text
TP / (TP + FP)
```

Recall / Sensitivity:

```text
TP / (TP + FN)
```

F1-score:

```text
2 * precision * recall / (precision + recall)
```

## Reporting Policy

Recall and F1-score are primary metrics because missed seizures have high clinical
cost and EEG seizure datasets are often imbalanced. Accuracy is reported as a
secondary metric and must not be used alone.

Use a held-out test set that was not augmented or oversampled. If SMOTE is used,
apply it only to the training split. For CHB-MIT experiments, prefer patient-wise
splits to avoid leakage between windows from the same patient.

Every report should include:

- positive-class definition
- decision threshold
- confusion matrix
- accuracy, precision, recall, and F1-score
- whether the split is patient-wise or window-wise
