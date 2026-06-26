from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


POSITIVE_LABELS = {"1", "true", "yes", "seizure", "ictal", "positive"}
NEGATIVE_LABELS = {"0", "false", "no", "normal", "non-ictal", "nonictal", "negative"}


@dataclass(frozen=True)
class ConfusionCounts:
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    @property
    def total(self) -> int:
        return self.true_positive + self.true_negative + self.false_positive + self.false_negative


@dataclass(frozen=True)
class MetricReport:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion: ConfusionCounts


def normalize_label(value: object) -> int:
    """Return 1 for seizure/ictal and 0 for normal/non-ictal labels."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)

    normalized = str(value).strip().lower()
    if normalized in POSITIVE_LABELS:
        return 1
    if normalized in NEGATIVE_LABELS:
        return 0
    raise ValueError(f"Unsupported label value: {value!r}")


def confusion_counts(y_true: Iterable[object], y_pred: Iterable[object]) -> ConfusionCounts:
    true_values = [normalize_label(value) for value in y_true]
    pred_values = [normalize_label(value) for value in y_pred]
    if len(true_values) != len(pred_values):
        raise ValueError("y_true and y_pred must have the same length")

    tp = tn = fp = fn = 0
    for actual, predicted in zip(true_values, pred_values):
        if actual == 1 and predicted == 1:
            tp += 1
        elif actual == 0 and predicted == 0:
            tn += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        elif actual == 1 and predicted == 0:
            fn += 1

    return ConfusionCounts(tp, tn, fp, fn)


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def metric_report(y_true: Iterable[object], y_pred: Iterable[object]) -> MetricReport:
    counts = confusion_counts(y_true, y_pred)
    accuracy = safe_divide(counts.true_positive + counts.true_negative, counts.total)
    precision = safe_divide(counts.true_positive, counts.true_positive + counts.false_positive)
    recall = safe_divide(counts.true_positive, counts.true_positive + counts.false_negative)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return MetricReport(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        confusion=counts,
    )


def threshold_probabilities(probabilities: Iterable[float], threshold: float = 0.5) -> list[int]:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    return [1 if probability >= threshold else 0 for probability in probabilities]
