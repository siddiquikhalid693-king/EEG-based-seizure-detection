from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import metric_report, normalize_label, threshold_probabilities
from .preprocessing import load_numeric_csv


@dataclass(frozen=True)
class PredictionResult:
    predicted_label: int
    confidence: float
    threshold: float

    @property
    def label_name(self) -> str:
        return "seizure" if self.predicted_label == 1 else "normal"


def build_baseline_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_from_csv(
    csv_path: str | Path,
    label_column: str = "label",
    model_path: str | Path = "models/seizure_model.joblib",
    threshold: float = 0.5,
) -> dict[str, Any]:
    features, labels = load_numeric_csv(csv_path, label_column=label_column)
    if labels is None:
        raise ValueError(f"Label column {label_column!r} was not found")
    normalized_labels = labels.map(normalize_label)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        normalized_labels,
        test_size=0.2,
        stratify=normalized_labels,
        random_state=42,
    )

    model = build_baseline_model()
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = threshold_probabilities(probabilities, threshold)
    report = metric_report(y_test, predictions)

    auc = None
    if len(set(y_test)) == 2:
        auc = float(roc_auc_score(y_test, probabilities))

    payload = {
        "model": model,
        "feature_columns": list(features.columns),
        "threshold": threshold,
        "label_column": label_column,
    }
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, model_path)

    return {
        "model_path": str(model_path),
        "metrics": asdict(report),
        "roc_auc": auc,
        "samples": int(len(features)),
        "test_samples": int(len(x_test)),
    }


def load_model(model_path: str | Path) -> dict[str, Any]:
    return joblib.load(model_path)


def predict_frame(frame: pd.DataFrame, model_payload: dict[str, Any]) -> PredictionResult:
    columns = model_payload["feature_columns"]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(missing)}")

    aligned = frame[columns]
    model = model_payload["model"]
    threshold = float(model_payload.get("threshold", 0.5))
    probability = float(model.predict_proba(aligned)[:, 1].mean())
    predicted_label = 1 if probability >= threshold else 0
    confidence = probability if predicted_label == 1 else 1 - probability
    return PredictionResult(predicted_label=predicted_label, confidence=confidence, threshold=threshold)


def predict_csv(csv_path: str | Path, model_path: str | Path) -> PredictionResult:
    payload = load_model(model_path)
    features, _ = load_numeric_csv(csv_path, label_column=payload.get("label_column"))
    return predict_frame(features, payload)
