from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.eeg_epilepsy.metrics import metric_report, threshold_probabilities
from app.eeg_epilepsy.model import load_model
from app.eeg_epilepsy.preprocessing import load_numeric_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained EEG seizure model.")
    parser.add_argument("csv_path", help="Held-out CSV with labels.")
    parser.add_argument("--model-path", default="models/seizure_model.joblib")
    args = parser.parse_args()

    payload = load_model(args.model_path)
    features, labels = load_numeric_csv(args.csv_path, label_column=payload.get("label_column", "label"))
    if labels is None:
        raise SystemExit("Evaluation CSV must include the configured label column.")

    aligned = features[payload["feature_columns"]]
    probabilities = payload["model"].predict_proba(aligned)[:, 1]
    predictions = threshold_probabilities(probabilities, payload.get("threshold", 0.5))
    report = metric_report(labels, predictions)
    print(json.dumps(report, default=lambda value: value.__dict__, indent=2))


if __name__ == "__main__":
    main()
