from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.eeg_epilepsy.model import predict_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict seizure likelihood for a CSV file.")
    parser.add_argument("csv_path", help="CSV with numeric EEG features.")
    parser.add_argument("--model-path", default="models/seizure_model.joblib")
    args = parser.parse_args()

    result = predict_csv(args.csv_path, args.model_path)
    print(json.dumps(result.__dict__ | {"label_name": result.label_name}, indent=2))


if __name__ == "__main__":
    main()
