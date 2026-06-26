from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.eeg_epilepsy.model import train_from_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the EEG seizure baseline model.")
    parser.add_argument("csv_path", help="Training CSV with numeric features and labels.")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--model-path", default="models/seizure_model.joblib")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    result = train_from_csv(
        args.csv_path,
        label_column=args.label_column,
        model_path=args.model_path,
        threshold=args.threshold,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
