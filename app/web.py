from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from flask import Flask, flash, redirect, render_template, request, send_file, url_for

from .eeg_epilepsy.model import predict_csv
from .eeg_epilepsy.reporting import build_prediction_report


ALLOWED_EXTENSIONS = {".csv"}


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_mapping(
        SECRET_KEY="dev-change-me",
        MODEL_PATH=Path("models/seizure_model.joblib"),
        UPLOAD_FOLDER=Path("uploads"),
        REPORT_FOLDER=Path("reports"),
    )

    @app.get("/")
    def index():
        model_ready = Path(app.config["MODEL_PATH"]).exists()
        return render_template("index.html", model_ready=model_ready)

    @app.post("/predict")
    def predict():
        model_path = Path(app.config["MODEL_PATH"])
        if not model_path.exists():
            flash("Train a model first: python scripts/train.py data/train.csv")
            return redirect(url_for("index"))

        upload = request.files.get("eeg_file")
        if not upload or upload.filename == "":
            flash("Choose a CSV file to analyze.")
            return redirect(url_for("index"))

        extension = Path(upload.filename).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            flash("Only CSV uploads are supported in this prototype.")
            return redirect(url_for("index"))

        upload_dir = Path(app.config["UPLOAD_FOLDER"])
        upload_dir.mkdir(parents=True, exist_ok=True)
        saved_path = upload_dir / f"{uuid4().hex}{extension}"
        upload.save(saved_path)

        try:
            result = predict_csv(saved_path, model_path)
            report_path = Path(app.config["REPORT_FOLDER"]) / f"{saved_path.stem}.pdf"
            build_prediction_report(report_path, upload.filename, result)
        except Exception as exc:
            flash(str(exc))
            return redirect(url_for("index"))

        return render_template(
            "result.html",
            result=result,
            source_filename=upload.filename,
            report_name=report_path.name,
        )

    @app.get("/reports/<report_name>")
    def report(report_name: str):
        report_path = Path(app.config["REPORT_FOLDER"]) / report_name
        return send_file(report_path, as_attachment=True)

    return app
