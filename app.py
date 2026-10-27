"""Local web dashboard for the Financial Fraud Analytics pipeline."""

import csv
import json
import logging
import os
import shutil
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, abort, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from run_pipeline import execute_pipeline


PROJECT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
METRICS_DIR = PROJECT_DIR / "output" / "metrics"
WEB_RUNS_DIR = PROJECT_DIR / "output" / "web_runs"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_SYNTHETIC_RECORDS = 100_000
REQUIRED_COLUMNS = {
    "step", "type", "amount", "nameOrig", "oldbalanceOrg",
    "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
    "isFraud", "isFlaggedFraud"
}
ARTIFACTS = {
    "report": ("full_evaluation_report.json", "application/json"),
    "transactions": ("transaction_level_metrics.csv", "text/csv"),
    "rings": ("ring_level_metrics.csv", "text/csv"),
    "scaling": ("spark_systems_scaling.csv", "text/csv"),
    "predictions": ("transaction_predictions.csv", "text/csv"),
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
logger = logging.getLogger("FraudAnalyticsWeb")
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fraud-pipeline")
jobs: Dict[str, Dict[str, Any]] = {}
jobs_lock = threading.Lock()
active_job_id: Optional[str] = None


def available_datasets():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        file.name for file in RAW_DATA_DIR.iterdir()
        if file.is_file() and file.suffix.lower() == ".csv"
    )


def validate_csv_headers(path: Path) -> Optional[str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            headers = next(csv.reader(csv_file), [])
    except (OSError, UnicodeError, csv.Error) as exc:
        return f"Could not read the CSV file: {exc}"

    missing = sorted(REQUIRED_COLUMNS.difference(header.strip() for header in headers))
    if missing:
        return "CSV is missing required PaySim columns: " + ", ".join(missing)
    return None


def run_job(
    job_id: str,
    num_records: int,
    model_type: str,
    force_synthetic: bool,
    run_scaling_benchmark: bool,
    input_csv_path: Optional[str]
) -> None:
    global active_job_id
    try:
        execute_pipeline(
            num_records=num_records,
            model_type=model_type,
            force_synthetic=force_synthetic,
            run_scaling_benchmark=run_scaling_benchmark,
            input_csv_path=input_csv_path
        )

        artifact_dir = WEB_RUNS_DIR / job_id
        artifact_dir.mkdir(parents=True, exist_ok=False)
        for filename, _ in ARTIFACTS.values():
            source = METRICS_DIR / filename
            if source.is_file():
                shutil.copy2(source, artifact_dir / filename)

        report_path = artifact_dir / ARTIFACTS["report"][0]
        if not report_path.is_file():
            raise RuntimeError("The pipeline completed without creating its evaluation report.")

        with report_path.open("r", encoding="utf-8") as report_file:
            report = json.load(report_file)
        with jobs_lock:
            jobs[job_id].update(status="completed", report=report)
    except Exception as exc:
        logger.exception("Pipeline run %s failed", job_id)
        with jobs_lock:
            jobs[job_id].update(status="failed", error=str(exc))
    finally:
        if input_csv_path and Path(input_csv_path).parent == Path(tempfile.gettempdir()):
            try:
                os.remove(input_csv_path)
            except FileNotFoundError:
                pass
            except OSError:
                logger.exception("Could not remove temporary uploaded data for run %s", job_id)
        with jobs_lock:
            active_job_id = None


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/datasets")
def list_datasets():
    return jsonify({"datasets": available_datasets()})


@app.post("/api/runs")
def start_run():
    global active_job_id

    source = request.form.get("source", "synthetic")
    model_type = request.form.get("model_type", "xgboost")
    if source not in {"synthetic", "existing", "upload"}:
        return jsonify(error="Choose a valid dataset source."), 400
    if model_type not in {"xgboost", "random_forest"}:
        return jsonify(error="Choose a valid model type."), 400

    try:
        num_records = int(request.form.get("num_records", "10000"))
    except ValueError:
        return jsonify(error="Record count must be a whole number."), 400
    if not 100 <= num_records <= MAX_SYNTHETIC_RECORDS:
        return jsonify(error=f"Synthetic record count must be between 100 and {MAX_SYNTHETIC_RECORDS:,}."), 400

    input_csv_path = None
    force_synthetic = source == "synthetic"
    if source == "existing":
        selected_dataset = request.form.get("dataset", "")
        if selected_dataset not in available_datasets():
            return jsonify(error="Select an available CSV dataset."), 400
        input_csv_path = str(RAW_DATA_DIR / selected_dataset)
    elif source == "upload":
        uploaded_file = request.files.get("dataset_file")
        if not uploaded_file or not secure_filename(uploaded_file.filename or ""):
            return jsonify(error="Choose a CSV file to upload."), 400
        if Path(uploaded_file.filename).suffix.lower() != ".csv":
            return jsonify(error="Only CSV uploads are supported."), 400
        temporary_file = tempfile.NamedTemporaryFile(
            prefix="fraud_upload_", suffix=".csv", delete=False
        )
        input_csv_path = temporary_file.name
        temporary_file.close()
        uploaded_file.save(input_csv_path)
        validation_error = validate_csv_headers(Path(input_csv_path))
        if validation_error:
            os.remove(input_csv_path)
            return jsonify(error=validation_error), 400

    job_id = uuid.uuid4().hex
    with jobs_lock:
        if active_job_id is not None:
            if source == "upload" and input_csv_path:
                os.remove(input_csv_path)
            return jsonify(error="A pipeline run is already in progress. Try again when it finishes."), 409
        active_job_id = job_id
        jobs[job_id] = {"status": "running", "error": None, "report": None}

    try:
        executor.submit(
            run_job,
            job_id,
            num_records,
            model_type,
            force_synthetic,
            request.form.get("run_scaling_benchmark") == "true",
            input_csv_path
        )
    except Exception as exc:
        with jobs_lock:
            active_job_id = None
            jobs[job_id].update(status="failed", error=str(exc))
        if source == "upload" and input_csv_path:
            os.remove(input_csv_path)
        logger.exception("Could not queue pipeline run %s", job_id)
        return jsonify(error="Could not start the pipeline. Check the server logs for details."), 500
    return jsonify({"job_id": job_id, "status": "running"}), 202


@app.get("/api/runs/<job_id>")
def get_run(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            abort(404)
        response = dict(job)

    if response["status"] == "completed":
        artifact_dir = WEB_RUNS_DIR / job_id
        response["artifacts"] = {
            key: f"/api/runs/{job_id}/artifacts/{key}"
            for key, (filename, _) in ARTIFACTS.items()
            if (artifact_dir / filename).is_file()
        }
    return jsonify(response)


@app.get("/api/runs/<job_id>/artifacts/<artifact_name>")
def get_artifact(job_id: str, artifact_name: str):
    if artifact_name not in ARTIFACTS:
        abort(404)
    with jobs_lock:
        if job_id not in jobs or jobs[job_id]["status"] != "completed":
            abort(404)
    filename, mimetype = ARTIFACTS[artifact_name]
    artifact_path = WEB_RUNS_DIR / job_id / filename
    if not artifact_path.is_file():
        abort(404)
    return send_file(artifact_path, mimetype=mimetype, as_attachment=True, download_name=filename)


@app.errorhandler(413)
def upload_too_large(_error):
    return jsonify(error=f"CSV uploads must be smaller than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
