"""Serve the local fraud analytics dashboard and read-only project APIs."""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
OUTPUT_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "data"

FUNCTIONS = [
    {
        "name": "create_spark_session",
        "file": "data_ingestion.py",
        "signature": "create_spark_session(app_name: str = 'PaySimFraudAnalytics') -> SparkSession",
        "purpose": "Starts a local[*] Spark session and loads the matching GraphFrames JVM package.",
        "inputs": [{"name": "app_name", "type": "str", "description": "Spark application display name."}],
        "outputs": [{"name": "SparkSession", "type": "pyspark.sql.SparkSession", "description": "Configured local Spark session."}],
    },
    {
        "name": "ensure_dataset",
        "file": "data_ingestion.py",
        "signature": "ensure_dataset(data_dir: Path = DATA_DIR) -> Path",
        "purpose": "Finds a CSV in data/ or downloads the PaySim dataset with the Kaggle CLI.",
        "inputs": [{"name": "data_dir", "type": "Path", "description": "Directory to search or populate."}],
        "outputs": [{"name": "CSV path", "type": "Path", "description": "Path to the PaySim CSV; raises a descriptive error if download fails."}],
    },
    {
        "name": "load_and_validate_data",
        "file": "data_ingestion.py",
        "signature": "load_and_validate_data(spark: SparkSession, csv_path: Optional[Path] = None) -> DataFrame",
        "purpose": "Loads PaySim with its explicit schema, checks headers, filters invalid rows, and adds transaction_id.",
        "inputs": [{"name": "spark", "type": "SparkSession", "description": "Active Spark session."}, {"name": "csv_path", "type": "Optional[Path]", "description": "Input CSV; when omitted, finds or downloads PaySim."}],
        "outputs": [{"name": "transactions", "type": "pyspark.sql.DataFrame", "description": "Validated transactions with a transaction_id column."}],
    },
    {
        "name": "build_transaction_graph",
        "file": "graph_construction.py",
        "signature": "build_transaction_graph(transactions: DataFrame) -> GraphFrame",
        "purpose": "Builds account vertices and directed transaction edges (origin → destination).",
        "inputs": [{"name": "transactions", "type": "DataFrame", "description": "Validated transactions with nameOrig and nameDest."}],
        "outputs": [{"name": "graph", "type": "graphframes.GraphFrame", "description": "Directed account transaction graph."}],
    },
    {
        "name": "extract_account_features",
        "file": "feature_engineering.py",
        "signature": "extract_account_features(graph: GraphFrame) -> DataFrame",
        "purpose": "Computes in/out degree, PageRank, triangle count, GraphX connected components, and Label Propagation communities.",
        "inputs": [{"name": "graph", "type": "GraphFrame", "description": "Account graph with id vertices and src/dst edges."}],
        "outputs": [{"name": "account_features", "type": "DataFrame", "description": "Account metrics, community/component IDs, and group sizes."}],
    },
    {
        "name": "add_transaction_graph_features",
        "file": "feature_engineering.py",
        "signature": "add_transaction_graph_features(transactions: DataFrame, account_features: DataFrame) -> DataFrame",
        "purpose": "Joins source/destination account metrics and community agreement onto each transaction.",
        "inputs": [{"name": "transactions", "type": "DataFrame", "description": "Validated PaySim transactions."}, {"name": "account_features", "type": "DataFrame", "description": "Per-account graph and community metrics."}],
        "outputs": [{"name": "enriched_transactions", "type": "DataFrame", "description": "Transactions enriched with origin_*, dest_*, and same_community features."}],
    },
    {
        "name": "sample_training_data",
        "file": "models.py",
        "signature": "sample_training_data(transactions: DataFrame, max_rows: int, seed: int = 42) -> pandas.DataFrame",
        "purpose": "Draws a class-aware, bounded sample before collecting records on the driver.",
        "inputs": [{"name": "transactions", "type": "DataFrame", "description": "Enriched transactions with isFraud."}, {"name": "max_rows", "type": "int", "description": "Upper bound for collected training rows."}, {"name": "seed", "type": "int", "description": "Sampling seed."}],
        "outputs": [{"name": "training_data", "type": "pandas.DataFrame", "description": "Bounded, class-aware data for scikit-learn."}],
    },
    {
        "name": "train_comparative_models",
        "file": "models.py",
        "signature": "train_comparative_models(data: pandas.DataFrame) -> (models, test_data, scores)",
        "purpose": "Fits baseline, graph-feature, and community/ring random forests on one stratified split.",
        "inputs": [{"name": "data", "type": "pandas.DataFrame", "description": "Sampled, labeled transaction features."}],
        "outputs": [{"name": "models", "type": "dict", "description": "Fitted scikit-learn pipelines."}, {"name": "test_data", "type": "pandas.DataFrame", "description": "Held-out labels with each model's predictions and probabilities."}, {"name": "scores", "type": "dict", "description": "ROC AUC scores for each model."}],
    },
    {
        "name": "evaluate_models",
        "file": "evaluation.py",
        "signature": "evaluate_models(test_data: pandas.DataFrame) -> dict",
        "purpose": "Calculates transaction metrics and aggregates predictions/labels by origin-account community for ring metrics.",
        "inputs": [{"name": "test_data", "type": "pandas.DataFrame", "description": "Held-out records with labels, communities, predictions, and probabilities."}],
        "outputs": [{"name": "evaluation", "type": "dict", "description": "Precision, recall, F1, ROC AUC, and ring counts for all three models."}],
    },
    {
        "name": "collect_spark_metrics",
        "file": "evaluation.py",
        "signature": "collect_spark_metrics(spark: SparkSession, timings: dict) -> dict",
        "purpose": "Reads executor memory, shuffle bytes, and stage count from Spark's local REST UI.",
        "inputs": [{"name": "spark", "type": "SparkSession", "description": "Active Spark application with UI enabled."}, {"name": "timings", "type": "dict", "description": "Measured pipeline-stage durations in seconds."}],
        "outputs": [{"name": "scalability", "type": "dict", "description": "Application ID, execution times, executor memory, shuffle, and stage metrics."}],
    },
    {
        "name": "run_pipeline",
        "file": "main.py",
        "signature": "run_pipeline(data_path: Optional[Path], output_dir: Path, max_training_rows: int) -> dict",
        "purpose": "Runs ingestion, graph construction, graph features, model training, evaluation, and report writing sequentially.",
        "inputs": [{"name": "data_path", "type": "Optional[Path]", "description": "PaySim CSV path; None discovers or downloads the dataset."}, {"name": "output_dir", "type": "Path", "description": "Folder for evaluation.json and scalability.json."}, {"name": "max_training_rows", "type": "int", "description": "Bound on the scikit-learn sample."}],
        "outputs": [{"name": "result", "type": "dict", "description": "Evaluation and scalability report data; also writes both JSON reports."}],
    },
]


class DashboardHandler(SimpleHTTPRequestHandler):
    """Serve the dashboard files and small JSON endpoints."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/reports":
            self._send_reports()
        elif path == "/api/functions":
            self._send_json({"functions": FUNCTIONS})
        elif path == "/api/health":
            self._send_health()
        elif path == "/":
            self.path = "/index.html"
            super().do_GET()
        else:
            super().do_GET()

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, indent=2, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_reports(self) -> None:
        evaluation_path = OUTPUT_DIR / "evaluation.json"
        scalability_path = OUTPUT_DIR / "scalability.json"
        try:
            with evaluation_path.open(encoding="utf-8") as file:
                evaluation = json.load(file)
            with scalability_path.open(encoding="utf-8") as file:
                scalability = json.load(file)
        except FileNotFoundError as error:
            self._send_json(
                {"error": f"Report not found: {error.filename}. Run python main.py first."},
                status=404,
            )
            return
        except (OSError, json.JSONDecodeError) as error:
            self._send_json({"error": f"Could not read pipeline reports: {error}"}, status=500)
            return
        report_timestamp = max(
            evaluation_path.stat().st_mtime,
            scalability_path.stat().st_mtime,
        )
        self._send_json(
            {
                "evaluation": evaluation,
                "scalability": scalability,
                "generated_at": report_timestamp,
            }
        )

    def _send_health(self) -> None:
        datasets = sorted(DATA_DIR.glob("*.csv")) if DATA_DIR.exists() else []
        latest_dataset = datasets[0] if datasets else None
        self._send_json(
            {
                "status": "online",
                "dataset": {
                    "available": latest_dataset is not None,
                    "filename": latest_dataset.name if latest_dataset else None,
                    "size_bytes": latest_dataset.stat().st_size if latest_dataset else None,
                },
                "reports_available": all(
                    (OUTPUT_DIR / name).is_file()
                    for name in ("evaluation.json", "scalability.json")
                ),
            }
        )

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[web] {self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the PaySim fraud analytics dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: localhost only).")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000).")
    parser.add_argument("--open", action="store_true", help="Open the dashboard in your browser.")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"PaySim Fraud Analytics dashboard: {url}")
    print("Press Ctrl+C to stop the server.")
    if args.open:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
