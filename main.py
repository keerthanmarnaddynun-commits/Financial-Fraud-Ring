"""End-to-end PaySim coordinated fraud-ring analytics runner."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from data_ingestion import (
    PAYMENTS_SCHEMA,
    build_data_quality_report,
    create_spark_session,
    ensure_dataset,
    load_and_validate_data,
)
from evaluation import collect_spark_metrics, evaluate_models, write_json
from feature_engineering import (
    add_transaction_graph_features,
    extract_account_features,
)
from graph_construction import build_transaction_graph
from models import sample_training_data, train_comparative_models

LOGGER = logging.getLogger("fraud_analytics")


def run_pipeline(
    data_path: Optional[Path],
    output_dir: Path,
    max_training_rows: int,
    sample_fraction: Optional[float] = None,
) -> Dict[str, Any]:
    pipeline_started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = data_path or ensure_dataset()
    spark = create_spark_session()
    timings: Dict[str, float] = {}

    try:
        started = time.perf_counter()
        quality_report = build_data_quality_report(spark, input_path)
        write_json(quality_report, str(output_dir / "data_quality.json"))
        transactions = load_and_validate_data(spark, input_path)
        if sample_fraction is not None:
            transactions = transactions.sample(withReplacement=False, fraction=sample_fraction, seed=42)
        transactions = transactions.cache()
        transaction_count = transactions.count()
        timings["ingestion_and_validation"] = time.perf_counter() - started
        LOGGER.info("Validated %s PaySim transactions", f"{transaction_count:,}")

        started = time.perf_counter()
        graph = build_transaction_graph(transactions)
        graph.vertices.cache()
        graph.edges.cache()
        graph.vertices.count()
        graph.edges.count()
        timings["graph_construction"] = time.perf_counter() - started

        started = time.perf_counter()
        account_features = extract_account_features(graph).cache()
        account_count = account_features.count()
        enriched = add_transaction_graph_features(
            transactions, account_features
        )
        timings["graph_feature_engineering"] = time.perf_counter() - started
        LOGGER.info("Computed graph features for %s accounts", f"{account_count:,}")

        started = time.perf_counter()
        training_data = sample_training_data(enriched, max_training_rows)
        models, test_data, _ = train_comparative_models(training_data, output_dir=output_dir)
        timings["model_training_and_prediction"] = time.perf_counter() - started

        report = evaluate_models(test_data)
        write_json(report, str(output_dir / "evaluation.json"))
        metrics = collect_spark_metrics(spark, timings)
        timings["pipeline_total"] = time.perf_counter() - pipeline_started
        metrics["transaction_count"] = transaction_count
        metrics["account_count"] = account_count
        metrics["sampled_training_rows"] = int(len(training_data))
        metrics["test_rows"] = int(len(test_data))
        metrics["model_names"] = list(models)
        write_json(metrics, str(output_dir / "scalability.json"))
        LOGGER.info("Evaluation report: %s", json.dumps(report, indent=2))
        LOGGER.info("Scalability metrics: %s", json.dumps(metrics, indent=2))
        return {"evaluation": report, "scalability": metrics, "data_quality": quality_report}
    finally:
        spark.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PaySim graph-based coordinated fraud-ring analytics."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="PaySim CSV path (default: discover or download a CSV in data/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for JSON evaluation and scalability reports.",
    )
    parser.add_argument(
        "--max-training-rows",
        type=int,
        default=100_000,
        help="Maximum class-aware sample collected for scikit-learn training.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=None,
        help="Optional sample fraction for fast debugging runs (0.0-1.0).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    if args.sample_fraction is not None and not 0 < args.sample_fraction <= 1:
        raise ValueError("--sample-fraction must be between 0 and 1.")
    run_pipeline(args.data, args.output_dir, args.max_training_rows, args.sample_fraction)


if __name__ == "__main__":
    main()
