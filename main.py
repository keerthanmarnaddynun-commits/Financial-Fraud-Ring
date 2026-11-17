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
    ensure_dataset,
    load_and_validate_data,
)
from run_info import write_run_info
from spark_session import create_spark_session
from evaluation import (
    collect_spark_metrics,
    evaluate_models,
    run_distributed_evaluation,
    write_json,
)
from feature_engineering import (
    add_transaction_graph_features,
    extract_account_features,
)
from graph_construction import build_transaction_graph
from models import train_comparative_models
from splits import draw_training_sample, get_dataset_split

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
    write_run_info(
        output_dir / "run_info.json",
        dataset_path=input_path,
        config={
            "sample_fraction": sample_fraction,
            "max_training_rows": max_training_rows,
        },
    )
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
        LOGGER.info("Ingested %s PaySim transactions", f"{transaction_count:,}")

        started = time.perf_counter()
        graph = build_transaction_graph(transactions)
        graph.vertices.cache()
        graph.edges.cache()
        graph.vertices.count()
        graph.edges.count()
        timings["graph_construction"] = time.perf_counter() - started

        started = time.perf_counter()
        account_features, engines, algo_info = extract_account_features(
            graph, transactions=transactions, timer_results=timings, spark=spark
        )
        account_features = account_features.cache()
        account_count = account_features.count()
        enriched = add_transaction_graph_features(
            transactions, account_features
        )
        timings["graph_feature_engineering"] = time.perf_counter() - started
        LOGGER.info("Computed graph features for %s accounts", f"{account_count:,}")

        # Persist enriched features to Parquet in the run folder
        enriched_parquet_path = output_dir / "enriched_features.parquet"
        LOGGER.info("Persisting enriched features to %s", enriched_parquet_path)
        enriched.write.mode("overwrite").parquet(str(enriched_parquet_path))
        enriched = spark.read.parquet(str(enriched_parquet_path)).cache()

        # Per-split training: call training once per split on each split's own train rows
        splits_to_run = ["time", "group_dest", "random"]
        models_by_split = {}
        training_samples_by_split = {}
        total_training_rows = 0

        started = time.perf_counter()
        for split_name in splits_to_run:
            LOGGER.info("Drawing training sample and training models for split: %s", split_name)
            split_train_df, _, _ = get_dataset_split(enriched, split_name)
            split_training_data = draw_training_sample(split_train_df, max_training_rows)
            training_samples_by_split[split_name] = int(len(split_training_data))
            total_training_rows += len(split_training_data)
            split_models, _, _ = train_comparative_models(
                split_training_data, output_dir=output_dir, split_name=split_name
            )
            models_by_split[split_name] = split_models

        timings["model_training"] = time.perf_counter() - started

        started = time.perf_counter()
        report = run_distributed_evaluation(
            spark=spark,
            enriched_df=enriched,
            models=models_by_split,
            output_dir=output_dir,
            splits_to_run=splits_to_run,
        )
        timings["distributed_scoring_and_eval"] = time.perf_counter() - started
        timings["pipeline_total"] = time.perf_counter() - pipeline_started

        metrics = collect_spark_metrics(spark, timings)
        metrics["phases"]["pipeline_total"] = time.perf_counter() - pipeline_started
        metrics["timings_seconds"]["pipeline_total"] = metrics["phases"]["pipeline_total"]
        metrics["transaction_count"] = transaction_count
        metrics["account_count"] = account_count

        # Requirement (c): record sampled_training_rows per split
        metrics["sampled_training_rows"] = training_samples_by_split
        metrics["sampled_training_rows_per_split"] = training_samples_by_split
        metrics["sampled_training_rows_total"] = int(total_training_rows)

        # Canonical model names only (no legacy aliases)
        canonical_models = [m for m in models_by_split.get("time", {}).keys() if not m.startswith("score_")]
        metrics["model_names"] = canonical_models
        metrics["engines"] = engines
        metrics["algorithm_info"] = algo_info
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
