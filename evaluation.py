"""Transaction-, ring-, and Spark-runtime evaluation helpers."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd
from pyspark.sql import SparkSession
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

LOGGER = logging.getLogger(__name__)


def _binary_metrics(
    actual: pd.Series, predicted: pd.Series
) -> Dict[str, Optional[float]]:
    try:
        auc: Optional[float] = float(roc_auc_score(actual, predicted))
    except ValueError:
        auc = None
    return {
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "roc_auc": auc,
    }


def evaluate_models(test_data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Report transaction metrics and community-aggregated ring detection metrics."""
    reports: Dict[str, Dict[str, Any]] = {}
    community = test_data["origin_community_id"].fillna(
        test_data["transaction_id"].astype(str)
    )
    for name in ("tabular_baseline", "graph_features", "community_ring"):
        prediction = test_data[f"{name}_prediction"]
        probability = test_data[f"{name}_probability"]
        transaction_metrics = _binary_metrics(test_data["actual"], prediction)
        transaction_metrics["roc_auc"] = (
            float(roc_auc_score(test_data["actual"], probability))
            if test_data["actual"].nunique() > 1
            else None
        )
        transaction_metrics["rows"] = int(len(test_data))
        reports[name] = {
            "transaction": transaction_metrics,
            "ring": _ring_metrics(test_data["actual"], prediction, community),
        }
    return reports


def _ring_metrics(
    actual: pd.Series, predicted: pd.Series, communities: pd.Series
) -> Dict[str, Any]:
    rings = pd.DataFrame(
        {"community": communities, "actual": actual, "predicted": predicted}
    ).groupby("community", dropna=False)[["actual", "predicted"]].max()
    return {
        **_binary_metrics(rings["actual"], rings["predicted"]),
        "rings_evaluated": int(len(rings)),
        "fraud_rings": int(rings["actual"].sum()),
        "detected_rings": int(rings["predicted"].sum()),
        "definition": (
            "A discovered origin-account community is a fraud ring when it contains "
            "at least one labeled fraudulent transaction; it is detected when at "
            "least one transaction in that community is flagged."
        ),
    }


def collect_spark_metrics(
    spark: SparkSession, timings: Dict[str, float]
) -> Dict[str, Any]:
    """Collect observed executor memory and shuffle bytes from Spark's local REST UI."""
    metrics: Dict[str, Any] = {
        "timings_seconds": timings,
        "spark_application_id": spark.sparkContext.applicationId,
        "spark_master": spark.sparkContext.master,
        "executor_memory_bytes": None,
        "shuffle_read_bytes": None,
        "shuffle_write_bytes": None,
        "stage_count": None,
    }
    ui_url = spark.sparkContext.uiWebUrl
    if not ui_url:
        LOGGER.warning("Spark UI is disabled; executor and shuffle metrics are unavailable")
        return metrics
    base = (
        f"{ui_url.rstrip('/')}/api/v1/applications/"
        f"{spark.sparkContext.applicationId}"
    )
    try:
        with urlopen(f"{base}/executors", timeout=5) as response:
            executors = json.load(response)
        with urlopen(f"{base}/stages", timeout=5) as response:
            stages = json.load(response)
        metrics["executor_memory_bytes"] = sum(
            int(executor.get("memoryUsed", 0))
            for executor in executors
        )
        metrics["shuffle_read_bytes"] = sum(
            int(stage.get("shuffleReadBytes", 0)) for stage in stages
        )
        metrics["shuffle_write_bytes"] = sum(
            int(stage.get("shuffleWriteBytes", 0)) for stage in stages
        )
        metrics["stage_count"] = len(stages)
    except (OSError, URLError, ValueError, TypeError) as error:
        LOGGER.warning("Could not collect Spark UI metrics from %s: %s", ui_url, error)
    return metrics


def write_json(data: Dict[str, Any], path: str) -> None:
    """Write an indented JSON report, handling optional metrics explicitly."""
    with open(path, "w", encoding="utf-8") as output:
        json.dump(data, output, indent=2, allow_nan=False)
        output.write("\n")
