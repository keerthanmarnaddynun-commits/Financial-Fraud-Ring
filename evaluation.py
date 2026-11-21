"""Distributed scoring and realistic evaluation across time, group, and random splits."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd
import psutil
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from feature_sets import (
    ALL_MODEL_NAMES,
    TRAINABLE_MODELS,
    get_model_features,
)
from splits import (
    find_best_f1_threshold,
    get_dataset_split,
)
from stage_timer import StageTimer

LOGGER = logging.getLogger(__name__)


def compute_binary_metrics_at_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Compute precision, recall, f1, and exact confusion counts at a given threshold."""
    pred = (y_scores >= threshold).astype(int)
    tp = int(((y_true == 1) & (pred == 1)).sum())
    fp = int(((y_true == 0) & (pred == 1)).sum())
    fn = int(((y_true == 1) & (pred == 0)).sum())
    tn = int(((y_true == 0) & (pred == 0)).sum())

    denom_p = tp + fp
    denom_r = tp + fn
    precision = float(tp / denom_p) if denom_p > 0 else 0.0
    recall = float(tp / denom_r) if denom_r > 0 else 0.0
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
    }


def evaluate_model_predictions(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    val_threshold: float,
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics at validation threshold and 0.5."""
    rows = int(len(y_true))
    fraud_count = int((y_true == 1).sum())
    prevalence = float(fraud_count / rows) if rows > 0 else 0.0

    at_threshold = compute_binary_metrics_at_threshold(y_true, y_scores, val_threshold)
    at_05 = compute_binary_metrics_at_threshold(y_true, y_scores, 0.5)

    try:
        auc = (
            float(roc_auc_score(y_true, y_scores))
            if len(np.unique(y_true)) > 1
            else 0.5
        )
    except Exception:
        auc = 0.5

    try:
        ap = (
            float(average_precision_score(y_true, y_scores))
            if len(np.unique(y_true)) > 1
            else prevalence
        )
    except Exception:
        ap = prevalence

    fp = at_threshold["confusion"]["fp"]
    false_alarms_per_10k = float(fp / rows * 10000.0) if rows > 0 else 0.0

    return {
        "threshold": float(val_threshold),
        "threshold_source": "validation",
        "at_threshold": at_threshold,
        "at_0.5": at_05,
        "roc_auc": auc,
        "average_precision": ap,
        "prevalence": prevalence,
        "false_alarms_per_10k": false_alarms_per_10k,
        "rows": rows,
    }


def compute_pr_curve_points(
    split_name: str,
    model_name: str,
    y_true: np.ndarray,
    y_scores: np.ndarray,
    max_points: int = 2000,
) -> pd.DataFrame:
    """Generate up to max_points precision-recall curve coordinates."""
    if len(np.unique(y_true)) < 2:
        return pd.DataFrame(columns=["split", "model", "threshold", "precision", "recall"])

    prec, rec, thresh = precision_recall_curve(y_true, y_scores)
    # Align lengths: thresh has length len(prec) - 1
    # Add 1.0 threshold for final point
    thresh_full = np.append(thresh, 1.0)

    n_points = len(thresh_full)
    if n_points > max_points:
        step = max(1, n_points // max_points)
        idx = np.arange(0, n_points, step)
        if idx[-1] != n_points - 1:
            idx = np.append(idx, n_points - 1)
        prec = prec[idx]
        rec = rec[idx]
        thresh_full = thresh_full[idx]

    return pd.DataFrame({
        "split": split_name,
        "model": model_name,
        "threshold": thresh_full,
        "precision": prec,
        "recall": rec,
    })


def score_spark_dataset_map_in_pandas(
    spark: SparkSession,
    df: DataFrame,
    models: Dict[str, Any],
) -> DataFrame:
    """Score a Spark DataFrame using mapInPandas with broadcasted models."""
    sc = spark.sparkContext
    # Trainable models to broadcast
    trainable_models = {
        name: models[name] for name in TRAINABLE_MODELS if name in models
    }
    broadcast_models = sc.broadcast(trainable_models)
    feature_map = {name: get_model_features(name) for name in trainable_models}

    # Define output schema
    schema_fields = [
        StructField("transaction_id", StringType(), True),
        StructField("step", IntegerType(), True),
        StructField("type", StringType(), True),
        StructField("amount", DoubleType(), True),
        StructField("isFraud", IntegerType(), True),
        StructField("isFlaggedFraud", IntegerType(), True),
        StructField("score_m0_rule", DoubleType(), True),
    ]
    for name in TRAINABLE_MODELS:
        schema_fields.append(StructField(f"score_{name}", DoubleType(), True))
    output_schema = StructType(schema_fields)
    output_cols = [f.name for f in schema_fields]

    def score_partition(iterator):
        loaded_models = broadcast_models.value
        for pdf in iterator:
            if pdf.empty:
                yield pd.DataFrame(columns=output_cols)
                continue

            out = pd.DataFrame()
            out["transaction_id"] = pdf["transaction_id"].astype(str)
            out["step"] = pdf["step"].astype(int)
            out["type"] = pdf["type"].astype(str)
            out["amount"] = pdf["amount"].astype(float)
            out["isFraud"] = pdf["isFraud"].astype(int)
            out["isFlaggedFraud"] = (
                pdf["isFlaggedFraud"].astype(int)
                if "isFlaggedFraud" in pdf.columns
                else pd.Series(0, index=pdf.index)
            )
            out["score_m0_rule"] = out["isFlaggedFraud"].astype(float)

            for name in TRAINABLE_MODELS:
                col_name = f"score_{name}"
                if name in loaded_models:
                    model = loaded_models[name]
                    feats = feature_map.get(name, [])
                    for f in feats:
                        if f not in pdf.columns:
                            pdf[f] = 0.0
                    try:
                        probs = model.predict_proba(pdf[feats])[:, 1]
                    except Exception:
                        probs = np.zeros(len(pdf))
                    out[col_name] = probs.astype(float)
                else:
                    out[col_name] = 0.0

            yield out[output_cols]

    return df.mapInPandas(score_partition, output_schema)


def run_distributed_evaluation(
    spark: SparkSession,
    enriched_df: DataFrame,
    models: Dict[str, Any],
    output_dir: Path,
    splits_to_run: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Execute distributed scoring and produce evaluation.json and PR curves across splits."""
    if splits_to_run is None:
        splits_to_run = ["time", "group_dest", "random"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits_report: Dict[str, Dict[str, Any]] = {}
    pr_curve_dfs: List[pd.DataFrame] = []
    scoring_throughput: Dict[str, float] = {}

    for split_name in splits_to_run:
        LOGGER.info("Starting distributed evaluation for split: %s", split_name)
        train_df, val_df, test_df = get_dataset_split(enriched_df, split_name)

        # Determine models to use for this split (per-split models or fallback to shared models)
        if isinstance(models, dict) and split_name in models and isinstance(models[split_name], dict):
            split_models = models[split_name]
        else:
            split_models = models

        # 1. Validation phase: find best F1 thresholds
        val_thresholds: Dict[str, float] = {}
        val_count = val_df.count()

        if val_count > 0:
            val_scored_df = score_spark_dataset_map_in_pandas(spark, val_df, split_models)
            val_pdf = (
                val_scored_df.select(
                    "isFraud",
                    "score_m0_rule",
                    *[f"score_{m}" for m in TRAINABLE_MODELS],
                )
                .toPandas()
            )
            val_labels = val_pdf["isFraud"].values

            val_thresholds["m0_rule"] = 0.5
            for m in TRAINABLE_MODELS:
                score_col = f"score_{m}"
                scores = val_pdf[score_col].values
                thresh, _ = find_best_f1_threshold(val_labels, scores)
                val_thresholds[m] = thresh
        else:
            val_thresholds = {m: 0.5 for m in ALL_MODEL_NAMES}

        # 2. Test phase: score full test rows in Spark with mapInPandas
        with StageTimer(f"scoring_{split_name}") as timer:
            scored_test_df = score_spark_dataset_map_in_pandas(spark, test_df, split_models)
            scored_test_df = scored_test_df.cache()

            # Save scored test set to parquet
            parquet_path = output_dir / f"scored_{split_name}.parquet"
            scored_test_df.write.mode("overwrite").parquet(str(parquet_path))

            # Collect test labels and scores for exact evaluation
            test_pdf = (
                scored_test_df.select(
                    "isFraud",
                    "score_m0_rule",
                    *[f"score_{m}" for m in TRAINABLE_MODELS],
                )
                .toPandas()
            )

        test_rows = len(test_pdf)
        wall_time = max(0.001, getattr(timer, "wall_seconds", 0.001))
        throughput = float(test_rows / wall_time)
        scoring_throughput[split_name] = throughput
        LOGGER.info(
            "Scored %d test rows for %s in %.2fs (%.1f rows/s)",
            test_rows,
            split_name,
            wall_time,
            throughput,
        )

        test_labels = test_pdf["isFraud"].values
        split_model_results: Dict[str, Any] = {}

        for m_name in ALL_MODEL_NAMES:
            score_col = f"score_{m_name}"
            scores = test_pdf[score_col].values
            thresh = val_thresholds.get(m_name, 0.5)

            metrics = evaluate_model_predictions(test_labels, scores, thresh)
            split_model_results[m_name] = metrics

            pr_df = compute_pr_curve_points(split_name, m_name, test_labels, scores)
            if not pr_df.empty:
                pr_curve_dfs.append(pr_df)

        splits_report[split_name] = split_model_results

    # Save combined PR curves CSV
    if pr_curve_dfs:
        combined_pr = pd.concat(pr_curve_dfs, ignore_index=True)
        combined_pr.to_csv(output_dir / "pr_curves.csv", index=False)

    evaluation_report = {
        "splits": splits_report,
        "models": list(ALL_MODEL_NAMES),
        "notes": [
            "Evaluation conducted on full held-out test splits without synthetic oversampling.",
            "Threshold picked on validation set at true prevalence; 0.5 default also reported.",
            "Confusion counts satisfy exact consistency: tp + fn == fraud_rows, tp + fp + fn + tn == rows.",
        ],
        "scoring_throughput_rows_per_second": scoring_throughput,
    }

    write_json(evaluation_report, str(output_dir / "evaluation.json"))
    return evaluation_report


# =========================================================================
# Legacy compatibility evaluation
# =========================================================================

def _binary_metrics(
    actual: pd.Series, predicted: pd.Series
) -> Dict[str, Optional[float]]:
    try:
        auc: Optional[float] = float(roc_auc_score(actual, predicted))
    except ValueError:
        auc = None
    from sklearn.metrics import f1_score, precision_score, recall_score
    return {
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "roc_auc": auc,
    }


def evaluate_models(test_data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Legacy evaluation report supporting previous test_data pandas structure."""
    reports: Dict[str, Dict[str, Any]] = {}
    actual = test_data["actual"] if "actual" in test_data.columns else test_data["isFraud"].astype(int)

    for name in ("m1_tabular", "m2_graph", "m3_community_mule", "tabular_baseline", "graph_features", "community_ring"):
        prob_col = f"{name}_probability"
        pred_col = f"{name}_prediction"
        if prob_col not in test_data.columns:
            continue
        prediction = test_data[pred_col]
        probability = test_data[prob_col]
        transaction_metrics = _binary_metrics(actual, prediction)
        transaction_metrics["roc_auc"] = (
            float(roc_auc_score(actual, probability))
            if actual.nunique() > 1
            else None
        )
        transaction_metrics["rows"] = int(len(test_data))
        reports[name] = {
            "transaction": transaction_metrics,
        }
    return reports


def collect_spark_metrics(
    spark: SparkSession, timings: Dict[str, Any]
) -> Dict[str, Any]:
    """Collect observed executor memory and shuffle bytes from Spark's local REST UI."""
    # 1. Block 'phases': ingestion_and_validation, graph_construction, graph_feature_engineering,
    #    model_training, distributed_scoring_and_eval, pipeline_total
    phases: Dict[str, float] = {
        "ingestion_and_validation": float(timings.get("ingestion_and_validation", 0.0)),
        "graph_construction": float(timings.get("graph_construction", 0.0)),
        "graph_feature_engineering": float(timings.get("graph_feature_engineering", 0.0)),
        "model_training": float(timings.get("model_training", 0.0)),
        "distributed_scoring_and_eval": float(timings.get("distributed_scoring_and_eval", 0.0)),
        "pipeline_total": float(timings.get("pipeline_total", 0.0)),
    }

    # 2. Block 'graph_stages': degrees, pagerank, triangles, components, lpa, behaviour_features
    graph_stages: Dict[str, float] = {}
    raw_graph = timings.get("graph_stages") or timings.get("timings_seconds") or timings.get("stages") or {}
    for stg in ["degrees", "pagerank", "triangles", "components", "lpa", "behaviour_features"]:
        if isinstance(raw_graph, dict) and stg in raw_graph and isinstance(raw_graph[stg], (int, float)):
            graph_stages[stg] = float(raw_graph[stg])
        elif "stages" in timings and isinstance(timings["stages"], dict) and stg in timings["stages"]:
            graph_stages[stg] = float(timings["stages"][stg].get("wall_seconds", 0.0))
        elif stg in timings and isinstance(timings[stg], (int, float)):
            graph_stages[stg] = float(timings[stg])

    # 3. Block 'stages': per-stage wall_seconds, peak_rss_bytes, baseline_rss_bytes, min_available_ram_bytes,
    #    stages_count_delta, shuffle_read_bytes_delta, shuffle_write_bytes_delta
    stages_block: Dict[str, Any] = {}
    if "stages" in timings and isinstance(timings["stages"], dict):
        stages_block = timings["stages"]

    # Flat timings_seconds for backward compatibility
    flat_timings: Dict[str, float] = {**phases, **graph_stages}

    avail_bytes = psutil.virtual_memory().available
    mem_warning = None
    if avail_bytes < 2 * 1024 * 1024 * 1024:
        mem_warning = (
            f"Low memory warning: available RAM ({avail_bytes / (1024**3):.2f} GB) dropped below 2.0 GB threshold."
        )
        LOGGER.warning(mem_warning)

    metrics: Dict[str, Any] = {
        "phases": phases,
        "graph_stages": graph_stages,
        "stages": stages_block,
        "timings_seconds": flat_timings,
        "spark_application_id": spark.sparkContext.applicationId,
        "spark_master": spark.sparkContext.master,
        "executor_memory_bytes": None,
        "shuffle_read_bytes": None,
        "shuffle_write_bytes": None,
        "stage_count": None,
        "memory_warning": mem_warning,
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
