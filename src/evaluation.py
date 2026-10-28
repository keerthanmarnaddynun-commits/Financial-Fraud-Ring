"""
evaluation.py
Comprehensive evaluation module for Financial Fraud Analytics.
Evaluates models at:
1. Transaction level: Precision, Recall, F1, AUC-ROC, PR-AUC, Confusion Matrix
2. Ring level: Ring detection rate, cluster purity, cluster completeness, V-measure
3. Spark Systems Performance: Execution time, shuffle size, and memory overhead scaling
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix,
    homogeneity_score, completeness_score, v_measure_score
)

from src.utils import logger, SparkPerformanceTracker
from src.graph_construction import construct_transaction_graph
from src.feature_engineering import compute_structural_graph_features


def evaluate_transaction_level(
    y_true: pd.Series,
    predictions: Dict[str, np.ndarray],
    probabilities: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Computes Precision, Recall, F1-Score, AUC-ROC, and PR-AUC for each comparative model.
    """
    logger.info("Computing transaction-level evaluation metrics...")
    metrics_list = []

    for name in predictions.keys():
        y_pred = predictions[name]
        y_prob = probabilities[name]

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
        pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        metrics_list.append({
            "Model": name,
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "AUC-ROC": round(auc, 4),
            "PR-AUC": round(pr_auc, 4),
            "True Positive": int(tp),
            "False Positive": int(fp),
            "False Negative": int(fn),
            "True Negative": int(tn)
        })

    df_metrics = pd.DataFrame(metrics_list)
    return df_metrics


def evaluate_ring_level(
    test_df: pd.DataFrame,
    predictions: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Evaluates fraud detection performance at the community / fraud ring level:
    - Ring Detection Rate: Fraction of true fraud rings detected by the model
    - Cluster Purity: Proportion of transactions in flagged rings that are actually fraudulent
    - Cluster Completeness: Fraction of all fraud ring accounts recovered within detected communities
    """
    logger.info("Computing ring-level (community clustering) evaluation metrics...")
    ring_metrics_list = []

    # Identify true fraud rings by connected components / communities where true fraud occurred
    fraud_mask = (test_df["isFraud"] == 1)
    true_fraud_rings = set(test_df.loc[fraud_mask, "src_community"].unique())
    total_fraud_rings = len(true_fraud_rings)

    for name, y_pred in predictions.items():
        df_eval = test_df.copy()
        df_eval["pred"] = y_pred

        # Identify communities flagged by the model (at least one predicted fraud txn)
        flagged_mask = (df_eval["pred"] == 1)
        flagged_rings = set(df_eval.loc[flagged_mask, "src_community"].unique())

        # 1. Ring Detection Rate: Fraction of true fraud communities captured
        captured_rings = true_fraud_rings.intersection(flagged_rings)
        ring_detection_rate = len(captured_rings) / max(1, total_fraud_rings)

        # 2. Cluster Purity: For flagged rings, fraction of transactions that are truly fraud
        flagged_txns = df_eval[df_eval["src_community"].isin(flagged_rings)]
        if len(flagged_txns) > 0:
            cluster_purity = flagged_txns["isFraud"].mean()
        else:
            cluster_purity = 0.0

        # 3. Cluster Completeness & V-Measure
        try:
            completeness = completeness_score(df_eval["isFraud"], df_eval["src_community"])
            homogeneity = homogeneity_score(df_eval["isFraud"], df_eval["src_community"])
            v_measure = v_measure_score(df_eval["isFraud"], df_eval["src_community"])
        except Exception:
            completeness, homogeneity, v_measure = 0.0, 0.0, 0.0

        ring_metrics_list.append({
            "Model": name,
            "Total Fraud Rings": total_fraud_rings,
            "Detected Fraud Rings": len(captured_rings),
            "Ring Detection Rate": round(ring_detection_rate, 4),
            "Cluster Purity": round(cluster_purity, 4),
            "Cluster Completeness": round(completeness, 4),
            "V-Measure": round(v_measure, 4)
        })

    return pd.DataFrame(ring_metrics_list)


def benchmark_spark_systems_scaling(
    spark,
    parquet_path: str,
    scale_fractions: List[float] = [0.25, 0.50, 1.0]
) -> pd.DataFrame:
    """
    Measures and logs Spark systems performance (execution time, shuffle size,
    and memory overhead as the graph size scales).
    """
    logger.info("Executing Spark Graph Scaling Benchmark across multiple graph sizes...")
    scaling_records = []

    for frac in scale_fractions:
        logger.info(f"Scaling Benchmark: Running at {frac*100:.0f}% graph scale...")
        tracker = SparkPerformanceTracker(spark).start()

        # Build Graph at scale fraction
        graph, v_df, e_df = construct_transaction_graph(spark, parquet_path, sample_fraction=frac)
        v_count = graph.vertices.count()
        e_count = graph.edges.count()

        # Compute graph algorithms
        _, alg_benchmarks = compute_structural_graph_features(
            spark,
            graph,
            pagerank_max_iter=3,
            lpa_max_iter=2
        )

        sys_metrics = tracker.stop()

        scaling_records.append({
            "Graph Scale": f"{int(frac*100)}%",
            "Nodes": v_count,
            "Edges": e_count,
            "PageRank Time (s)": alg_benchmarks.get("pagerank_duration_sec", 0.0),
            "Connected Comp Time (s)": alg_benchmarks.get("connected_components_duration_sec", 0.0),
            "Triangle Count Time (s)": alg_benchmarks.get("triangle_count_duration_sec", 0.0),
            "LPA Time (s)": alg_benchmarks.get("lpa_duration_sec", 0.0),
            "Total Time (s)": sys_metrics["duration_seconds"],
            "JVM Used Mem (MB)": sys_metrics["jvm_used_memory_mb"],
            "JVM Max Mem (MB)": sys_metrics["jvm_max_memory_mb"]
        })

    df_scaling = pd.DataFrame(scaling_records)
    return df_scaling


def run_full_evaluation(
    spark,
    training_results: Dict[str, Any],
    clean_parquet_path: str,
    output_dir: str = "output/metrics",
    run_scaling_benchmark: bool = True
) -> Dict[str, Any]:
    """
    Orchestrates transaction-level evaluation, ring-level evaluation, and Spark system scaling benchmarks.
    Saves results to JSON and Markdown.
    """
    os.makedirs(output_dir, exist_ok=True)
    y_true = training_results["y_test"]
    predictions = training_results["predictions"]
    probabilities = training_results["probabilities"]
    test_df = training_results["test_df"]

    # 1. Transaction-level metrics
    df_txn_metrics = evaluate_transaction_level(y_true, predictions, probabilities)
    txn_csv = os.path.join(output_dir, "transaction_level_metrics.csv")
    df_txn_metrics.to_csv(txn_csv, index=False)

    # 2. Ring-level metrics
    df_ring_metrics = evaluate_ring_level(test_df, predictions)
    ring_csv = os.path.join(output_dir, "ring_level_metrics.csv")
    df_ring_metrics.to_csv(ring_csv, index=False)

    # 3. Spark systems scaling benchmark
    df_scaling = None
    if run_scaling_benchmark:
        df_scaling = benchmark_spark_systems_scaling(spark, clean_parquet_path, scale_fractions=[0.3, 0.6, 1.0])
        scaling_csv = os.path.join(output_dir, "spark_systems_scaling.csv")
        df_scaling.to_csv(scaling_csv, index=False)

    # Compile comprehensive report
    report = {
        "transaction_level": df_txn_metrics.to_dict(orient="records"),
        "ring_level": df_ring_metrics.to_dict(orient="records"),
        "spark_scaling": df_scaling.to_dict(orient="records") if df_scaling is not None else []
    }

    report_json = os.path.join(output_dir, "full_evaluation_report.json")
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Evaluation report successfully saved to {report_json}")
    return {
        "df_txn_metrics": df_txn_metrics,
        "df_ring_metrics": df_ring_metrics,
        "df_scaling": df_scaling,
        "report_json": report_json
    }
