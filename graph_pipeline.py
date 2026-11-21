"""Reusable Spark graph-stage pipeline runner shared by main.py and scaling benchmarks."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import psutil
from pyspark.sql import DataFrame, SparkSession

from data_ingestion import load_and_validate_data
from feature_engineering import (
    add_transaction_graph_features,
    extract_account_features,
)
from graph_construction import build_transaction_graph
from stage_timer import StageTimer, get_process_tree_rss

LOGGER = logging.getLogger("fraud_analytics.graph_pipeline")


def run_graph_stages(
    spark: SparkSession,
    df: Optional[DataFrame] = None,
    timer: Optional[Dict[str, Any]] = None,
    *,
    input_path: Optional[Path] = None,
    input_format: str = "csv",
    sample_fraction: Optional[float] = None,
    seed: int = 42,
    materialize_enriched: bool = True,
) -> Dict[str, Any]:
    """Execute graph stages and return timings, graph sizes, shuffle bytes, and peak RSS.

    Stages measured:
      ingest_validate, graph_build, degrees, pagerank, triangles, components,
      lpa, behaviour_features, feature_join.
    """
    overall_start = time.perf_counter()
    timer_results: Dict[str, Any] = timer if timer is not None else {}

    # 1. Ingest and validate
    with StageTimer("ingest_validate", timer_results, spark):
        if df is None:
            transactions = load_and_validate_data(
                spark, csv_path=input_path, input_format=input_format
            )
        else:
            transactions = df
        if sample_fraction is not None and sample_fraction < 1.0:
            transactions = transactions.sample(
                withReplacement=False, fraction=float(sample_fraction), seed=seed
            )
        transactions = transactions.cache()
        rows = int(transactions.count())
    LOGGER.info("ingest_validate finished: %s transactions", f"{rows:,}")

    # 2. Graph construction
    with StageTimer("graph_build", timer_results, spark):
        graph = build_transaction_graph(transactions)
        graph.vertices.cache()
        graph.edges.cache()
        nodes = int(graph.vertices.count())
        edges = int(graph.edges.count())
    LOGGER.info("graph_build finished: %s nodes, %s edges", f"{nodes:,}", f"{edges:,}")

    # 3. Account graph features (degrees, pagerank, triangles, components, lpa, behaviour_features)
    gfe_start = time.perf_counter()
    account_features, engines, algo_info = extract_account_features(
        graph, transactions=transactions, timer_results=timer_results, spark=spark
    )
    account_features = account_features.cache()
    account_count = int(account_features.count())

    # 4. Feature join back onto transactions
    with StageTimer("feature_join", timer_results, spark):
        enriched = add_transaction_graph_features(transactions, account_features)
        if materialize_enriched:
            enriched.count()
    gfe_seconds = time.perf_counter() - gfe_start
    total_seconds = time.perf_counter() - overall_start

    stages_detail: Dict[str, Dict[str, Any]] = timer_results.get("stages", {})
    stage_names = [
        "ingest_validate",
        "graph_build",
        "degrees",
        "pagerank",
        "triangles",
        "components",
        "lpa",
        "behaviour_features",
        "feature_join",
    ]

    seconds: Dict[str, float] = {}
    peak_rss = get_process_tree_rss()
    min_avail_ram = psutil.virtual_memory().available
    shuffle_read_total = 0
    shuffle_write_total = 0

    for stg in stage_names:
        info = stages_detail.get(stg)
        if isinstance(info, dict):
            seconds[stg] = round(float(info.get("wall_seconds", 0.0)), 4)
            peak_rss = max(peak_rss, int(info.get("peak_rss_bytes", 0)))
            stg_avail = int(info.get("min_available_ram_bytes", 0))
            if stg_avail > 0:
                min_avail_ram = min(min_avail_ram, stg_avail)
            shuffle_read_total += int(info.get("shuffle_read_bytes_delta", 0))
            shuffle_write_total += int(info.get("shuffle_write_bytes_delta", 0))
        else:
            seconds[stg] = 0.0

    seconds["graph_feature_engineering"] = round(float(gfe_seconds), 4)
    seconds["total"] = round(float(total_seconds), 4)

    return {
        "enriched": enriched,
        "transactions": transactions,
        "account_features": account_features,
        "rows": rows,
        "nodes": nodes,
        "edges": edges,
        "account_count": account_count,
        "engines": engines,
        "algo_info": algo_info,
        "seconds": seconds,
        "stages": stages_detail,
        "peak_rss_bytes": int(peak_rss),
        "min_available_ram_bytes": int(min_avail_ram),
        "shuffle_read_bytes": int(shuffle_read_total),
        "shuffle_write_bytes": int(shuffle_write_total),
        "shuffle_bytes": int(shuffle_read_total + shuffle_write_total),
        "timings_dict": timer_results,
    }
