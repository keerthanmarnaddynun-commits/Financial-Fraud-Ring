"""Compute account-level graph metrics and transaction-level feature joins using Spark-native DataFrames."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from stage_timer import StageTimer

LOGGER = logging.getLogger("fraud_analytics.features")


def compute_degrees(
    vertices: DataFrame, edges: DataFrame
) -> Tuple[DataFrame, Dict[str, Any]]:
    """Compute in-degree and out-degree per account using Spark DataFrame aggregations."""
    in_deg = (
        edges.groupBy("dst")
        .agg(F.count("*").cast("double").alias("in_degree"))
        .withColumnRenamed("dst", "id")
    )
    out_deg = (
        edges.groupBy("src")
        .agg(F.count("*").cast("double").alias("out_degree"))
        .withColumnRenamed("src", "id")
    )
    result = (
        vertices.select("id")
        .join(in_deg, "id", "left")
        .join(out_deg, "id", "left")
        .fillna({"in_degree": 0.0, "out_degree": 0.0})
    )
    info = {"engine": "spark-native", "iterations": 1, "converged": True}
    return result, info


def compute_pagerank(
    vertices: DataFrame,
    edges: DataFrame,
    num_iter: int = 10,
    damping: float = 0.85,
) -> Tuple[DataFrame, Dict[str, Any]]:
    """Compute PageRank using Spark-native iterative DataFrame joins and mass distribution."""
    # Ensure distinct directed edges to prevent duplicate mass transfers
    distinct_edges = edges.select("src", "dst").distinct().cache()

    # Out-degree counts per source
    out_counts = (
        distinct_edges.groupBy("src")
        .agg(F.count("*").cast("double").alias("out_deg"))
    )
    edges_with_out = (
        distinct_edges.join(out_counts, on="src", how="inner").select("src", "dst", "out_deg").cache()
    )

    # Initial rank 1.0 for every vertex
    ranks = vertices.select("id").withColumn("pagerank", F.lit(1.0)).cache()
    alpha = 1.0 - damping

    for i in range(num_iter):
        contribs = (
            edges_with_out.join(
                ranks.withColumnRenamed("id", "src"), on="src", how="inner"
            ).select(F.col("dst"), (F.col("pagerank") / F.col("out_deg")).alias("contrib"))
        )
        sum_contribs = contribs.groupBy("dst").agg(F.sum("contrib").alias("sum_contrib"))

        new_ranks = (
            vertices.select("id")
            .join(sum_contribs, vertices.id == sum_contribs.dst, "left")
            .select(
                vertices.id,
                (F.lit(alpha) + F.lit(damping) * F.coalesce(F.col("sum_contrib"), F.lit(0.0))).alias(
                    "pagerank"
                ),
            )
        )
        if (i + 1) % 4 == 0:
            new_ranks = new_ranks.localCheckpoint()
        ranks = new_ranks

    info = {"engine": "spark-native", "iterations": num_iter, "converged": True}
    return ranks, info


def compute_triangle_count(
    vertices: DataFrame, edges: DataFrame
) -> Tuple[DataFrame, Dict[str, Any]]:
    """Compute undirected triangle count per account using 3-way self-joins on undirected edges."""
    # Build undirected simple edge set: u < v
    u_edges = (
        edges.select(
            F.least("src", "dst").alias("u"),
            F.greatest("src", "dst").alias("v"),
        )
        .filter(F.col("u") != F.col("v"))
        .distinct()
        .cache()
    )

    e1 = u_edges.select(F.col("u").alias("u"), F.col("v").alias("v"))
    e2 = u_edges.select(F.col("u").alias("v"), F.col("v").alias("w"))
    e3 = u_edges.select(F.col("u").alias("u"), F.col("v").alias("w"))

    triangles = (
        e1.join(e2, on="v")
        .filter(F.col("u") < F.col("w"))
        .join(e3, on=["u", "w"])
    )

    t_u = triangles.groupBy(F.col("u").alias("id")).agg(F.count("*").alias("cnt"))
    t_v = triangles.groupBy(F.col("v").alias("id")).agg(F.count("*").alias("cnt"))
    t_w = triangles.groupBy(F.col("w").alias("id")).agg(F.count("*").alias("cnt"))

    t_all = (
        t_u.unionByName(t_v)
        .unionByName(t_w)
        .groupBy("id")
        .agg(F.sum("cnt").cast("double").alias("triangle_count"))
    )

    tc_df = (
        vertices.select("id")
        .join(t_all, on="id", how="left")
        .fillna({"triangle_count": 0.0})
    )
    info = {"engine": "spark-native", "iterations": 1, "converged": True}
    return tc_df, info


def compute_connected_components(
    vertices: DataFrame,
    edges: DataFrame,
    max_iter: int = 30,
    checkpoint_interval: int = 3,
) -> Tuple[DataFrame, Dict[str, Any]]:
    """Compute weakly connected components using distributed iterative min-label propagation."""
    # Symmetric edge list
    sym_edges = (
        edges.select(F.col("src").alias("u"), F.col("dst").alias("v"))
        .union(edges.select(F.col("dst").alias("u"), F.col("src").alias("v")))
        .filter(F.col("u") != F.col("v"))
        .distinct()
        .cache()
    )

    labels = vertices.select(F.col("id"), F.col("id").alias("component_id")).cache()
    converged = False
    iterations_run = 0

    for i in range(1, max_iter + 1):
        iterations_run = i
        # Join edges with current neighbor component labels
        nbr_labels = sym_edges.join(
            labels.withColumnRenamed("id", "v").withColumnRenamed("component_id", "nbr_label"),
            on="v",
        )
        min_nbr = (
            nbr_labels.groupBy("u")
            .agg(F.min("nbr_label").alias("min_label"))
            .withColumnRenamed("u", "id")
        )

        new_labels = (
            labels.join(min_nbr, on="id", how="left")
            .select(
                F.col("id"),
                F.least(
                    F.col("component_id"),
                    F.coalesce(F.col("min_label"), F.col("component_id")),
                ).alias("new_component_id"),
            )
        )

        # Check for convergence
        changed = (
            new_labels.join(labels, on="id")
            .filter(F.col("new_component_id") != F.col("component_id"))
            .count()
        )

        labels = new_labels.withColumnRenamed("new_component_id", "component_id")
        if i % checkpoint_interval == 0:
            labels = labels.localCheckpoint()

        if changed == 0:
            converged = True
            break

    # Calculate component sizes
    comp_sizes = (
        labels.groupBy("component_id")
        .agg(F.count("*").cast("double").alias("component_size"))
    )
    final_df = labels.join(comp_sizes, on="component_id", how="left")

    info = {
        "engine": "spark-native",
        "iterations": iterations_run,
        "converged": converged,
    }
    return final_df, info


def compute_label_propagation(
    vertices: DataFrame,
    edges: DataFrame,
    max_iter: int = 5,
) -> Tuple[DataFrame, Dict[str, Any]]:
    """Compute community detection via iterative neighbor label mode propagation."""
    sym_edges = (
        edges.select(F.col("src").alias("u"), F.col("dst").alias("v"))
        .union(edges.select(F.col("dst").alias("u"), F.col("src").alias("v")))
        .filter(F.col("u") != F.col("v"))
        .distinct()
        .cache()
    )

    labels = vertices.select(F.col("id"), F.col("id").alias("community_id")).cache()
    converged = False
    iterations_run = 0

    for i in range(1, max_iter + 1):
        iterations_run = i
        joined = sym_edges.join(
            labels.withColumnRenamed("id", "v").withColumnRenamed("community_id", "nbr_comm"),
            on="v",
        )
        freq = (
            joined.groupBy("u", "nbr_comm")
            .agg(F.count("*").alias("freq"))
        )

        window_spec = Window.partitionBy("u").orderBy(
            F.col("freq").desc(), F.col("nbr_comm").asc()
        )
        mode_comm = (
            freq.withColumn("rank", F.row_number().over(window_spec))
            .filter(F.col("rank") == 1)
            .select(F.col("u").alias("id"), F.col("nbr_comm").alias("new_community_id"))
        )

        new_labels = (
            labels.join(mode_comm, on="id", how="left")
            .select(
                F.col("id"),
                F.coalesce(F.col("new_community_id"), F.col("community_id")).alias("new_community_id"),
            )
        )

        changed = (
            new_labels.join(labels, on="id")
            .filter(F.col("new_community_id") != F.col("community_id"))
            .count()
        )
        labels = new_labels.withColumnRenamed("new_community_id", "community_id").localCheckpoint()

        if changed == 0:
            converged = True
            break

    comm_sizes = (
        labels.groupBy("community_id")
        .agg(F.count("*").cast("double").alias("community_size"))
    )
    final_df = labels.join(comm_sizes, on="community_id", how="left")

    info = {
        "engine": "spark-native",
        "iterations": iterations_run,
        "converged": converged,
    }
    return final_df, info


def compute_behaviour_features(
    transactions: DataFrame, vertices: DataFrame
) -> Tuple[DataFrame, Dict[str, Any]]:
    """Compute account-level behaviour metrics: flow amounts, pass-through ratio, distinct partners."""
    # Outflow metrics (from sender)
    out_agg = (
        transactions.groupBy("nameOrig")
        .agg(
            F.sum("amount").cast("double").alias("total_out_amount"),
            F.countDistinct("nameDest").cast("double").alias("distinct_receivers"),
        )
        .withColumnRenamed("nameOrig", "id")
    )

    # Inflow metrics (from receiver)
    in_agg = (
        transactions.groupBy("nameDest")
        .agg(
            F.sum("amount").cast("double").alias("total_in_amount"),
            F.countDistinct("nameOrig").cast("double").alias("distinct_senders"),
        )
        .withColumnRenamed("nameDest", "id")
    )

    joined = (
        vertices.select("id")
        .join(out_agg, on="id", how="left")
        .join(in_agg, on="id", how="left")
        .fillna(
            {
                "total_out_amount": 0.0,
                "total_in_amount": 0.0,
                "distinct_senders": 0.0,
                "distinct_receivers": 0.0,
            }
        )
    )

    # Pass-through ratio: outflow / inflow (guarded against division by zero)
    with_ratio = joined.withColumn(
        "pass_through_ratio",
        F.when(
            F.col("total_in_amount") > 0.0,
            F.col("total_out_amount") / F.col("total_in_amount"),
        ).otherwise(0.0),
    ).withColumn(
        "is_merchant",
        F.when(F.col("id").startswith("M"), 1.0).otherwise(0.0),
    )

    info = {"engine": "spark-native", "iterations": 1, "converged": True}
    return with_ratio, info


def extract_account_features(
    graph: Any,
    transactions: Optional[DataFrame] = None,
    timer_results: Optional[Dict[str, Any]] = None,
    spark: Optional[SparkSession] = None,
) -> Tuple[DataFrame, Dict[str, str], Dict[str, Any]]:
    """Compute all structural and behavioural graph features natively on Spark."""
    vertices = graph.vertices.select("id").distinct().cache()
    edges = graph.edges.select("src", "dst").cache()

    engines: Dict[str, str] = {}
    algo_info: Dict[str, Any] = {}

    # 1. Degrees
    with StageTimer("degrees", timer_results, spark):
        deg_df, d_info = compute_degrees(vertices, edges)
        deg_df = deg_df.cache()
        deg_df.count()
    engines["degrees"] = d_info["engine"]
    algo_info["degrees"] = d_info

    # 2. PageRank
    with StageTimer("pagerank", timer_results, spark):
        pr_df, pr_info = compute_pagerank(vertices, edges, num_iter=10)
        pr_df = pr_df.cache()
        pr_df.count()
    engines["pagerank"] = pr_info["engine"]
    algo_info["pagerank"] = pr_info

    # 3. Triangle Count
    with StageTimer("triangles", timer_results, spark):
        tc_df, tc_info = compute_triangle_count(vertices, edges)
        tc_df = tc_df.cache()
        tc_df.count()
    engines["triangles"] = tc_info["engine"]
    algo_info["triangles"] = tc_info

    # 4. Connected Components
    with StageTimer("components", timer_results, spark):
        cc_df, cc_info = compute_connected_components(vertices, edges, max_iter=30)
        cc_df = cc_df.cache()
        cc_df.count()
    engines["components"] = cc_info["engine"]
    algo_info["components"] = cc_info

    # 5. Label Propagation Communities
    with StageTimer("lpa", timer_results, spark):
        lpa_df, lpa_info = compute_label_propagation(vertices, edges, max_iter=5)
        lpa_df = lpa_df.cache()
        lpa_df.count()
    engines["lpa"] = lpa_info["engine"]
    algo_info["lpa"] = lpa_info

    # 6. Behavioural Flow Features (if transactions provided)
    behav_df = None
    if transactions is not None:
        with StageTimer("behaviour_features", timer_results, spark):
            behav_df, b_info = compute_behaviour_features(transactions, vertices)
            behav_df = behav_df.cache()
            behav_df.count()
        engines["behaviour"] = b_info["engine"]
        algo_info["behaviour"] = b_info

    # Combine all account features
    features = (
        vertices.join(deg_df, "id", "left")
        .join(pr_df, "id", "left")
        .join(tc_df, "id", "left")
        .join(cc_df, "id", "left")
        .join(lpa_df, "id", "left")
    )

    if behav_df is not None:
        features = features.join(behav_df, "id", "left")

    features = features.fillna(
        {
            "in_degree": 0.0,
            "out_degree": 0.0,
            "pagerank": 0.15,
            "triangle_count": 0.0,
            "component_size": 1.0,
            "community_size": 1.0,
            "total_in_amount": 0.0,
            "total_out_amount": 0.0,
            "pass_through_ratio": 0.0,
            "distinct_senders": 0.0,
            "distinct_receivers": 0.0,
            "is_merchant": 0.0,
        }
    ).cache()

    return features, engines, algo_info


GRAPH_FEATURE_COLUMNS = [
    "origin_in_degree",
    "origin_out_degree",
    "origin_pagerank",
    "origin_triangle_count",
    "origin_component_size",
    "origin_community_size",
    "dest_in_degree",
    "dest_out_degree",
    "dest_pagerank",
    "dest_triangle_count",
    "dest_component_size",
    "dest_community_size",
    "same_community",
]


def add_transaction_graph_features(
    transactions: DataFrame, account_features: DataFrame
) -> DataFrame:
    """Enrich transactions with origin/destination graph features and balance error accounting."""
    # Prefix features for origin accounts
    orig_cols = [c for c in account_features.columns if c != "id"]
    orig_df = account_features.select(
        [F.col("id").alias("nameOrig")]
        + [F.col(c).alias(f"origin_{c}") for c in orig_cols]
    )

    # Prefix features for destination accounts
    dest_df = account_features.select(
        [F.col("id").alias("nameDest")]
        + [F.col(c).alias(f"dest_{c}") for c in orig_cols]
    )

    enriched = transactions.join(orig_df, on="nameOrig", how="left").join(
        dest_df, on="nameDest", how="left"
    )

    # Compute community and component intersection flags
    if "origin_community_id" in enriched.columns and "dest_community_id" in enriched.columns:
        enriched = enriched.withColumn(
            "same_community",
            F.when(
                F.col("origin_community_id") == F.col("dest_community_id"), 1.0
            ).otherwise(0.0),
        )
    if "origin_component_id" in enriched.columns and "dest_component_id" in enriched.columns:
        enriched = enriched.withColumn(
            "same_component",
            F.when(
                F.col("origin_component_id") == F.col("dest_component_id"), 1.0
            ).otherwise(0.0),
        )

    # Balance error accounting features
    enriched = (
        enriched.withColumn(
            "errorBalanceOrig",
            F.col("newbalanceOrig") + F.col("amount") - F.col("oldbalanceOrg"),
        )
        .withColumn(
            "errorBalanceDest",
            F.col("oldbalanceDest") + F.col("amount") - F.col("newbalanceDest"),
        )
        .withColumn(
            "is_merchant",
            F.when(F.col("nameDest").startswith("M"), 1.0).otherwise(0.0),
        )
    )

    return enriched
