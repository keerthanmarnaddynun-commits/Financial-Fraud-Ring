"""Compute account-level graph metrics and transaction-level feature joins.

Includes native PySpark DataFrame and NetworkX fallbacks for PySpark 4.2.0+
when GraphFrames GraphX bindings are unavailable.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

LOGGER = logging.getLogger("fraud_analytics")

try:
    from graphframes import GraphFrame
except ImportError:
    GraphFrame = None


def extract_account_features(graph: Any) -> DataFrame:
    """Calculate degree, PageRank, triangles, components, and LPA communities.

    Attempts GraphFrames GraphX first; falls back to native PySpark DataFrame
    joins, aggregations, and NetworkX algorithms if GraphFrames is unavailable.
    """
    # 1. Degree Features
    in_deg = None
    out_deg = None
    try:
        if hasattr(graph, "inDegrees") and hasattr(graph, "outDegrees"):
            in_deg = graph.inDegrees.select("id", F.col("inDegree").cast("double"))
            out_deg = graph.outDegrees.select("id", F.col("outDegree").cast("double"))
        else:
            raise AttributeError("Native graph object; computing degrees via Spark DataFrames.")
    except Exception as error:
        LOGGER.warning("GraphFrames degrees unavailable (%s); falling back to native PySpark.", error)
        in_deg = (
            graph.edges.groupBy("dst")
            .agg(F.count("*").cast("double").alias("inDegree"))
            .withColumnRenamed("dst", "id")
        )
        out_deg = (
            graph.edges.groupBy("src")
            .agg(F.count("*").cast("double").alias("outDegree"))
            .withColumnRenamed("src", "id")
        )

    # 2. PageRank
    pr_df = None
    try:
        if hasattr(graph, "pageRank"):
            pr_result = graph.pageRank(resetProbability=0.15, tol=0.01)
            pr_df = pr_result.vertices.select("id", F.col("pagerank").cast("double"))
    except Exception as error:
        LOGGER.warning("GraphFrames pageRank failed (%s); falling back to native PySpark PageRank.", error)

    if pr_df is None:
        reset_prob = 0.15
        alpha = 1.0 - reset_prob
        ranks = graph.vertices.select("id").withColumn("pagerank", F.lit(1.0))
        out_counts = graph.edges.groupBy("src").agg(F.count("*").alias("out_cnt"))
        edges_with_out = graph.edges.select("src", "dst").join(out_counts, on="src", how="inner")

        for _ in range(5):
            contribs = (
                edges_with_out.join(ranks.withColumnRenamed("id", "src"), on="src")
                .select(F.col("dst"), (F.col("pagerank") / F.col("out_cnt")).alias("contrib"))
            )
            new_ranks = (
                contribs.groupBy("dst")
                .agg(F.sum("contrib").alias("sum_contrib"))
                .withColumn("pagerank", F.lit(reset_prob) + F.lit(alpha) * F.col("sum_contrib"))
                .withColumnRenamed("dst", "id")
            )
            ranks = graph.vertices.select("id").join(new_ranks, on="id", how="left").fillna({"pagerank": reset_prob})
        pr_df = ranks.select("id", F.col("pagerank").cast("double"))

    # 3. Triangle Count
    tc_df = None
    try:
        if hasattr(graph, "triangleCount"):
            tc_result = graph.triangleCount()
            tc_df = tc_result.select("id", F.col("count").cast("double").alias("triangle_count"))
    except Exception as error:
        LOGGER.warning("GraphFrames triangleCount failed (%s); falling back to native PySpark Triangle Count.", error)

    if tc_df is None:
        u_edges = (
            graph.edges.select(
                F.least("src", "dst").alias("u"),
                F.greatest("src", "dst").alias("v"),
            )
            .filter(F.col("u") != F.col("v"))
            .distinct()
        )
        e1 = u_edges.select(F.col("u").alias("u"), F.col("v").alias("v"))
        e2 = u_edges.select(F.col("u").alias("v"), F.col("v").alias("w"))
        e3 = u_edges.select(F.col("u").alias("u"), F.col("v").alias("w"))
        triangles = e1.join(e2, on="v").filter(F.col("u") < F.col("w")).join(e3, on=["u", "w"])

        t_u = triangles.groupBy(F.col("u").alias("id")).agg(F.count("*").alias("cnt"))
        t_v = triangles.groupBy(F.col("v").alias("id")).agg(F.count("*").alias("cnt"))
        t_w = triangles.groupBy(F.col("w").alias("id")).agg(F.count("*").alias("cnt"))
        t_all = t_u.unionByName(t_v).unionByName(t_w).groupBy("id").agg(F.sum("cnt").alias("triangle_count"))
        tc_df = (
            graph.vertices.select("id")
            .join(t_all, on="id", how="left")
            .fillna({"triangle_count": 0.0})
            .select("id", F.col("triangle_count").cast("double"))
        )

    # 4. Connected Components
    components = None
    try:
        if hasattr(graph, "connectedComponents"):
            components = graph.connectedComponents(algorithm="graphx").select(
                "id", F.col("component").cast("string").alias("component_id")
            )
    except Exception as error:
        LOGGER.warning("GraphFrames connectedComponents failed (%s); falling back to native Graph clustering.", error)

    if components is None:
        import networkx as nx

        edges_pdf = graph.edges.select("src", "dst").toPandas()
        g_cc = nx.Graph()
        g_cc.add_edges_from(zip(edges_pdf["src"], edges_pdf["dst"]))
        comp_map = {}
        for c_id, comp in enumerate(nx.connected_components(g_cc)):
            for node in comp:
                comp_map[node] = str(c_id)
        cc_pdf = pd.DataFrame(list(comp_map.items()), columns=["id", "component_id"])
        spark = graph.vertices.sparkSession
        components = spark.createDataFrame(cc_pdf)

    # 5. Label Propagation Communities
    communities = None
    try:
        if hasattr(graph, "labelPropagation"):
            communities = graph.labelPropagation(maxIter=5).select(
                "id", F.col("label").cast("string").alias("community_id")
            )
    except Exception as error:
        LOGGER.warning("GraphFrames labelPropagation failed (%s); falling back to native Community Detection.", error)

    if communities is None:
        import networkx as nx

        edges_pdf = graph.edges.select("src", "dst").toPandas()
        g_lpa = nx.Graph()
        g_lpa.add_edges_from(zip(edges_pdf["src"], edges_pdf["dst"]))
        comm_map = {}
        for c_id, comm in enumerate(nx.community.label_propagation_communities(g_lpa)):
            for node in comm:
                comm_map[node] = str(c_id)
        comm_pdf = pd.DataFrame(list(comm_map.items()), columns=["id", "community_id"])
        spark = graph.vertices.sparkSession
        communities = spark.createDataFrame(comm_pdf)

    features = (
        graph.vertices.join(in_deg, "id", "left")
        .join(out_deg, "id", "left")
        .join(pr_df, "id", "left")
        .join(tc_df, "id", "left")
        .join(components, "id", "left")
        .join(communities, "id", "left")
    )

    community_sizes = features.groupBy("community_id").agg(
        F.count("*").cast("double").alias("community_size")
    )
    component_sizes = features.groupBy("component_id").agg(
        F.count("*").cast("double").alias("component_size")
    )
    return (
        features.join(community_sizes, "community_id", "left")
        .join(component_sizes, "component_id", "left")
        .select(
            "id",
            F.coalesce("inDegree", F.lit(0.0)).alias("in_degree"),
            F.coalesce("outDegree", F.lit(0.0)).alias("out_degree"),
            F.coalesce("pagerank", F.lit(0.0)).alias("pagerank"),
            F.coalesce("triangle_count", F.lit(0.0)).alias("triangle_count"),
            F.coalesce("component_id", F.col("id")).alias("component_id"),
            F.coalesce("community_id", F.col("id")).alias("community_id"),
            F.coalesce("component_size", F.lit(1.0)).alias("component_size"),
            F.coalesce("community_size", F.lit(1.0)).alias("community_size"),
        )
    )


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
    """Attach source/destination account metrics and a community-match indicator."""
    metric_columns = (
        "in_degree",
        "out_degree",
        "pagerank",
        "triangle_count",
        "component_size",
        "community_size",
        "community_id",
    )
    origin = account_features.select(
        F.col("id").alias("origin_id"),
        *[F.col(column).alias(f"origin_{column}") for column in metric_columns],
    )
    destination = account_features.select(
        F.col("id").alias("dest_id"),
        *[F.col(column).alias(f"dest_{column}") for column in metric_columns],
    )
    return (
        transactions.join(origin, F.col("nameOrig") == F.col("origin_id"), "left")
        .join(destination, F.col("nameDest") == F.col("dest_id"), "left")
        .withColumn(
            "same_community",
            F.when(
                F.col("origin_community_id").isNotNull()
                & (F.col("origin_community_id") == F.col("dest_community_id")),
                1.0,
            ).otherwise(0.0),
        )
        .drop("origin_id", "dest_id", "dest_community_id")
    )


def compute_structural_graph_features(
    spark: SparkSession,
    graph: Any,
    pagerank_max_iter: int = 5,
    lpa_max_iter: int = 3,
) -> Tuple[DataFrame, Dict[str, Any]]:
    """Legacy helper computing structural features with benchmarks for compatibility."""
    benchmarks: Dict[str, Any] = {}
    t0 = time.perf_counter()
    enriched = extract_account_features(graph)
    benchmarks["total_duration_sec"] = round(time.perf_counter() - t0, 3)
    return enriched, benchmarks
