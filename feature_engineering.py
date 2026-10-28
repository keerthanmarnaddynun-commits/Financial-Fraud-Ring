"""Compute account-level graph metrics and transaction-level feature joins."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from graphframes import GraphFrame


def extract_account_features(graph: GraphFrame) -> DataFrame:
    """Calculate degree, PageRank, triangles, components, and LPA communities."""
    features = (
        graph.vertices.join(
            graph.inDegrees.select("id", F.col("inDegree").cast("double")),
            "id",
            "left",
        )
        .join(
            graph.outDegrees.select("id", F.col("outDegree").cast("double")),
            "id",
            "left",
        )
        .join(
            graph.pageRank(resetProbability=0.15, tol=0.01)
            .vertices.select("id", F.col("pagerank").cast("double")),
            "id",
            "left",
        )
        .join(
            graph.triangleCount().select(
                "id", F.col("count").cast("double").alias("triangle_count")
            ),
            "id",
            "left",
        )
    )
    components = graph.connectedComponents(algorithm="graphx").select(
        "id", F.col("component").cast("string").alias("component_id")
    )
    communities = graph.labelPropagation(maxIter=5).select(
        "id", F.col("label").cast("string").alias("community_id")
    )
    features = features.join(components, "id", "left").join(communities, "id", "left")

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
            "component_id",
            "community_id",
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
