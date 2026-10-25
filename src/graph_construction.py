"""
graph_construction.py
Constructs a Spark GraphFrames network from clean transaction Parquet data.
Nodes represent financial accounts/entities and directed edges represent transactions
and temporal/relational links.
"""

from typing import Tuple
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from graphframes import GraphFrame

from src.utils import logger, SparkPerformanceTracker, get_spark_session


def build_vertices_dataframe(df_txns: DataFrame) -> DataFrame:
    """
    Extracts unique accounts from nameOrig and nameDest and aggregates entity-level features.
    """
    logger.info("Extracting vertices (accounts/entities)...")

    # Aggregate outbound features
    outbound = (
        df_txns.groupBy("nameOrig")
        .agg(
            F.count("amount").alias("out_txn_count"),
            F.sum("amount").alias("total_sent_amount"),
            F.avg("amount").alias("avg_sent_amount"),
            F.max("amount").alias("max_sent_amount")
        )
        .withColumnRenamed("nameOrig", "id")
    )

    # Aggregate inbound features
    inbound = (
        df_txns.groupBy("nameDest")
        .agg(
            F.count("amount").alias("in_txn_count"),
            F.sum("amount").alias("total_recv_amount"),
            F.avg("amount").alias("avg_recv_amount"),
            F.max("amount").alias("max_recv_amount")
        )
        .withColumnRenamed("nameDest", "id")
    )

    # Full outer join to cover all unique account entities
    vertices = (
        outbound.join(inbound, on="id", how="full_outer")
        .fillna({
            "out_txn_count": 0,
            "total_sent_amount": 0.0,
            "avg_sent_amount": 0.0,
            "max_sent_amount": 0.0,
            "in_txn_count": 0,
            "total_recv_amount": 0.0,
            "avg_recv_amount": 0.0,
            "max_recv_amount": 0.0
        })
        .withColumn("entity_type", F.when(F.col("id").startswith("M"), "MERCHANT").otherwise("CUSTOMER"))
        .withColumn("is_merchant", F.when(F.col("id").startswith("M"), 1).otherwise(0))
        .withColumn("net_flow", F.col("total_recv_amount") - F.col("total_sent_amount"))
        .withColumn("total_activity_count", F.col("out_txn_count") + F.col("in_txn_count"))
    )

    return vertices


def build_edges_dataframe(df_txns: DataFrame) -> DataFrame:
    """
    Builds directed transaction edges with temporal and relational metadata.
    """
    logger.info("Building directed transaction edges...")

    # Assign unique transaction ID
    edges = (
        df_txns
        .withColumn("txn_id", F.monotonically_increasing_id())
        .select(
            F.col("nameOrig").alias("src"),
            F.col("nameDest").alias("dst"),
            F.col("type").alias("relationship"),
            F.col("amount"),
            F.col("step"),
            F.col("oldbalanceOrg"),
            F.col("newbalanceOrig"),
            F.col("oldbalanceDest"),
            F.col("newbalanceDest"),
            F.col("errorBalanceOrig"),
            F.col("errorBalanceDest"),
            F.col("isFraud"),
            F.col("isFlaggedFraud"),
            F.col("txn_id"),
            F.log1p(F.col("amount")).alias("edge_weight")
        )
    )

    return edges


def construct_transaction_graph(
    spark: SparkSession,
    parquet_path: str,
    sample_fraction: float = 1.0
) -> Tuple[GraphFrame, DataFrame, DataFrame]:
    """
    Constructs and returns the GraphFrames network (GraphFrame, vertices, edges).
    """
    tracker = SparkPerformanceTracker(spark).start()
    logger.info(f"Constructing transaction graph from {parquet_path}...")

    df_txns = spark.read.parquet(parquet_path)
    if sample_fraction < 1.0:
        logger.info(f"Sampling {sample_fraction*100:.1f}% of transactions for graph construction...")
        df_txns = df_txns.sample(fraction=sample_fraction, seed=42)

    df_txns = df_txns.cache()

    vertices = build_vertices_dataframe(df_txns).cache()
    edges = build_edges_dataframe(df_txns).cache()

    graph = GraphFrame(vertices, edges)

    v_count = graph.vertices.count()
    e_count = graph.edges.count()

    metrics = tracker.stop()
    logger.info(
        f"Graph construction completed in {metrics['duration_seconds']}s. "
        f"Nodes (Accounts): {v_count:,} | Directed Edges (Transactions): {e_count:,}"
    )

    return graph, vertices, edges


if __name__ == "__main__":
    spark = get_spark_session("GraphConstructionStandalone")
    try:
        g, v, e = construct_transaction_graph(spark, "data/processed/transactions_clean.parquet")
        print(f"Constructed Graph: {v.count()} vertices, {e.count()} edges")
    finally:
        spark.stop()
