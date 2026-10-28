"""Build account vertices and transaction edges as a GraphFrame."""

from pyspark.sql import DataFrame
from graphframes import GraphFrame
from pyspark.sql import functions as F


def build_transaction_graph(transactions: DataFrame) -> GraphFrame:
    """Create a directed graph; each transaction is an account-to-account edge."""
    vertices = (
        transactions.select(F.col("nameOrig").alias("id"))
        .union(transactions.select(F.col("nameDest").alias("id")))
        .distinct()
    )
    edges = transactions.select(
        F.col("nameOrig").alias("src"),
        F.col("nameDest").alias("dst"),
    )
    return GraphFrame(vertices, edges)
