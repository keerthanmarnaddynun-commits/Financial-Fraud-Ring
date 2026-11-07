"""Build account vertices and transaction edges as a GraphFrame or native graph."""

from __future__ import annotations

import logging
from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

LOGGER = logging.getLogger("fraud_analytics")

try:
    from graphframes import GraphFrame
except ImportError:
    GraphFrame = None


class NativeGraph:
    """Native PySpark DataFrame graph wrapper when GraphFrames is unavailable."""

    def __init__(self, vertices: DataFrame, edges: DataFrame) -> None:
        self.vertices = vertices
        self.edges = edges


def build_transaction_graph(transactions: DataFrame) -> Any:
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

    graph = None
    if GraphFrame is not None:
        try:
            graph = GraphFrame(vertices, edges)
        except Exception as error:
            LOGGER.warning(
                "GraphFrame construction unavailable or failed (%s). Using native graph wrapper.",
                error,
            )

    if graph is None:
        graph = NativeGraph(vertices, edges)

    return graph
