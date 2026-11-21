"""Unit tests validating Spark-native graph algorithms against NetworkX on reference graphs."""

import networkx as nx
import pytest
from pyspark.sql import SparkSession

from feature_engineering import (
    compute_connected_components,
    compute_degrees,
    compute_label_propagation,
    compute_pagerank,
    compute_triangle_count,
)
from spark_session import create_spark_session


@pytest.fixture(scope="module")
def spark():
    session = create_spark_session("GraphAlgoTests", master="local[2]", driver_memory="2g")
    yield session
    session.stop()


def test_degrees_equal_networkx(spark: SparkSession):
    # Directed graph: a -> b, b -> c, c -> a, c -> d
    nx_g = nx.DiGraph([("a", "b"), ("b", "c"), ("c", "a"), ("c", "d")])
    nx_g.add_node("e")  # isolated node

    nodes = list(nx_g.nodes())
    edges = list(nx_g.edges())

    v_df = spark.createDataFrame([(n,) for n in nodes], ["id"])
    e_df = spark.createDataFrame(edges, ["src", "dst"])

    deg_df, info = compute_degrees(v_df, e_df)
    assert info["engine"] == "spark-native"
    assert info["converged"] is True

    spark_res = {r["id"]: (r["in_degree"], r["out_degree"]) for r in deg_df.collect()}

    for n in nodes:
        expected_in = float(nx_g.in_degree(n))
        expected_out = float(nx_g.out_degree(n))
        actual_in, actual_out = spark_res[n]
        assert actual_in == expected_in, f"In-degree mismatch for {n}"
        assert actual_out == expected_out, f"Out-degree mismatch for {n}"


def test_triangles_equal_networkx(spark: SparkSession):
    # Simple undirected graph with 2 triangles: (a,b,c) and (b,c,d)
    nx_g = nx.Graph([("a", "b"), ("b", "c"), ("c", "a"), ("b", "d"), ("c", "d"), ("d", "e")])
    nodes = list(nx_g.nodes())
    edges = list(nx_g.edges())

    v_df = spark.createDataFrame([(n,) for n in nodes], ["id"])
    # Feed directed edge pairs to simulate transaction edges
    e_df = spark.createDataFrame(edges, ["src", "dst"])

    tc_df, info = compute_triangle_count(v_df, e_df)
    assert info["engine"] == "spark-native"
    assert info["converged"] is True

    nx_tc = nx.triangles(nx_g)
    spark_tc = {r["id"]: int(r["triangle_count"]) for r in tc_df.collect()}

    for n in nodes:
        assert spark_tc[n] == nx_tc[n], f"Triangle count mismatch for {n}"


def test_connected_components_partition_equal_networkx(spark: SparkSession):
    # Graph with 2 disconnected components: {a, b, c} and {d, e}
    nx_g = nx.Graph([("a", "b"), ("b", "c"), ("d", "e")])
    nx_g.add_node("f")  # third component of size 1

    nodes = list(nx_g.nodes())
    edges = list(nx_g.edges())

    v_df = spark.createDataFrame([(n,) for n in nodes], ["id"])
    e_df = spark.createDataFrame(edges, ["src", "dst"])

    cc_df, info = compute_connected_components(v_df, e_df, max_iter=10)
    assert info["engine"] == "spark-native"
    assert info["converged"] is True

    spark_cc = {r["id"]: r["component_id"] for r in cc_df.collect()}

    # Check partition equivalence: two nodes share component iff connected in NetworkX
    for u in nodes:
        for v in nodes:
            expected_connected = nx.has_path(nx_g, u, v)
            actual_connected = spark_cc[u] == spark_cc[v]
            assert actual_connected == expected_connected, f"Component mismatch between {u} and {v}"


def test_pagerank_within_tolerance(spark: SparkSession):
    # 20-node directed graph (cycle + chords)
    nx_g = nx.DiGraph()
    for i in range(20):
        nx_g.add_edge(f"n{i}", f"n{(i+1)%20}")
        if i % 3 == 0:
            nx_g.add_edge(f"n{i}", f"n{(i+5)%20}")

    nodes = list(nx_g.nodes())
    edges = list(nx_g.edges())

    v_df = spark.createDataFrame([(n,) for n in nodes], ["id"])
    e_df = spark.createDataFrame(edges, ["src", "dst"])

    pr_df, info = compute_pagerank(v_df, e_df, num_iter=30, damping=0.85)
    assert info["engine"] == "spark-native"
    assert info["converged"] is True

    spark_ranks = {r["id"]: float(r["pagerank"]) for r in pr_df.collect()}
    nx_ranks = nx.pagerank(nx_g, alpha=0.85, max_iter=200, tol=1e-6)

    # Normalize Spark PageRank (sum = 1) for direct comparison with NetworkX
    tot_spark = sum(spark_ranks.values())
    spark_norm = {k: v / tot_spark for k, v in spark_ranks.items()}

    for n in nodes:
        diff = abs(spark_norm[n] - nx_ranks[n])
        assert diff < 1e-3, f"PageRank diff for {n} exceeds tolerance: {diff:.6f}"


def test_lpa_subsets_of_connected_components(spark: SparkSession):
    # Graph with 2 disconnected cliques: {a, b, c, d} and {x, y, z}
    edges = [
        ("a", "b"), ("b", "c"), ("c", "a"), ("a", "d"), ("b", "d"),
        ("x", "y"), ("y", "z"), ("z", "x"),
    ]
    nodes = ["a", "b", "c", "d", "x", "y", "z"]

    v_df = spark.createDataFrame([(n,) for n in nodes], ["id"])
    e_df = spark.createDataFrame(edges, ["src", "dst"])

    cc_df, _ = compute_connected_components(v_df, e_df)
    lpa_df, info = compute_label_propagation(v_df, e_df, max_iter=5)
    assert info["engine"] == "spark-native"

    cc_map = {r["id"]: r["component_id"] for r in cc_df.collect()}
    lpa_map = {r["id"]: r["community_id"] for r in lpa_df.collect()}

    # Group nodes by LPA community and verify all nodes in each community share the same connected component
    comm_to_nodes = {}
    for node, comm in lpa_map.items():
        comm_to_nodes.setdefault(comm, []).append(node)

    for comm, members in comm_to_nodes.items():
        comp_ids = {cc_map[m] for m in members}
        assert len(comp_ids) == 1, f"LPA community {comm} spans multiple connected components: {comp_ids}"
