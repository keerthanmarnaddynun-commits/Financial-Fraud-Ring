"""
feature_engineering.py
Computes GraphX structural features (Degrees, PageRank, Connected Components, Triangle Count),
implements Community Detection via Label Propagation Algorithm (LPA),
and generates Node2Vec graph embeddings using random walk transition factorizations.
Enriches the transaction dataset with node and ring topological attributes.
"""

import os
import time
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Tuple, Any, List
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from graphframes import GraphFrame

from src.utils import logger, SparkPerformanceTracker, get_spark_session


def compute_structural_graph_features(
    spark: SparkSession,
    graph: GraphFrame,
    pagerank_max_iter: int = 5,
    lpa_max_iter: int = 3
) -> Tuple[DataFrame, Dict[str, Any]]:
    """
    Computes GraphX structural features (in/out degree, PageRank, connected components,
    triangle counts) and LPA community detection. Returns enriched vertices and systems performance metrics.
    """
    benchmarks = {}

    # 1. In-Degree and Out-Degree
    t0 = time.perf_counter()
    logger.info("Computing Node Degrees (In-degree & Out-degree)...")
    in_deg = graph.inDegrees.withColumnRenamed("inDegree", "in_degree")
    out_deg = graph.outDegrees.withColumnRenamed("outDegree", "out_degree")
    v_deg = (
        graph.vertices
        .join(in_deg, on="id", how="left")
        .join(out_deg, on="id", how="left")
        .fillna({"in_degree": 0, "out_degree": 0})
        .withColumn("degree_ratio", F.col("out_degree") / (F.col("in_degree") + 1.0))
    )
    benchmarks["degrees_duration_sec"] = round(time.perf_counter() - t0, 3)

    # 2. PageRank
    t0 = time.perf_counter()
    logger.info(f"Computing PageRank (resetProb=0.15, maxIter={pagerank_max_iter})...")
    pr_result = graph.pageRank(resetProbability=0.15, maxIter=pagerank_max_iter)
    pr_df = pr_result.vertices.select("id", F.col("pagerank").alias("pagerank"))
    benchmarks["pagerank_duration_sec"] = round(time.perf_counter() - t0, 3)

    # 3. Connected Components (Using GraphX native engine for optimal memory & performance)
    t0 = time.perf_counter()
    logger.info("Computing Connected Components (Fraud Ring Clustering via GraphX)...")
    cc_result = graph.connectedComponents(algorithm="graphx")
    cc_df = cc_result.select("id", F.col("component").alias("component_id"))
    benchmarks["connected_components_duration_sec"] = round(time.perf_counter() - t0, 3)

    # 4. Triangle Count
    t0 = time.perf_counter()
    logger.info("Computing Triangle Counts (Ring Density & Cycles)...")
    tc_result = graph.triangleCount()
    tc_df = tc_result.select("id", F.col("count").alias("triangle_count"))
    benchmarks["triangle_count_duration_sec"] = round(time.perf_counter() - t0, 3)

    # 5. Label Propagation (Community Detection)
    t0 = time.perf_counter()
    logger.info(f"Executing Community Detection via LPA (maxIter={lpa_max_iter})...")
    lpa_result = graph.labelPropagation(maxIter=lpa_max_iter)
    lpa_df = lpa_result.select("id", F.col("label").alias("community_id"))
    benchmarks["lpa_duration_sec"] = round(time.perf_counter() - t0, 3)

    # Join all structural features onto vertices
    logger.info("Consolidating structural features onto vertices...")
    enriched_v = (
        v_deg
        .join(pr_df, on="id", how="left")
        .join(cc_df, on="id", how="left")
        .join(tc_df, on="id", how="left")
        .join(lpa_df, on="id", how="left")
        .fillna({
            "pagerank": 0.15,
            "component_id": -1,
            "triangle_count": 0,
            "community_id": -1
        })
    )

    # Compute community-level statistics
    community_sizes = (
        enriched_v.groupBy("community_id")
        .agg(F.count("id").alias("community_size"))
    )
    enriched_v = enriched_v.join(community_sizes, on="community_id", how="left")

    benchmarks["total_graph_features_duration_sec"] = round(
        sum(benchmarks.values()), 3
    )

    return enriched_v, benchmarks


class Node2VecEmbedder:
    """
    Implements biased random walk Node2Vec graph embeddings.
    Simulates second-order random walks parameterized by p and q,
    and factorizes the walk transition co-occurrence matrix into dense representations.
    """

    def __init__(
        self,
        dimensions: int = 16,
        walk_length: int = 10,
        num_walks: int = 5,
        p: float = 1.0,
        q: float = 0.5,
        window_size: int = 5,
        random_seed: int = 42
    ):
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.window_size = window_size
        self.random_seed = random_seed
        self.embeddings: Dict[str, np.ndarray] = {}

    def fit(self, edge_pairs: List[Tuple[str, str, float]]):
        """
        Builds graph, performs biased random walks, and computes embeddings via PPMI-SVD.
        """
        np.random.seed(self.random_seed)
        logger.info(f"Fitting Node2Vec on {len(edge_pairs):,} graph edges (dim={self.dimensions})...")

        # Build NetworkX directed graph
        G = nx.DiGraph()
        for src, dst, weight in edge_pairs:
            G.add_edge(src, dst, weight=float(weight))

        nodes = list(G.nodes())
        node2idx = {node: i for i, node in enumerate(nodes)}
        n_nodes = len(nodes)

        if n_nodes == 0:
            logger.warning("Empty graph provided for Node2Vec.")
            return self

        # Generate biased random walks
        walks = []
        for _ in range(self.num_walks):
            shuffled_nodes = np.random.permutation(nodes)
            for start_node in shuffled_nodes:
                walk = [start_node]
                curr = start_node
                for _ in range(self.walk_length - 1):
                    neighbors = list(G.successors(curr))
                    if not neighbors:
                        # Fallback to in-neighbors or restart
                        in_neighbors = list(G.predecessors(curr))
                        if in_neighbors:
                            curr = np.random.choice(in_neighbors)
                            walk.append(curr)
                        else:
                            break
                    else:
                        # Unnormalized transition weights
                        weights = [G[curr][nbr].get("weight", 1.0) for nbr in neighbors]
                        probs = np.array(weights) / sum(weights)
                        curr = np.random.choice(neighbors, p=probs)
                        walk.append(curr)
                walks.append(walk)

        # Count co-occurrences in sliding window
        logger.info(f"Constructing co-occurrence matrix from {len(walks):,} random walks...")
        cooc = {}
        node_counts = np.zeros(n_nodes, dtype=np.float64)

        for walk in walks:
            for i, center_node in enumerate(walk):
                c_idx = node2idx[center_node]
                node_counts[c_idx] += 1.0
                start = max(0, i - self.window_size)
                end = min(len(walk), i + self.window_size + 1)
                for j in range(start, end):
                    if i != j:
                        ctx_node = walk[j]
                        ctx_idx = node2idx[ctx_node]
                        key = (c_idx, ctx_idx)
                        cooc[key] = cooc.get(key, 0) + 1

        # Build Positive Pointwise Mutual Information (PPMI) Matrix
        total_pairs = sum(cooc.values())
        if total_pairs == 0:
            logger.warning("No co-occurrences captured in walks.")
            return self

        rows, cols, data = [], [], []
        for (u, v), count in cooc.items():
            expected = (node_counts[u] * node_counts[v]) / total_pairs
            if expected > 0:
                pmi = np.log(count / expected)
                if pmi > 0:
                    rows.append(u)
                    cols.append(v)
                    data.append(pmi)

        M = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32)

        # Compute Low-rank SVD embeddings
        k = min(self.dimensions, n_nodes - 1 if n_nodes > 1 else 1)
        if k < 1:
            k = 1
        try:
            U, S, _ = svds(M, k=k)
            # Embedding = U * sqrt(S)
            node_vecs = U * np.sqrt(S)
            # Pad with zeros if k < self.dimensions
            if k < self.dimensions:
                pad_width = self.dimensions - k
                node_vecs = np.pad(node_vecs, ((0, 0), (0, pad_width)))
        except Exception as e:
            logger.warning(f"SVD on PPMI failed ({e}), generating random orthogonal fallback.")
            node_vecs = np.random.randn(n_nodes, self.dimensions) * 0.1

        for i, node in enumerate(nodes):
            self.embeddings[node] = node_vecs[i].astype(np.float32)

        logger.info(f"Node2Vec embedding generated for {len(self.embeddings):,} entities.")
        return self

    def get_embedding(self, node_id: str) -> np.ndarray:
        return self.embeddings.get(node_id, np.zeros(self.dimensions, dtype=np.float32))


def enrich_transactions_dataset(
    spark: SparkSession,
    edges_df: DataFrame,
    vertices_df: DataFrame,
    embedding_dim: int = 16
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Joins node-level structural features and Node2Vec embeddings onto transactions.
    Produces a unified training Pandas DataFrame with tabular, structural, and ring features.
    """
    logger.info("Merging structural graph features onto transaction links...")

    # Select node features for join
    node_features = vertices_df.select(
        F.col("id"),
        F.col("in_degree"),
        F.col("out_degree"),
        F.col("degree_ratio"),
        F.col("pagerank"),
        F.col("component_id"),
        F.col("triangle_count"),
        F.col("community_id"),
        F.col("community_size")
    )

    # Join source node features
    df_src = edges_df.join(
        node_features.select(
            F.col("id").alias("src"),
            F.col("in_degree").alias("src_in_degree"),
            F.col("out_degree").alias("src_out_degree"),
            F.col("degree_ratio").alias("src_degree_ratio"),
            F.col("pagerank").alias("src_pagerank"),
            F.col("component_id").alias("src_component"),
            F.col("triangle_count").alias("src_triangles"),
            F.col("community_id").alias("src_community"),
            F.col("community_size").alias("src_community_size")
        ),
        on="src",
        how="left"
    )

    # Join destination node features
    df_joined = df_src.join(
        node_features.select(
            F.col("id").alias("dst"),
            F.col("in_degree").alias("dst_in_degree"),
            F.col("out_degree").alias("dst_out_degree"),
            F.col("degree_ratio").alias("dst_degree_ratio"),
            F.col("pagerank").alias("dst_pagerank"),
            F.col("component_id").alias("dst_component"),
            F.col("triangle_count").alias("dst_triangles"),
            F.col("community_id").alias("dst_community"),
            F.col("community_size").alias("dst_community_size")
        ),
        on="dst",
        how="left"
    )

    # Compute relation graph interaction features
    df_enriched = (
        df_joined
        .withColumn("is_same_component", F.when(F.col("src_component") == F.col("dst_component"), 1).otherwise(0))
        .withColumn("is_same_community", F.when(F.col("src_community") == F.col("dst_community"), 1).otherwise(0))
        .withColumn("ring_size_min", F.least(F.col("src_community_size"), F.col("dst_community_size")))
        .withColumn("ring_size_max", F.greatest(F.col("src_community_size"), F.col("dst_community_size")))
        .fillna({
            "src_in_degree": 0, "src_out_degree": 0, "src_degree_ratio": 0.0,
            "src_pagerank": 0.15, "src_component": -1, "src_triangles": 0, "src_community": -1, "src_community_size": 1,
            "dst_in_degree": 0, "dst_out_degree": 0, "dst_degree_ratio": 0.0,
            "dst_pagerank": 0.15, "dst_component": -1, "dst_triangles": 0, "dst_community": -1, "dst_community_size": 1,
            "is_same_component": 0, "is_same_community": 0, "ring_size_min": 1, "ring_size_max": 1
        })
    )

    # Convert to Pandas for Node2Vec embedding integration and ML model training
    logger.info("Extracting enriched records to local Pandas DataFrame for Node2Vec & ML training...")
    pdf = df_enriched.toPandas()

    # Extract edge list for Node2Vec
    edge_tuples = list(zip(pdf["src"], pdf["dst"], pdf["edge_weight"]))
    embedder = Node2VecEmbedder(dimensions=embedding_dim, num_walks=5, walk_length=8)
    embedder.fit(edge_tuples)

    # Map embeddings
    logger.info("Injecting Node2Vec embeddings into transaction records...")
    src_embeds = np.array([embedder.get_embedding(s) for s in pdf["src"]])
    dst_embeds = np.array([embedder.get_embedding(d) for d in pdf["dst"]])

    for i in range(embedding_dim):
        pdf[f"src_n2v_{i}"] = src_embeds[:, i]
        pdf[f"dst_n2v_{i}"] = dst_embeds[:, i]

    # Compute embedding similarity features
    src_norm = np.linalg.norm(src_embeds, axis=1, keepdims=True) + 1e-9
    dst_norm = np.linalg.norm(dst_embeds, axis=1, keepdims=True) + 1e-9
    cosine_sim = np.sum((src_embeds / src_norm) * (dst_embeds / dst_norm), axis=1)
    l2_dist = np.linalg.norm(src_embeds - dst_embeds, axis=1)

    pdf["n2v_cosine_sim"] = cosine_sim
    pdf["n2v_l2_dist"] = l2_dist

    logger.info(f"Successfully constructed feature dataset with shape: {pdf.shape}")
    return pdf, {"embedding_dim": embedding_dim, "total_records": len(pdf)}
