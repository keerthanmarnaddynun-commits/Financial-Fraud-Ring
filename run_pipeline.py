"""
run_pipeline.py
Master execution pipeline for Financial Fraud Analytics.
Executes end-to-end data ingestion, GraphFrames graph construction,
GraphX & Node2Vec feature engineering, comparative model training (Baseline, M1, M2, M3),
and comprehensive transaction/ring/systems evaluation.
"""

import os
import sys
import time
import argparse
from typing import Dict, Any, Optional

from src.utils import logger, get_spark_session, setup_environment
from src.data_ingestion import load_and_clean_data, run_ingestion
from src.graph_construction import construct_transaction_graph
from src.feature_engineering import compute_structural_graph_features, enrich_transactions_dataset
from src.model_training import train_comparative_models
from src.evaluation import run_full_evaluation


def print_banner(stage_num: int, title: str):
    width = 75
    sep = "=" * width
    print("\n" + sep)
    print(f" STAGE {stage_num}: {title.upper()}")
    print(sep + "\n")


def print_summary_table(df, title: str):
    width = 75
    print("\n" + "-" * width)
    print(f" {title.upper()}")
    print("-" * width)
    if hasattr(df, "to_string"):
        print(df.to_string(index=False))
    else:
        print(df)
    print("-" * width + "\n")


def execute_pipeline(
    num_records: int = 35000,
    model_type: str = "xgboost",
    force_synthetic: bool = False,
    run_scaling_benchmark: bool = True,
    input_csv_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes the entire Financial Fraud Analytics workflow.
    """
    total_pipeline_start = time.perf_counter()
    setup_environment()
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_dir, "data")
    models_dir = os.path.join(project_dir, "output", "models")
    metrics_dir = os.path.join(project_dir, "output", "metrics")

    # ---------------------------------------------------------
    # STAGE 1: Spark Session Initialization
    # ---------------------------------------------------------
    print_banner(1, "Spark Session & Environment Initialization")
    logger.info("Initializing PySpark Session with GraphFrames support...")
    spark = get_spark_session(
        app_name="FraudAnalyticsPipeline",
        driver_memory="4g",
        shuffle_partitions=8
    )
    logger.info(f"PySpark {spark.version} Session online. Master: {spark.sparkContext.master}")

    clean_parquet_path = None
    try:
        # ---------------------------------------------------------
        # STAGE 2: Data Ingestion & Cleaning
        # ---------------------------------------------------------
        print_banner(2, "Data Ingestion, Validation & Parquet Persistence")
        if input_csv_path:
            clean_parquet_path = os.path.join(
                data_dir, "processed", "transactions_clean.parquet"
            )
            load_and_clean_data(spark, input_csv_path, clean_parquet_path)
        else:
            clean_parquet_path = run_ingestion(
                spark=spark,
                data_dir=data_dir,
                num_synthetic_records=num_records,
                force_synthetic=force_synthetic
            )
        logger.info(f"Optimized Parquet dataset persisted at: {clean_parquet_path}")

        # ---------------------------------------------------------
        # STAGE 3: Spark GraphFrames Network Construction
        # ---------------------------------------------------------
        print_banner(3, "Graph Construction (Entities & Transaction Edges)")
        graph, vertices_df, edges_df = construct_transaction_graph(
            spark=spark,
            parquet_path=clean_parquet_path
        )
        v_count = graph.vertices.count()
        e_count = graph.edges.count()
        logger.info(f"Graph Construction Completed: {v_count:,} accounts/nodes, {e_count:,} directed edges.")

        # ---------------------------------------------------------
        # STAGE 4: Feature Engineering (GraphX, LPA & Node2Vec)
        # ---------------------------------------------------------
        print_banner(4, "Feature Engineering (GraphX, LPA Communities & Node2Vec)")
        enriched_vertices, graph_benchmarks = compute_structural_graph_features(
            spark=spark,
            graph=graph,
            pagerank_max_iter=5,
            lpa_max_iter=3
        )

        df_enriched, feat_meta = enrich_transactions_dataset(
            spark=spark,
            edges_df=edges_df,
            vertices_df=enriched_vertices,
            embedding_dim=16
        )
        logger.info(
            f"Feature Enrichment Complete: {len(df_enriched):,} transactions enriched with "
            f"{df_enriched.shape[1]} features (Tabular + GraphX + Node2Vec + LPA)."
        )

        # ---------------------------------------------------------
        # STAGE 5: Comparative Model Training
        # ---------------------------------------------------------
        print_banner(5, "Comparative Model Training (Baseline vs M1 vs M2 vs M3)")
        training_results = train_comparative_models(
            df=df_enriched,
            embedding_dim=16,
            model_type=model_type,
            output_dir=models_dir
        )

        # ---------------------------------------------------------
        # STAGE 6: Model Evaluation & Spark Systems Benchmarking
        # ---------------------------------------------------------
        print_banner(6, "Evaluation (Transaction-Level, Ring-Level & Spark Scaling)")
        eval_results = run_full_evaluation(
            spark=spark,
            training_results=training_results,
            clean_parquet_path=clean_parquet_path,
            output_dir=metrics_dir,
            run_scaling_benchmark=run_scaling_benchmark
        )

        # Print Executive Summaries
        print_summary_table(
            eval_results["df_txn_metrics"],
            "Transaction-Level Performance Comparison"
        )

        print_summary_table(
            eval_results["df_ring_metrics"],
            "Fraud Ring / Community Detection Metrics"
        )

        if eval_results["df_scaling"] is not None:
            print_summary_table(
                eval_results["df_scaling"],
                "Spark Graph Systems Scaling & Resource Overhead"
            )

        total_elapsed = round(time.perf_counter() - total_pipeline_start, 2)
        print("\n" + "=" * 75)
        print(f" PIPELINE EXECUTION SUCCESSFUL | TOTAL TIME: {total_elapsed}s")
        print("=" * 75 + "\n")

        return eval_results

    finally:
        logger.info("Stopping PySpark Session and releasing resources...")
        spark.stop()


def main():
    parser = argparse.ArgumentParser(description="Financial Fraud Analytics Pipeline")
    parser.add_argument("--num-records", type=int, default=30000, help="Number of records for synthetic dataset")
    parser.add_argument("--model-type", type=str, default="xgboost", choices=["xgboost", "random_forest"], help="ML classifier")
    parser.add_argument("--force-synthetic", action="store_true", help="Force synthetic dataset generation")
    parser.add_argument("--skip-scaling", action="store_true", help="Skip Spark scaling benchmark")
    parser.add_argument("--input-csv", type=str, help="Use a PaySim-compatible CSV file")

    args = parser.parse_args()

    execute_pipeline(
        num_records=args.num_records,
        model_type=args.model_type,
        force_synthetic=args.force_synthetic,
        run_scaling_benchmark=not args.skip_scaling,
        input_csv_path=args.input_csv
    )


if __name__ == "__main__":
    main()
