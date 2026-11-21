"""Helper script to run the 2% sample graph pipeline and keep the Spark UI alive for screenshots."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from data_ingestion import resolve_input_path
from graph_pipeline import run_graph_stages
from spark_session import create_spark_session

LOGGER = logging.getLogger("fraud_analytics.demo_run")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 2% sample pipeline with Spark UI enabled for screenshots."
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.02,
        help="Sample fraction for the demo run (default: 0.02).",
    )
    parser.add_argument(
        "--input-format",
        type=str,
        choices=["csv", "parquet"],
        default="parquet",
        help="Input dataset format (default: parquet).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Optional explicit input path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_test") / "demo_run",
        help="Directory for demo summary output (never touches outputs/).",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Exit immediately after the sample run without waiting for Enter.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_path = resolve_input_path(args.data, input_format=args.input_format)

    spark = create_spark_session(app_name="PaySimFraudAnalyticsDemo")
    try:
        ui_url = spark.sparkContext.uiWebUrl or "http://localhost:4040"
        print("Spark UI: http://localhost:4040", flush=True)
        if ui_url.rstrip("/") != "http://localhost:4040":
            print(f"Bound Spark UI URL: {ui_url}", flush=True)

        res = run_graph_stages(
            spark,
            df=None,
            input_path=input_path,
            input_format=args.input_format,
            sample_fraction=args.sample_fraction,
            seed=42,
            materialize_enriched=True,
        )

        summary = {
            "sample_fraction": args.sample_fraction,
            "input_format": args.input_format,
            "spark_application_id": spark.sparkContext.applicationId,
            "spark_ui_url": ui_url,
            "rows": res["rows"],
            "nodes": res["nodes"],
            "edges": res["edges"],
            "seconds": res["seconds"],
            "peak_rss_bytes": res["peak_rss_bytes"],
            "min_available_ram_bytes": res["min_available_ram_bytes"],
            "shuffle_bytes": res["shuffle_bytes"],
        }
        out_path = args.output_dir / "demo_run_summary.json"
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print(f"Sample run finished ({res['rows']:,} rows, {res['nodes']:,} nodes, {res['edges']:,} edges).", flush=True)
        print("Spark UI: http://localhost:4040", flush=True)
        print("Suggested screenshots to save under docs/:", flush=True)
        print("  1. Jobs page      (http://localhost:4040/jobs/)", flush=True)
        print("  2. Stages page    (http://localhost:4040/stages/)", flush=True)
        print("  3. SQL/DF page    (http://localhost:4040/SQL/)", flush=True)
        print(f"Summary saved to: {out_path}", flush=True)

        if not args.no_wait:
            input("Press Enter to stop the Spark session and exit...")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
