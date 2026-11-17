"""Storage benchmark comparing CSV (explicit schema) vs partitioned and unpartitioned Parquet."""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import shutil
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

import psutil
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from data_ingestion import DATA_DIR, PAYMENTS_SCHEMA, ensure_dataset
from provenance import make_readonly, sha256, write_card, write_manifest
from run_info import write_run_info
from spark_session import create_spark_session
from to_parquet import dir_size_bytes

LOGGER = logging.getLogger("fraud_analytics.storage_benchmark")


def _load_df(spark: SparkSession, fmt: str, path: Path) -> DataFrame:
    if fmt == "csv":
        return (
            spark.read.option("header", True)
            .option("mode", "PERMISSIVE")
            .schema(PAYMENTS_SCHEMA)
            .csv(str(path))
        )
    return spark.read.parquet(str(path))


def _time_operation(fn) -> tuple[float, Any, int]:
    """Execute fn() and return (wall_seconds, return_value, min_available_ram_bytes)."""
    avail_before = psutil.virtual_memory().available
    t0 = time.perf_counter()
    res = fn()
    elapsed = time.perf_counter() - t0
    avail_after = psutil.virtual_memory().available
    return elapsed, res, min(avail_before, avail_after)


def run_storage_benchmark(
    csv_path: Path,
    partitioned_dir: Path,
    unpartitioned_dir: Path,
    run_dir: Path,
    repeats: int = 3,
    copy_to_outputs: bool = True,
) -> Dict[str, Any]:
    """Benchmark CSV vs Parquet (partitioned by type & unpartitioned) across 3 repeats."""
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_info(
        run_dir / "run_info.json",
        dataset_path=csv_path if csv_path.is_file() else None,
        config={
            "repeats": repeats,
            "csv_path": str(csv_path),
            "partitioned_dir": str(partitioned_dir),
            "unpartitioned_dir": str(unpartitioned_dir),
        },
    )

    targets = [
        ("csv", "csv", csv_path),
        ("parquet_partitioned", "parquet", partitioned_dir),
        ("parquet_unpartitioned", "parquet", unpartitioned_dir),
    ]

    csv_bytes = dir_size_bytes(csv_path)
    formats_result: Dict[str, Any] = {}
    min_avail_overall = psutil.virtual_memory().available

    spark = create_spark_session(app_name="PaySimStorageBenchmark")
    try:
        for label, fmt, path in targets:
            if not path.exists():
                raise FileNotFoundError(f"Benchmark input path does not exist: {path}")

            size_b = dir_size_bytes(path)
            size_mb = size_b / (1024 * 1024)
            ratio = (csv_bytes / size_b) if size_b > 0 else 0.0

            count_times: List[float] = []
            two_col_times: List[float] = []
            filter_times: List[float] = []
            row_count_val = None
            two_col_check = None
            filter_count_val = None
            min_avail_fmt = psutil.virtual_memory().available

            for r in range(1, repeats + 1):
                # 1. Count rows
                sec, cnt, avail = _time_operation(lambda: int(_load_df(spark, fmt, path).count()))
                count_times.append(round(sec, 4))
                row_count_val = cnt
                min_avail_fmt = min(min_avail_fmt, avail)

                # 2. Read two columns (amount, isFraud) with aggregation to force full column decode
                def _read_two_cols() -> Dict[str, float]:
                    row = (
                        _load_df(spark, fmt, path)
                        .select("amount", "isFraud")
                        .agg(F.sum("amount").alias("sum_amount"), F.sum("isFraud").alias("sum_fraud"))
                        .collect()[0]
                    )
                    return {"sum_amount": float(row["sum_amount"]), "sum_fraud": int(row["sum_fraud"])}

                sec, agg_res, avail = _time_operation(_read_two_cols)
                two_col_times.append(round(sec, 4))
                two_col_check = agg_res
                min_avail_fmt = min(min_avail_fmt, avail)

                # 3. Filter type == 'TRANSFER' and isFraud == 1
                def _filter_tf() -> int:
                    return int(
                        _load_df(spark, fmt, path)
                        .filter((F.col("type") == "TRANSFER") & (F.col("isFraud") == 1))
                        .count()
                    )

                sec, f_cnt, avail = _time_operation(_filter_tf)
                filter_times.append(round(sec, 4))
                filter_count_val = f_cnt
                min_avail_fmt = min(min_avail_fmt, avail)

                LOGGER.info(
                    "[%s repeat %d/%d] count=%.3fs two_cols=%.3fs filter=%.3fs",
                    label,
                    r,
                    repeats,
                    count_times[-1],
                    two_col_times[-1],
                    filter_times[-1],
                )

            min_avail_overall = min(min_avail_overall, min_avail_fmt)
            formats_result[label] = {
                "path": str(path),
                "size_bytes": size_b,
                "size_mb": round(size_mb, 2),
                "compression_ratio_vs_csv": round(ratio, 3),
                "row_count": row_count_val,
                "filter_transfer_fraud_count": filter_count_val,
                "two_columns_check": two_col_check,
                "min_available_ram_bytes": min_avail_fmt,
                "operations": {
                    "count_rows": {
                        "repeats_seconds": count_times,
                        "mean_seconds": round(statistics.mean(count_times), 4),
                        "min_seconds": min(count_times),
                        "max_seconds": max(count_times),
                    },
                    "read_two_columns": {
                        "columns": ["amount", "isFraud"],
                        "method": "select + sum aggregation to force disk/column decode",
                        "repeats_seconds": two_col_times,
                        "mean_seconds": round(statistics.mean(two_col_times), 4),
                        "min_seconds": min(two_col_times),
                        "max_seconds": max(two_col_times),
                    },
                    "filter_transfer_fraud": {
                        "predicate": "type == 'TRANSFER' and isFraud == 1",
                        "repeats_seconds": filter_times,
                        "mean_seconds": round(statistics.mean(filter_times), 4),
                        "min_seconds": min(filter_times),
                        "max_seconds": max(filter_times),
                    },
                },
            }
    finally:
        spark.stop()

    # Compute speedups vs CSV
    speedups: Dict[str, Dict[str, float]] = {}
    csv_ops = formats_result["csv"]["operations"]
    for label in ("parquet_partitioned", "parquet_unpartitioned"):
        ops = formats_result[label]["operations"]
        speedups[label] = {}
        for op_name in ("count_rows", "read_two_columns", "filter_transfer_fraud"):
            p_mean = ops[op_name]["mean_seconds"]
            c_mean = csv_ops[op_name]["mean_seconds"]
            speedups[label][op_name] = round(c_mean / p_mean, 3) if p_mean > 0 else 0.0

    payload: Dict[str, Any] = {
        "benchmark": "storage_csv_vs_parquet",
        "timestamp": datetime.datetime.now().isoformat(),
        "repeats": repeats,
        "min_available_ram_bytes": min_avail_overall,
        "formats": formats_result,
        "speedup_vs_csv": speedups,
        "caveats": [
            "Measured on a single Windows laptop using local[*] Spark session with default 6g driver memory and 48 shuffle partitions.",
            "OS page cache is warm after the first repeat; individual repeat values are retained alongside means.",
            "read_two_columns uses sum(amount), sum(isFraud) so Spark cannot answer solely from Parquet footer row counts.",
        ],
    }

    out_json = run_dir / "storage_benchmark.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Provenance CARD.md and SHA256SUMS.txt
    csv_hash = sha256(csv_path) if csv_path.is_file() else "directory"
    write_card(
        run_dir=run_dir,
        run_id=run_dir.name,
        purpose="Compare storage footprint and Spark query read latency across CSV (explicit schema), type-partitioned Parquet, and unpartitioned Parquet (3 repeats each).",
        command=f"python storage_benchmark.py --run-dir {run_dir} --repeats {repeats}",
        inputs=[
            ("PaySim CSV", f"{csv_path} (sha256: {csv_hash})"),
            ("Parquet (partitioned by type)", str(partitioned_dir)),
            ("Parquet (unpartitioned)", str(unpartitioned_dir)),
        ],
        outputs=[("storage_benchmark.json", "Measured file sizes and 3-repeat query latencies")],
        key_numbers=[
            ("CSV Size (MB)", "storage_benchmark.json", "formats.csv.size_mb"),
            ("Parquet Partitioned Size (MB)", "storage_benchmark.json", "formats.parquet_partitioned.size_mb"),
            ("Parquet Unpartitioned Size (MB)", "storage_benchmark.json", "formats.parquet_unpartitioned.size_mb"),
            ("CSV Count Mean (s)", "storage_benchmark.json", "formats.csv.operations.count_rows.mean_seconds"),
            ("Parquet Partitioned Count Mean (s)", "storage_benchmark.json", "formats.parquet_partitioned.operations.count_rows.mean_seconds"),
            ("CSV Two-Col Read Mean (s)", "storage_benchmark.json", "formats.csv.operations.read_two_columns.mean_seconds"),
            ("Parquet Partitioned Two-Col Read Mean (s)", "storage_benchmark.json", "formats.parquet_partitioned.operations.read_two_columns.mean_seconds"),
            ("CSV Filter TRANSFER+Fraud Mean (s)", "storage_benchmark.json", "formats.csv.operations.filter_transfer_fraud.mean_seconds"),
            ("Parquet Partitioned Filter Mean (s)", "storage_benchmark.json", "formats.parquet_partitioned.operations.filter_transfer_fraud.mean_seconds"),
        ],
        caveats=payload["caveats"],
        not_shown=[
            "Does not measure distributed HDFS/S3 network I/O or multi-node cluster disk throughput.",
            "Does not include downstream graph computation or model training time.",
        ],
    )
    write_manifest(run_dir)
    make_readonly(run_dir)

    if copy_to_outputs:
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_json, outputs_dir / "storage_benchmark.json")
        snap_dir = Path("results_snapshot")
        if snap_dir.exists():
            shutil.copy2(out_json, snap_dir / "storage_benchmark.json")

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CSV vs Parquet storage benchmark.")
    parser.add_argument("--csv", type=Path, default=None, help="Input PaySim CSV path.")
    parser.add_argument(
        "--partitioned-dir",
        type=Path,
        default=DATA_DIR / "parquet" / "paysim",
        help="Partitioned Parquet directory.",
    )
    parser.add_argument(
        "--unpartitioned-dir",
        type=Path,
        default=DATA_DIR / "parquet" / "paysim_unpartitioned",
        help="Unpartitioned Parquet directory.",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeats per query.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Immutable output run folder.")
    parser.add_argument("--no-copy-outputs", action="store_true", help="Do not copy JSON to outputs/.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    csv_path = args.csv or ensure_dataset()
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    run_dir = args.run_dir or (Path("runs") / f"{ts}_c3-storage")
    result = run_storage_benchmark(
        csv_path=csv_path,
        partitioned_dir=args.partitioned_dir,
        unpartitioned_dir=args.unpartitioned_dir,
        run_dir=run_dir,
        repeats=args.repeats,
        copy_to_outputs=not args.no_copy_outputs,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
