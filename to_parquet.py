"""Convert the PaySim CSV dataset to partitioned and unpartitioned Parquet copies."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict

from data_ingestion import DATA_DIR, PAYMENTS_SCHEMA, ensure_dataset
from spark_session import create_spark_session

LOGGER = logging.getLogger("fraud_analytics.to_parquet")


def dir_size_bytes(path: Path) -> int:
    """Return total byte size of a file or all files under a directory."""
    if path.is_file():
        return path.stat().st_size
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def convert_csv_to_parquet(
    csv_path: Path | None = None,
    partitioned_dir: Path | None = None,
    unpartitioned_dir: Path | None = None,
) -> Dict[str, object]:
    """Convert PaySim CSV once to partitioned (by type) and unpartitioned Parquet."""
    src_csv = csv_path or ensure_dataset()
    part_dir = partitioned_dir or (DATA_DIR / "parquet" / "paysim")
    unpart_dir = unpartitioned_dir or (DATA_DIR / "parquet" / "paysim_unpartitioned")

    part_dir.parent.mkdir(parents=True, exist_ok=True)
    unpart_dir.parent.mkdir(parents=True, exist_ok=True)

    spark = create_spark_session(app_name="PaySimToParquet")
    try:
        t0 = time.perf_counter()
        raw = (
            spark.read.option("header", True)
            .option("mode", "PERMISSIVE")
            .schema(PAYMENTS_SCHEMA)
            .csv(str(src_csv))
        )
        row_count = int(raw.count())
        read_sec = time.perf_counter() - t0
        LOGGER.info("Loaded CSV %s (%s rows) in %.2fs", src_csv, f"{row_count:,}", read_sec)

        t1 = time.perf_counter()
        raw.write.mode("overwrite").partitionBy("type").parquet(str(part_dir))
        part_sec = time.perf_counter() - t1
        LOGGER.info("Wrote partitioned Parquet to %s in %.2fs", part_dir, part_sec)

        t2 = time.perf_counter()
        raw.write.mode("overwrite").parquet(str(unpart_dir))
        unpart_sec = time.perf_counter() - t2
        LOGGER.info("Wrote unpartitioned Parquet to %s in %.2fs", unpart_dir, unpart_sec)

        part_count = int(spark.read.parquet(str(part_dir)).count())
        unpart_count = int(spark.read.parquet(str(unpart_dir)).count())
        if part_count != row_count or unpart_count != row_count:
            raise RuntimeError(
                f"Row count mismatch after Parquet conversion: csv={row_count}, "
                f"partitioned={part_count}, unpartitioned={unpart_count}"
            )

        summary = {
            "csv_path": str(src_csv),
            "partitioned_path": str(part_dir),
            "unpartitioned_path": str(unpart_dir),
            "rows": row_count,
            "csv_size_bytes": dir_size_bytes(src_csv),
            "partitioned_size_bytes": dir_size_bytes(part_dir),
            "unpartitioned_size_bytes": dir_size_bytes(unpart_dir),
            "csv_read_seconds": round(read_sec, 4),
            "partitioned_write_seconds": round(part_sec, 4),
            "unpartitioned_write_seconds": round(unpart_sec, 4),
        }
        return summary
    finally:
        spark.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PaySim CSV to Parquet.")
    parser.add_argument("--csv", type=Path, default=None, help="Input CSV path.")
    parser.add_argument(
        "--partitioned-dir",
        type=Path,
        default=DATA_DIR / "parquet" / "paysim",
        help="Output directory for type-partitioned Parquet.",
    )
    parser.add_argument(
        "--unpartitioned-dir",
        type=Path,
        default=DATA_DIR / "parquet" / "paysim_unpartitioned",
        help="Output directory for unpartitioned Parquet.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    summary = convert_csv_to_parquet(args.csv, args.partitioned_dir, args.unpartitioned_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
