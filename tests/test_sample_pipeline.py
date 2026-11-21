"""Tests validating schema ingestion, data quality, and leakage-free splits on sample_1000.csv."""

from pathlib import Path

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from data_ingestion import build_data_quality_report, load_and_validate_data
from spark_session import create_spark_session
from splits import create_group_dest_split, create_time_split

SAMPLE_CSV = Path("tests") / "data" / "sample_1000.csv"


@pytest.fixture(scope="module")
def spark():
    session = create_spark_session(app_name="test_sample_pipeline", master="local[2]")
    yield session
    session.stop()


def test_schema_validation_keeps_merchants(spark):
    """Verify that schema validation preserves merchant payment transactions."""
    assert SAMPLE_CSV.exists(), f"Missing test dataset at {SAMPLE_CSV}"

    df = load_and_validate_data(spark, SAMPLE_CSV)
    count = df.count()
    assert count > 0, "No rows kept after validation"

    # Verify merchant payments exist
    merchant_payments = df.filter(
        (F.col("type") == "PAYMENT") & F.col("nameDest").startswith("M")
    ).count()
    assert merchant_payments > 0, "Merchant payments were incorrectly dropped"
    assert "is_merchant" in df.columns
    merchant_flag_count = df.filter(F.col("is_merchant") == 1).count()
    assert merchant_flag_count == merchant_payments


def test_data_quality_counts_add_up(spark):
    """Verify that data quality report counts balance exactly: read == kept + dropped."""
    assert SAMPLE_CSV.exists()
    report = build_data_quality_report(spark, SAMPLE_CSV)

    rows_read = report["rows_read"]
    rows_kept = report["rows_kept"]
    dropped = report["dropped_by_rule"]

    total_accounted = rows_kept + sum(dropped.values())
    assert rows_read == total_accounted, (
        f"Data quality counts do not balance: rows_read={rows_read} vs accounted={total_accounted}"
    )

    assert report["rows_read"] == 1000
    assert report["fraud_rows"] == 30
    assert report["by_type"]["PAYMENT"] == 250
    assert report["distinct_accounts"] > 0


def test_splits_no_leakage_on_sample_1000(spark):
    """Verify time and group_dest splits exhibit no temporal or account leakage on sample_1000."""
    df = load_and_validate_data(spark, SAMPLE_CSV)

    # Time split
    train_t, val_t, test_t = create_time_split(df)
    if train_t.count() > 0 and test_t.count() > 0:
        max_train_step = train_t.agg(F.max("step")).collect()[0][0]
        min_test_step = test_t.agg(F.min("step")).collect()[0][0]
        assert max_train_step < min_test_step

    # Group destination split
    train_g, val_g, test_g = create_group_dest_split(df)
    train_dests = set(r[0] for r in train_g.select("nameDest").distinct().collect())
    test_dests = set(r[0] for r in test_g.select("nameDest").distinct().collect())

    overlap = train_dests.intersection(test_dests)
    assert len(overlap) == 0, f"Destination leakage detected: {overlap}"
