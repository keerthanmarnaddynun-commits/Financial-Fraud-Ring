"""Unit tests for dataset splits and validation threshold optimization."""

import numpy as np
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from spark_session import create_spark_session
from splits import (
    create_group_dest_split,
    create_random_split,
    create_time_split,
    draw_training_sample,
    find_best_f1_threshold,
)


@pytest.fixture(scope="module")
def spark():
    session = create_spark_session(app_name="test_splits", master="local[2]")
    yield session
    session.stop()


def test_time_split_max_train_step_less_than_min_test_step(spark):
    """Verify that time split strictly maintains temporal ordering: max(train step) < min(test step)."""
    rows = []
    # 10 steps, each with 10 rows
    for s in range(1, 11):
        for i in range(10):
            rows.append(
                (
                    f"tx_{s}_{i}",
                    s,
                    f"C_orig_{s}_{i}",
                    f"C_dest_{s}_{i}",
                    float(100.0 * s),
                    1 if (s == 10 and i == 0) or (s == 2 and i == 0) else 0,
                    0,
                    "PAYMENT",
                )
            )

    df = spark.createDataFrame(
        rows,
        [
            "transaction_id",
            "step",
            "nameOrig",
            "nameDest",
            "amount",
            "isFraud",
            "isFlaggedFraud",
            "type",
        ],
    )

    train, val, test = create_time_split(df)

    assert train.count() > 0, "Train set should not be empty"
    assert test.count() > 0, "Test set should not be empty"

    max_train_step = train.agg(F.max("step")).collect()[0][0]
    min_test_step = test.agg(F.min("step")).collect()[0][0]

    assert max_train_step < min_test_step, (
        f"Temporal leakage: max(train step) = {max_train_step} is not < min(test step) = {min_test_step}"
    )

    if val.count() > 0:
        max_val_step = val.agg(F.max("step")).collect()[0][0]
        min_val_step = val.agg(F.min("step")).collect()[0][0]
        assert max_train_step < min_val_step or max_train_step == min_val_step
        assert max_val_step < min_test_step


def test_group_dest_split_no_dest_leakage(spark):
    """Verify that group_dest split has completely disjoint nameDest between train and test."""
    rows = []
    # 50 distinct destinations, 2 transactions each
    for d in range(50):
        dest = f"C_dest_{d}"
        for i in range(2):
            rows.append(
                (
                    f"tx_{d}_{i}",
                    1,
                    f"C_orig_{d}_{i}",
                    dest,
                    50.0,
                    1 if d == 0 else 0,
                    0,
                    "TRANSFER",
                )
            )

    df = spark.createDataFrame(
        rows,
        [
            "transaction_id",
            "step",
            "nameOrig",
            "nameDest",
            "amount",
            "isFraud",
            "isFlaggedFraud",
            "type",
        ],
    )

    train, val, test = create_group_dest_split(df)

    train_dests = set(r[0] for r in train.select("nameDest").distinct().collect())
    test_dests = set(r[0] for r in test.select("nameDest").distinct().collect())

    overlap = train_dests.intersection(test_dests)
    assert len(overlap) == 0, f"Account leakage detected: {overlap} appear in both train and test"
    assert len(test_dests) > 0, "Test destinations should not be empty"
    assert len(train_dests) > 0, "Train destinations should not be empty"


def test_random_split_no_tx_leakage(spark):
    """Verify that random split has completely disjoint transaction_id between train and test."""
    rows = [(f"tx_{i}", 1, f"C_orig_{i}", f"C_dest_{i}", 10.0, 0, 0, "PAYMENT") for i in range(100)]
    df = spark.createDataFrame(
        rows,
        [
            "transaction_id",
            "step",
            "nameOrig",
            "nameDest",
            "amount",
            "isFraud",
            "isFlaggedFraud",
            "type",
        ],
    )
    train, val, test = create_random_split(df)

    train_ids = set(r[0] for r in train.select("transaction_id").distinct().collect())
    test_ids = set(r[0] for r in test.select("transaction_id").distinct().collect())

    assert len(train_ids.intersection(test_ids)) == 0


def test_find_best_f1_threshold():
    """Verify that find_best_f1_threshold finds the optimal operating point."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    # Predicted scores: separated around 0.6
    y_scores = np.array([0.1, 0.2, 0.25, 0.3, 0.55, 0.65, 0.7, 0.8, 0.85, 0.9])

    best_thresh, best_f1 = find_best_f1_threshold(y_true, y_scores)
    assert 0.55 < best_thresh <= 0.65
    assert best_f1 == 1.0


def test_draw_training_sample_ratio(spark):
    """Verify that training sample maintains 10:1 negative to positive ratio."""
    rows = []
    # 5 fraud rows, 100 non-fraud rows
    for i in range(5):
        rows.append((f"tx_f_{i}", 1, "C1", "C2", 100.0, 1, 0, "TRANSFER"))
    for i in range(100):
        rows.append((f"tx_n_{i}", 1, "C1", "C2", 50.0, 0, 0, "TRANSFER"))

    df = spark.createDataFrame(
        rows,
        [
            "transaction_id",
            "step",
            "nameOrig",
            "nameDest",
            "amount",
            "isFraud",
            "isFlaggedFraud",
            "type",
        ],
    )
    sample_pdf = draw_training_sample(df, max_rows=1000, seed=42)

    fraud_in_sample = int((sample_pdf["isFraud"] == 1).sum())
    non_fraud_in_sample = int((sample_pdf["isFraud"] == 0).sum())

    assert fraud_in_sample == 5
    assert non_fraud_in_sample == 50  # 10:1 ratio
