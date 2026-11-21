"""Dataset splitting, training sampling, and validation threshold optimization.

Implements three leakage-free partition schemes on the full enriched DataFrame:
  1. time: train = step <= q75 of step; test = step > q75. Validation = last 15% of train steps.
  2. group_dest: test = abs(hash(nameDest)) % 4 == 0; val = abs(hash(nameDest)) % 4 == 1;
                 train = rest (abs(hash(nameDest)) % 4 in (2, 3)).
  3. random: test = abs(hash(transaction_id)) % 4 == 0; val = abs(hash(transaction_id)) % 4 == 1;
             train = rest (reference only).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, functions as F
from sklearn.metrics import precision_recall_curve

from feature_sets import MODEL_FEATURE_SETS


def create_time_split(df: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """Split DataFrame by temporal step: train, validation, and test.

    train: step <= val_cutoff
    val: step > val_cutoff and step <= q75
    test: step > q75
    Guarantees: max(train step) < min(test step).
    """
    # Collect distinct steps to determine exact cutoffs
    step_rows = df.select("step").distinct().orderBy("step").collect()
    steps = [int(r[0]) for r in step_rows if r[0] is not None]

    if not steps:
        raise ValueError("DataFrame contains no valid step values")

    if len(steps) == 1:
        # Edge case: single step (e.g. tiny test)
        single_step = steps[0]
        train = df.filter(F.col("step") <= single_step)
        val = df.filter(F.lit(False))
        test = df.filter(F.lit(False))
        return train, val, test

    if len(steps) == 2:
        train = df.filter(F.col("step") == steps[0])
        val = df.filter(F.lit(False))
        test = df.filter(F.col("step") == steps[1])
        return train, val, test

    # Normal case: 3 or more distinct steps
    # q75 cutoff for test split
    q75_val = float(np.quantile(steps, 0.75))
    test_steps = [s for s in steps if s > q75_val]
    if not test_steps:
        # Guarantee test has at least the latest step
        q75_val = float(steps[-2])
        test_steps = [steps[-1]]

    train_val_steps = [s for s in steps if s <= q75_val]
    if len(train_val_steps) >= 2:
        val_cutoff = float(np.quantile(train_val_steps, 0.85))
        train_steps = [s for s in train_val_steps if s <= val_cutoff]
        if not train_steps or set(train_steps) == set(train_val_steps):
            val_cutoff = float(train_val_steps[-2])
    else:
        val_cutoff = float(train_val_steps[0])

    train = df.filter(F.col("step") <= val_cutoff)
    val = df.filter((F.col("step") > val_cutoff) & (F.col("step") <= q75_val))
    test = df.filter(F.col("step") > q75_val)

    return train, val, test


def create_group_dest_split(df: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """Split DataFrame by destination account hash to prevent destination leakage.

    test: abs(hash(nameDest)) % 4 == 0
    val: abs(hash(nameDest)) % 4 == 1
    train: abs(hash(nameDest)) % 4 in (2, 3)
    Guarantees: disjoint nameDest sets between train and test.
    """
    dest_hash = F.abs(F.hash(F.col("nameDest")).cast("bigint")) % 4
    test = df.filter(dest_hash == 0)
    val = df.filter(dest_hash == 1)
    train = df.filter((dest_hash != 0) & (dest_hash != 1))
    return train, val, test


def create_random_split(df: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """Split DataFrame deterministically by transaction_id hash for baseline reference."""
    tx_hash = F.abs(F.hash(F.col("transaction_id")).cast("bigint")) % 4
    test = df.filter(tx_hash == 0)
    val = df.filter(tx_hash == 1)
    train = df.filter((tx_hash != 0) & (tx_hash != 1))
    return train, val, test


def get_dataset_split(
    df: DataFrame, split_type: str
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """Dispatcher for the supported split types: 'time', 'group_dest', 'random'."""
    if split_type == "time":
        return create_time_split(df)
    elif split_type == "group_dest":
        return create_group_dest_split(df)
    elif split_type == "random":
        return create_random_split(df)
    else:
        raise ValueError(
            f"Unknown split_type: {split_type}. Supported: 'time', 'group_dest', 'random'"
        )


def draw_training_sample(
    train_df: DataFrame,
    max_rows: int = 300_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample class-aware bounded training data strictly from train_df.

    Includes all fraud rows plus negatives at a 10:1 ratio, capped at max_rows,
    collected to a pandas DataFrame on the driver.
    """
    fraud_df = train_df.filter(F.col("isFraud") == 1)
    fraud_count = fraud_df.count()
    if fraud_count == 0:
        return train_df.limit(max_rows).toPandas()

    desired_negatives = fraud_count * 10
    total_desired = fraud_count + desired_negatives

    non_fraud_df = train_df.filter(F.col("isFraud") == 0)
    non_fraud_count = non_fraud_df.count()

    if total_desired > max_rows:
        max_fraud = max(1, max_rows // 11)
        max_neg = max_rows - max_fraud
        sampled_fraud = fraud_df.limit(max_fraud)
        if non_fraud_count <= 2 * max_neg:
            sampled_non_fraud = non_fraud_df.orderBy(F.rand(seed)).limit(max_neg)
        else:
            frac = min(1.0, (max_neg * 1.5) / max(1, non_fraud_count))
            sampled_non_fraud = (
                non_fraud_df.sample(withReplacement=False, fraction=frac, seed=seed)
                .limit(max_neg)
            )
        combined = sampled_fraud.unionByName(sampled_non_fraud)
    else:
        if non_fraud_count <= 2 * desired_negatives:
            sampled_non_fraud = non_fraud_df.orderBy(F.rand(seed)).limit(desired_negatives)
        else:
            frac = min(1.0, (desired_negatives * 1.5) / max(1, non_fraud_count))
            sampled_non_fraud = (
                non_fraud_df.sample(withReplacement=False, fraction=frac, seed=seed)
                .limit(desired_negatives)
            )
        combined = fraud_df.unionByName(sampled_non_fraud)

    all_features = set()
    for feats in MODEL_FEATURE_SETS.values():
        all_features.update(feats)

    cols_to_select = [
        c
        for c in [
            "transaction_id",
            "step",
            "isFraud",
            "isFlaggedFraud",
            "nameOrig",
            "nameDest",
            *all_features,
        ]
        if c in combined.columns
    ]
    return combined.select(*cols_to_select).toPandas()


def find_best_f1_threshold(
    y_true: np.ndarray, y_scores: np.ndarray
) -> Tuple[float, float]:
    """Find the threshold in [0, 1] maximizing F1 score at true prevalence.

    Returns:
        (best_threshold, best_f1)
    """
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5, 0.0

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1_denom = precisions[:-1] + recalls[:-1]
    f1_scores = np.where(
        f1_denom > 0,
        2 * (precisions[:-1] * recalls[:-1]) / f1_denom,
        0.0,
    )
    if len(f1_scores) == 0:
        return 0.5, 0.0

    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    return best_threshold, best_f1
