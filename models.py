"""Bounded scikit-learn comparisons for transaction and graph feature sets."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, functions as F
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from feature_engineering import GRAPH_FEATURE_COLUMNS

TABULAR_NUMERIC = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
STRUCTURAL_FEATURES = [
    feature
    for feature in GRAPH_FEATURE_COLUMNS
    if "community" not in feature and feature != "same_community"
]
COMMUNITY_FEATURES = [
    "origin_community_size",
    "dest_community_size",
    "same_community",
]


def sample_training_data(
    transactions: DataFrame, max_rows: int, seed: int = 42
) -> pd.DataFrame:
    """Sample a class-aware, bounded dataset before collecting it to the driver."""
    if max_rows < 20:
        raise ValueError("max_rows must be at least 20")
    counts = {
        row["isFraud"]: row["count"]
        for row in transactions.groupBy("isFraud").count().collect()
    }
    positives = int(counts.get(1, 0))
    negatives = int(counts.get(0, 0))
    if positives == 0 or negatives == 0:
        raise ValueError("Training requires at least one fraudulent and one legitimate row")

    positive_budget = min(positives, max(1, max_rows // 4))
    negative_budget = max(1, max_rows - positive_budget)
    fractions = {
        0: min(1.0, negative_budget / negatives),
        1: min(1.0, positive_budget / positives),
    }
    sampled = (
        transactions.sampleBy("isFraud", fractions, seed=seed)
        .orderBy(F.rand(seed))
        .limit(max_rows)
    )
    columns = list(
        dict.fromkeys(
            [
                "transaction_id",
                "isFraud",
                "type",
                *TABULAR_NUMERIC,
                *GRAPH_FEATURE_COLUMNS,
                "origin_community_id",
            ]
        )
    )
    pdf = sampled.select(*columns).toPandas()
    if pdf.empty or pdf["isFraud"].nunique() < 2:
        raise ValueError(
            "The bounded sample did not contain both classes; increase max_rows and retry"
        )
    return pdf


def _create_pipeline(numeric_features: Iterable[str]) -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                list(numeric_features),
            ),
            ("type", OneHotEncoder(handle_unknown="ignore"), ["type"]),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=160,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_comparative_models(
    data: pd.DataFrame,
) -> Tuple[Dict[str, Pipeline], pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Train baseline, graph, and community models on one shared stratified split."""
    labels = data["isFraud"].astype(int)
    train_idx, test_idx = train_test_split(
        np.arange(len(data)),
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )
    train = data.iloc[train_idx]
    test = data.iloc[test_idx].copy()
    feature_sets = {
        "tabular_baseline": TABULAR_NUMERIC,
        "graph_features": [*TABULAR_NUMERIC, *STRUCTURAL_FEATURES],
        "community_ring": [
            *TABULAR_NUMERIC,
            *STRUCTURAL_FEATURES,
            *COMMUNITY_FEATURES,
        ],
    }

    models: Dict[str, Pipeline] = {}
    scores: Dict[str, Dict[str, float]] = {}
    for name, features in feature_sets.items():
        model = _create_pipeline(features)
        model.fit(train[features + ["type"]], train["isFraud"].astype(int))
        probabilities = model.predict_proba(test[features + ["type"]])[:, 1]
        models[name] = model
        test[f"{name}_probability"] = probabilities
        test[f"{name}_prediction"] = (probabilities >= 0.5).astype(int)
        scores[name] = {
            "roc_auc": float(roc_auc_score(test["isFraud"], probabilities))
        }

    test["actual"] = test["isFraud"].astype(int)
    return models, test, scores
