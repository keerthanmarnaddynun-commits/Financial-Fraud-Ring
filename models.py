"""Comparative model definitions and training for fraud analytics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, functions as F
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from feature_sets import (
    BASE,
    BASE_NOERR,
    COMMUNITY_FEATURES,
    MODEL_FEATURE_SETS,
    STRUCTURAL_FEATURES,
    TRAINABLE_MODELS,
    get_model_features,
)
from splits import (
    create_group_dest_split,
    create_random_split,
    create_time_split,
    draw_training_sample,
    find_best_f1_threshold,
    get_dataset_split,
)

VALID_TYPES: List[str] = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def load_model_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load XGBoost hyperparameters from configs/model_config.json."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "configs" / "model_config.json"
    if not Path(config_path).exists():
        return {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": 42,
        }
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return dict(data.get("xgboost", data))


def create_model_pipeline(
    feature_names: List[str],
    model_config: Optional[Dict[str, Any]] = None,
    scale_pos_weight: float = 1.0,
) -> Pipeline:
    """Create a scikit-learn Pipeline with median imputation, one-hot encoding, and XGBoost."""
    if model_config is None:
        model_config = load_model_config()

    numeric_cols = [f for f in feature_names if f != "type"]
    has_type = "type" in feature_names

    transformers = []
    if numeric_cols:
        transformers.append(
            ("numeric", SimpleImputer(strategy="median"), numeric_cols)
        )
    if has_type:
        transformers.append(
            (
                "type",
                OneHotEncoder(categories=[VALID_TYPES], handle_unknown="ignore"),
                ["type"],
            )
        )

    preprocessing = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    xgb_params = dict(model_config)
    xgb_params["scale_pos_weight"] = scale_pos_weight
    xgb_params.setdefault("eval_metric", "logloss")

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", XGBClassifier(**xgb_params)),
        ]
    )


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

    all_features = set()
    for feats in MODEL_FEATURE_SETS.values():
        all_features.update(feats)

    columns = list(
        dict.fromkeys(
            [
                "transaction_id",
                "step",
                "isFraud",
                "isFlaggedFraud",
                "nameOrig",
                "nameDest",
                *all_features,
            ]
        )
    )
    available = set(transactions.columns)
    selected = [c for c in columns if c in available]
    pdf = sampled.select(*selected).toPandas()
    if pdf.empty or pdf["isFraud"].nunique() < 2:
        raise ValueError(
            "The bounded sample did not contain both classes; increase max_rows and retry"
        )
    return pdf


def train_comparative_models(
    data: pd.DataFrame,
    output_dir: Optional[Path] = None,
    split_name: Optional[str] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Train comparative models across the feature ladder."""
    labels = data["isFraud"].astype(int)
    train_idx, test_idx = train_test_split(
        np.arange(len(data)),
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )
    train = data.iloc[train_idx].copy()
    test = data.iloc[test_idx].copy()

    models: Dict[str, Any] = {}
    scores: Dict[str, Dict[str, float]] = {}

    # M0: Rule-based baseline (uses isFlaggedFraud, never trained)
    if "isFlaggedFraud" in test.columns:
        test["m0_rule_prediction"] = test["isFlaggedFraud"].astype(int)
        test["m0_rule_probability"] = test["isFlaggedFraud"].astype(float)
        scores["m0_rule"] = {
            "roc_auc": float(roc_auc_score(test["isFraud"], test["m0_rule_probability"]))
            if test["isFraud"].nunique() > 1
            else 0.5
        }

    # Trainable models: m1_tabular, m1_tabular_noerr, m2_graph, m3_community_mule
    positives = int(train["isFraud"].sum())
    negatives = int(len(train) - positives)
    scale_pos = negatives / max(1, positives)

    for name in TRAINABLE_MODELS:
        features = get_model_features(name)
        # Verify required columns exist
        for col in features:
            if col not in train.columns:
                train[col] = 0.0
            if col not in test.columns:
                test[col] = 0.0

        model = create_model_pipeline(features, scale_pos_weight=scale_pos)
        model.fit(train[features], train["isFraud"].astype(int))
        probabilities = model.predict_proba(test[features])[:, 1]
        models[name] = model
        test[f"{name}_probability"] = probabilities
        test[f"{name}_prediction"] = (probabilities >= 0.5).astype(int)
        scores[name] = {
            "roc_auc": float(roc_auc_score(test["isFraud"], probabilities))
            if test["isFraud"].nunique() > 1
            else 0.5
        }

    # Backward compatibility aliases dropped per STEP 1.15 specification

    if output_dir is not None:
        target_dir = Path(output_dir) / "models"
        if split_name:
            target_dir = target_dir / split_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for m_name in TRAINABLE_MODELS:
            joblib.dump(models[m_name], target_dir / f"{m_name}.joblib")

        # Save feature_names.json
        feature_names_dict = {
            m_name: get_model_features(m_name) for m_name in TRAINABLE_MODELS
        }
        with open(target_dir / "feature_names.json", "w", encoding="utf-8") as f:
            json.dump(feature_names_dict, f, indent=2)

        # Compute and write feature_importance.json (XGBoost gain, top 20 per model)
        importance_dict: Dict[str, Any] = {}
        for m_name in TRAINABLE_MODELS:
            pipe = models[m_name]
            try:
                prep = pipe.named_steps["preprocessing"]
                clf = pipe.named_steps["classifier"]
                raw_names = prep.get_feature_names_out()
                clean_names = [
                    f.replace("numeric__", "").replace("type__", "") for f in raw_names
                ]
                importances = clf.feature_importances_
                paired = sorted(
                    zip(clean_names, importances),
                    key=lambda x: x[1],
                    reverse=True,
                )
                top20 = [
                    {"feature": feat, "importance": float(imp)}
                    for feat, imp in paired[:20]
                ]
                importance_dict[m_name] = top20
            except Exception:
                importance_dict[m_name] = []

        with open(target_dir / "feature_importance.json", "w", encoding="utf-8") as f:
            json.dump(importance_dict, f, indent=2)

        # Write root models directory for backward compatibility if time split
        if split_name in ("time", None):
            root_dir = Path(output_dir) / "models"
            root_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(models["m1_tabular"], root_dir / "baseline.joblib")
            for m_name in TRAINABLE_MODELS:
                joblib.dump(models[m_name], root_dir / f"{m_name}.joblib")
            with open(root_dir / "feature_names.json", "w", encoding="utf-8") as f:
                json.dump(feature_names_dict, f, indent=2)
            with open(root_dir / "feature_importance.json", "w", encoding="utf-8") as f:
                json.dump(importance_dict, f, indent=2)

    test["actual"] = test["isFraud"].astype(int)
    return models, test, scores
