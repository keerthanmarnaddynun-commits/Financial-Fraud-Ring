"""
model_training.py
Trains four comparative fraud detection models (Baseline, Model 1: Node2Vec,
Model 2: GraphX Structural, Model 3: Community Detection / Fraud Rings)
using XGBoost or Random Forest with class imbalance handling.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.utils import logger


def prepare_feature_subsets(
    df: pd.DataFrame,
    embedding_dim: int = 16
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, List[str]]]:
    """
    Encodes categorical features and defines exact feature lists for all four models.
    """
    df_feat = df.copy()

    # One-hot encode transaction type
    type_dummies = pd.get_dummies(df_feat["relationship"], prefix="type", drop_first=False)
    for col in type_dummies.columns:
        df_feat[col] = type_dummies[col].astype(int)

    type_cols = list(type_dummies.columns)

    # 1. Baseline Features: Tabular accounting & transaction properties
    baseline_cols = [
        "step", "amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest", "errorBalanceOrig",
        "errorBalanceDest", "isFlaggedFraud"
    ] + type_cols

    # 2. Model 1 Features: Tabular + Node2Vec embeddings
    n2v_cols = (
        [f"src_n2v_{i}" for i in range(embedding_dim)] +
        [f"dst_n2v_{i}" for i in range(embedding_dim)] +
        ["n2v_cosine_sim", "n2v_l2_dist"]
    )
    model1_cols = baseline_cols + n2v_cols

    # 3. Model 2 Features: Tabular + Structural GraphX properties
    graphx_cols = [
        "src_in_degree", "src_out_degree", "src_degree_ratio", "src_pagerank", "src_triangles",
        "dst_in_degree", "dst_out_degree", "dst_degree_ratio", "dst_pagerank", "dst_triangles",
        "is_same_component"
    ]
    model2_cols = baseline_cols + graphx_cols

    # 4. Model 3 Features: Tabular + Community Detection / Ring attributes
    community_cols = [
        "src_community_size", "dst_community_size", "is_same_community",
        "ring_size_min", "ring_size_max"
    ]
    model3_cols = baseline_cols + community_cols

    feature_map = {
        "Baseline": baseline_cols,
        "Model 1 (Node2Vec)": model1_cols,
        "Model 2 (Structural GraphX)": model2_cols,
        "Model 3 (Community / Rings)": model3_cols
    }

    target = df_feat["isFraud"].astype(int)
    return df_feat, target, feature_map


def train_comparative_models(
    df: pd.DataFrame,
    embedding_dim: int = 16,
    model_type: str = "xgboost",
    output_dir: str = "output/models",
    test_size: float = 0.25,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains all four comparative models, logs feature importances, and returns
    predictions and artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)
    df_feat, y, feature_map = prepare_feature_subsets(df, embedding_dim=embedding_dim)

    # Train / Test split with stratification
    train_idx, test_idx = train_test_split(
        df_feat.index,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    # Calculate class imbalance ratio
    neg_count = sum(y_train == 0)
    pos_count = sum(y_train == 1)
    scale_pos = (neg_count / max(1, pos_count))

    logger.info(
        f"Training set: {len(y_train):,} (Fraud: {pos_count:,}) | "
        f"Test set: {len(y_test):,} (Fraud: {sum(y_test==1):,}) | "
        f"Imbalance Scale: {scale_pos:.2f}"
    )

    results = {
        "y_test": y_test,
        "test_df": df_feat.loc[test_idx],
        "models": {},
        "predictions": {},
        "probabilities": {},
        "feature_importances": {},
        "feature_names": {}
    }

    for name, cols in feature_map.items():
        logger.info(f"--- Training {name} with {len(cols)} features ({model_type.upper()}) ---")
        X_train = df_feat.loc[train_idx, cols]
        X_test = df_feat.loc[test_idx, cols]

        if model_type.lower() == "xgboost":
            clf = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=scale_pos,
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1
            )
        else:
            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1
            )

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        # Save model
        clean_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        model_path = os.path.join(output_dir, f"{clean_name}.joblib")
        joblib.dump(clf, model_path)

        # Extract feature importances
        if hasattr(clf, "feature_importances_"):
            importances = dict(zip(cols, clf.feature_importances_))
            top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info(f"Top 5 predictive features for {name}: {top_features}")
            results["feature_importances"][name] = importances

        results["models"][name] = clf
        results["predictions"][name] = y_pred
        results["probabilities"][name] = y_prob
        results["feature_names"][name] = cols

    logger.info("All four models trained and persisted successfully.")
    return results
