"""Unit tests for shared feature sets and model configurations."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from feature_sets import (
    ALL_MODEL_NAMES,
    BASE,
    BASE_NOERR,
    BEHAVIOUR_FEATURES,
    COMMUNITY_FEATURES,
    MODEL_FEATURE_SETS,
    STRUCTURAL_FEATURES,
    TRAINABLE_MODELS,
    get_categorical_features,
    get_model_features,
    get_numeric_features,
)
from models import create_model_pipeline, load_model_config


def test_neither_step_nor_isflaggedfraud_in_any_model():
    """Verify strictly that neither 'step' nor 'isFlaggedFraud' appears in any model feature list."""
    forbidden = {"step", "isFlaggedFraud"}
    for model_name, features in MODEL_FEATURE_SETS.items():
        overlap = forbidden.intersection(features)
        assert not overlap, f"Forbidden feature(s) {overlap} found in model '{model_name}'"


def test_all_expected_models_present():
    """Verify the expected comparison ladder models are defined."""
    expected = {"m0_rule", "m1_tabular", "m1_tabular_noerr", "m2_graph", "m3_community_mule"}
    assert set(MODEL_FEATURE_SETS.keys()) == expected
    assert set(ALL_MODEL_NAMES) == expected
    assert set(TRAINABLE_MODELS) == {
        "m1_tabular",
        "m1_tabular_noerr",
        "m2_graph",
        "m3_community_mule",
    }


def test_m0_rule_has_no_features():
    """M0 rule baseline uses domain logic on isFlaggedFraud without ML feature inputs."""
    assert len(get_model_features("m0_rule")) == 0


def test_m1_tabular_noerr_excludes_balance_errors():
    """m1_tabular_noerr must exclude errorBalanceOrig and errorBalanceDest."""
    noerr_feats = get_model_features("m1_tabular_noerr")
    base_feats = get_model_features("m1_tabular")

    assert "errorBalanceOrig" not in noerr_feats
    assert "errorBalanceDest" not in noerr_feats
    assert "errorBalanceOrig" in base_feats
    assert "errorBalanceDest" in base_feats
    assert len(noerr_feats) == len(base_feats) - 2


def test_feature_hierarchy_ladder():
    """Verify progressive superset relationships in the feature ladder."""
    m1 = set(get_model_features("m1_tabular"))
    m2 = set(get_model_features("m2_graph"))
    m3 = set(get_model_features("m3_community_mule"))

    assert m1.issubset(m2), "m2_graph must be a superset of m1_tabular"
    assert m2.issubset(m3), "m3_community_mule must be a superset of m2_graph"

    # Specific feature groups
    struct_set = set(STRUCTURAL_FEATURES)
    comm_set = set(COMMUNITY_FEATURES)
    behav_set = set(BEHAVIOUR_FEATURES)

    assert struct_set.issubset(m2)
    assert comm_set.issubset(m3)
    assert behav_set.issubset(m3)


def test_model_config_loading():
    """Verify model_config.json contains required XGBoost parameters."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "model_config.json"
    assert config_path.exists(), f"Missing config file at {config_path}"

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "xgboost" in data
    xgb_cfg = data["xgboost"]
    assert xgb_cfg["n_estimators"] == 300
    assert xgb_cfg["max_depth"] == 6
    assert xgb_cfg["learning_rate"] == 0.1
    assert xgb_cfg["subsample"] == 0.8
    assert xgb_cfg["colsample_bytree"] == 0.8
    assert xgb_cfg["tree_method"] == "hist"
    assert xgb_cfg["random_state"] == 42


def test_create_model_pipeline_trains_and_predicts():
    """Verify pipeline creation, fitting, and inference on synthetic data."""
    features = get_model_features("m1_tabular")
    pipeline = create_model_pipeline(features, scale_pos_weight=2.0)

    # Generate synthetic training batch
    np.random.seed(42)
    n_samples = 50
    data = {
        "type": np.random.choice(["PAYMENT", "TRANSFER", "CASH_OUT"], size=n_samples),
        "amount": np.random.exponential(1000.0, size=n_samples),
        "oldbalanceOrg": np.random.uniform(0, 10000, size=n_samples),
        "newbalanceOrig": np.random.uniform(0, 10000, size=n_samples),
        "oldbalanceDest": np.random.uniform(0, 10000, size=n_samples),
        "newbalanceDest": np.random.uniform(0, 10000, size=n_samples),
        "errorBalanceOrig": np.random.normal(0, 10, size=n_samples),
        "errorBalanceDest": np.random.normal(0, 10, size=n_samples),
        "is_merchant": np.random.choice([0.0, 1.0], size=n_samples),
    }
    df = pd.DataFrame(data)
    y = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])

    pipeline.fit(df[features], y)
    probs = pipeline.predict_proba(df[features])[:, 1]

    assert len(probs) == n_samples
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
