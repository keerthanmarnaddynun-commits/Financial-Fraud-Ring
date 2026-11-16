"""Unit tests verifying serialization, loading, and inference of model artefacts."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from feature_sets import (
    MODEL_FEATURE_SETS,
    TRAINABLE_MODELS,
    get_model_features,
)
from models import train_comparative_models


def test_model_artefacts_save_and_predict_fresh_process(tmp_path):
    """Verify that models serialize, reload in a fresh context, and predict on 5 rows."""
    # Generate 50 synthetic training rows
    np.random.seed(42)
    n = 50

    all_features = set()
    for feats in MODEL_FEATURE_SETS.values():
        all_features.update(feats)

    data = {
        "type": np.random.choice(["PAYMENT", "TRANSFER", "CASH_OUT"], size=n),
        "isFraud": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
        "isFlaggedFraud": np.zeros(n, dtype=int),
        "transaction_id": [f"tx_{i}" for i in range(n)],
        "step": np.random.randint(1, 10, size=n),
    }
    for f in all_features:
        if f != "type":
            data[f] = np.random.uniform(0.0, 1000.0, size=n)

    df = pd.DataFrame(data)

    models, _, _ = train_comparative_models(df, output_dir=tmp_path)

    models_dir = tmp_path / "models"
    assert models_dir.exists()

    # Verify all expected model files and feature_names.json exist
    for m_name in TRAINABLE_MODELS:
        model_file = models_dir / f"{m_name}.joblib"
        assert model_file.exists(), f"Missing model file: {model_file}"

    feature_names_file = models_dir / "feature_names.json"
    assert feature_names_file.exists()
    with open(feature_names_file, "r", encoding="utf-8") as f:
        feature_names_map = json.load(f)

    importance_file = tmp_path / "feature_importance.json"
    assert importance_file.exists()
    with open(importance_file, "r", encoding="utf-8") as f:
        importance_map = json.load(f)

    # 5 test rows
    test_5 = df.head(5).copy()

    for m_name in TRAINABLE_MODELS:
        model_path = models_dir / f"{m_name}.joblib"
        loaded_model = joblib.load(model_path)
        features = feature_names_map[m_name]

        probs = loaded_model.predict_proba(test_5[features])[:, 1]
        assert len(probs) == 5
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

        preds = loaded_model.predict(test_5[features])
        assert len(preds) == 5
        assert set(preds).issubset({0, 1})

        # Check feature importance
        assert m_name in importance_map
        importances = importance_map[m_name]
        assert len(importances) > 0
        assert "feature" in importances[0] and "importance" in importances[0]
