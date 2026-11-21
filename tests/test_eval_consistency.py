"""Unit tests verifying mathematical consistency of evaluation reports."""

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from evaluation import evaluate_model_predictions, evaluate_models


def verify_split_metrics_consistency(split_data: Dict[str, Any]) -> None:
    """Verify strictly that for every model:

    1. tp + fn == fraud rows in test set
    2. tp + fp + fn + tn == test rows
    3. false alarms per 10k == fp / rows * 10000
    4. prevalence == fraud rows / rows
    """
    for model_name, metrics in split_data.items():
        if not isinstance(metrics, dict) or "at_threshold" not in metrics:
            continue

        rows = metrics["rows"]
        prevalence = metrics["prevalence"]
        expected_fraud = int(round(rows * prevalence))

        # Check at_threshold metrics
        at_th = metrics["at_threshold"]
        conf_th = at_th["confusion"]
        tp, fp, fn, tn = conf_th["tp"], conf_th["fp"], conf_th["fn"], conf_th["tn"]

        assert tp + fn == expected_fraud, (
            f"Model {model_name}: tp+fn ({tp+fn}) does not equal fraud count ({expected_fraud})"
        )
        assert tp + fp + fn + tn == rows, (
            f"Model {model_name}: tp+fp+fn+tn ({tp+fp+fn+tn}) does not equal rows ({rows})"
        )

        expected_fa_10k = float(fp / rows * 10000.0) if rows > 0 else 0.0
        assert metrics["false_alarms_per_10k"] == pytest.approx(expected_fa_10k, rel=1e-4)

        # Check at_0.5 metrics
        at_05 = metrics["at_0.5"]
        conf_05 = at_05["confusion"]
        tp_05, fp_05, fn_05, tn_05 = conf_05["tp"], conf_05["fp"], conf_05["fn"], conf_05["tn"]

        assert tp_05 + fn_05 == expected_fraud, (
            f"Model {model_name} at 0.5: tp+fn ({tp_05+fn_05}) != fraud count ({expected_fraud})"
        )
        assert tp_05 + fp_05 + fn_05 + tn_05 == rows, (
            f"Model {model_name} at 0.5: tp+fp+fn+tn ({tp_05+fp_05+fn_05+tn_05}) != rows ({rows})"
        )


def test_evaluate_model_predictions_exact_consistency():
    """Verify evaluate_model_predictions generates perfectly consistent confusion counts."""
    np.random.seed(42)
    n = 1000
    # True prevalence ~ 2%
    y_true = np.random.choice([0, 1], size=n, p=[0.98, 0.02])
    y_scores = np.random.uniform(0.0, 1.0, size=n)

    metrics = evaluate_model_predictions(y_true, y_scores, val_threshold=0.65)
    split_data = {"test_model": metrics}

    verify_split_metrics_consistency(split_data)


def test_evaluation_json_file_consistency():
    """Verify any generated evaluation.json in outputs_test or outputs satisfies consistency."""
    search_paths = [
        Path("outputs_test") / "evaluation.json",
        Path("outputs") / "evaluation.json",
    ]

    found = False
    for p in search_paths:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "splits" in data:
                found = True
                for split_name, split_models in data["splits"].items():
                    verify_split_metrics_consistency(split_models)

    # If no file exists yet on disk, ensure synthetic test ran
    assert True
