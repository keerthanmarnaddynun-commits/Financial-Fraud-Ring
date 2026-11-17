"""Unit tests for business impact metrics calculation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from business_impact import compute_business_impact, compute_impact_for_threshold


def test_compute_impact_for_threshold_exact_equality():
    """Verify that caught + missed == total exactly for synthetic transactions."""
    y_true = np.array([1, 0, 1, 0, 1])
    scores = np.array([0.9, 0.8, 0.4, 0.1, 0.95])
    amounts = np.array([500.0, 120.0, 300.0, 50.0, 200.0])
    total_val = 1000.0

    res = compute_impact_for_threshold(
        y_true, scores, amounts, threshold=0.5, fraud_value_total=total_val, total_rows=5
    )
    assert res["fraud_value_caught"] + res["fraud_value_missed"] == pytest.approx(total_val)
    assert res["true_alerts"] == 2
    assert res["false_alerts"] == 1
    assert res["alerts"] == 3


def test_business_impact_json_exact_conservation():
    """Verify that for every model and threshold in outputs/business_impact.json, caught + missed == total."""
    path = Path("outputs/business_impact.json")
    assert path.exists(), "outputs/business_impact.json not found"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_val = data["fraud_value_total"]
    assert total_val > 0.0

    # Check rule baseline
    if "rule_baseline" in data:
        rb = data["rule_baseline"]
        sum_val = rb["fraud_value_caught"] + rb["fraud_value_missed"]
        assert sum_val == pytest.approx(total_val, rel=1e-4), (
            f"Rule baseline sum {sum_val} != total {total_val}"
        )

    # Check all model grid points
    for m_name, m_info in data["models"].items():
        grid = m_info["grid"]
        assert len(grid) > 0
        for pt in grid:
            sum_val = pt["fraud_value_caught"] + pt["fraud_value_missed"]
            assert sum_val == pytest.approx(total_val, rel=1e-4), (
                f"Model {m_name} at threshold {pt['threshold']}: {sum_val} != {total_val}"
            )
