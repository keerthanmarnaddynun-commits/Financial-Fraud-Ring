"""Unit tests verifying real-data fraud ring analysis outputs."""

import json
from pathlib import Path
import pandas as pd
import pytest


def test_ring_analysis_json_schema():
    """Verify outputs/ring_analysis.json structure and factual consistency."""
    path = Path("outputs/ring_analysis.json")
    assert path.exists(), "outputs/ring_analysis.json missing"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "number_of_fraud_accounts" in data
    assert "number_of_components" in data
    assert "count_size_ge_3" in data
    assert "finding" in data
    assert data["count_size_ge_3"] == 44
    assert data["number_of_components"] == 8169


def test_rings_summary_csv_and_exported_rings():
    """Verify rings_summary.csv and exported ring json files in outputs/rings/."""
    csv_path = Path("outputs/rings_summary.csv")
    assert csv_path.exists(), "outputs/rings_summary.csv missing"
    df = pd.read_csv(csv_path)

    expected_cols = {"ring_id", "size", "total_amount", "step_min", "step_max", "steps_spanned", "transaction_count"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 8169

    # Verify exported top ring files
    rings_dir = Path("outputs/rings")
    ring_files = list(rings_dir.glob("ring_*.json"))
    assert len(ring_files) == 20

    for rf in ring_files:
        with open(rf, "r", encoding="utf-8") as f:
            rdata = json.load(f)
        assert "ring_id" in rdata
        assert "nodes" in rdata
        assert "edges" in rdata
        assert len(rdata["nodes"]) >= 3
        for node in rdata["nodes"]:
            assert node["role"] in ("origin", "destination", "both")
