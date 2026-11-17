"""Unit tests for synthetic planted-ring generation and recovery metrics."""

import json
from pathlib import Path
import pytest

from planted_rings import evaluate_ring_recovery, generate_planted_rings


def test_generate_planted_rings_counts():
    """Verify generate_planted_rings generates 60 rings (20 per pattern)."""
    normal_accounts = [f"CNORM_{i}" for i in range(100)]
    df_txs, ground_truth = generate_planted_rings(normal_accounts, seed=42)

    assert len(ground_truth) == 60
    patterns = [meta["pattern"] for meta in ground_truth.values()]
    assert patterns.count("chain") == 20
    assert patterns.count("funnel") == 20
    assert patterns.count("cycle") == 20
    assert not df_txs.empty


def test_evaluate_ring_recovery_tiny_synthetic_graph():
    """Verify evaluate_ring_recovery on a tiny synthetic graph with known community assignments."""
    ground_truth = {
        "R1_CHAIN": {
            "ring_id": "R1_CHAIN",
            "pattern": "chain",
            "camouflage": 0.0,
            "accounts": ["A1", "A2", "A3", "A4"],
        },
        "R2_CYCLE": {
            "ring_id": "R2_CYCLE",
            "pattern": "cycle",
            "camouflage": 0.0,
            "accounts": ["B1", "B2", "B3"],
        },
    }

    # Perfect recovery: all R1 in comm 1, all R2 in comm 2
    account_to_comm = {
        "A1": 1, "A2": 1, "A3": 1, "A4": 1,
        "B1": 2, "B2": 2, "B3": 2,
    }

    report = evaluate_ring_recovery(ground_truth, account_to_comm)
    assert report["total_planted_rings"] == 2
    assert report["total_recovered_rings"] == 2
    assert report["overall_recovery_rate"] == 1.0


def test_planted_ring_validation_json_counts_and_consistency():
    """Verify outputs/planted_ring_validation.json results satisfy recovered <= planted."""
    path = Path("outputs/planted_ring_validation.json")
    assert path.exists(), "outputs/planted_ring_validation.json missing"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_planted_rings"] == 60
    assert data["total_recovered_rings"] <= 60
    assert "results_by_pattern_and_camouflage" in data

    for group in data["results_by_pattern_and_camouflage"]:
        assert group["recovered"] <= group["planted"]
        assert 0.0 <= group["recovery_rate"] <= 1.0
        assert 0.0 <= group["homogeneity"] <= 1.0
        assert 0.0 <= group["completeness"] <= 1.0
        assert 0.0 <= group["v_measure"] <= 1.0
