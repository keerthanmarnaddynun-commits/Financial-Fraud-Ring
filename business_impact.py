"""Business impact analysis calculating value caught vs missed across threshold grids."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("business_impact")


def compute_impact_for_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    threshold: float,
    fraud_value_total: float,
    total_rows: int,
) -> Dict[str, Any]:
    """Compute business impact metrics at a specific decision threshold."""
    alert_mask = scores >= threshold
    fraud_mask = y_true == 1

    true_alert_mask = alert_mask & fraud_mask
    false_alert_mask = alert_mask & (~fraud_mask)
    missed_fraud_mask = (~alert_mask) & fraud_mask

    alerts = int(np.sum(alert_mask))
    true_alerts = int(np.sum(true_alert_mask))
    false_alerts = int(np.sum(false_alert_mask))

    fraud_value_caught = float(np.sum(amounts[true_alert_mask]))
    fraud_value_missed = float(np.sum(amounts[missed_fraud_mask]))

    alerts_per_10k = float(alerts / total_rows * 10000.0) if total_rows > 0 else 0.0
    share_caught = (
        float(fraud_value_caught / fraud_value_total) if fraud_value_total > 0 else 0.0
    )

    return {
        "threshold": float(threshold),
        "alerts": alerts,
        "true_alerts": true_alerts,
        "false_alerts": false_alerts,
        "alerts_per_10k": round(alerts_per_10k, 2),
        "fraud_value_total": round(fraud_value_total, 2),
        "fraud_value_caught": round(fraud_value_caught, 2),
        "fraud_value_missed": round(fraud_value_missed, 2),
        "share_fraud_value_caught": round(share_caught, 4),
    }


def generate_threshold_grid(scores: np.ndarray, num_points: int = 100) -> np.ndarray:
    """Generate threshold grid based on quantiles plus linspace bounds."""
    percentiles = np.linspace(0.0, 100.0, num_points)
    quant_thresh = np.percentile(scores, percentiles)
    # Add standard linspace points and bounds
    lin_thresh = np.linspace(0.0, 1.0, 21)
    combined = np.unique(np.clip(np.concatenate([quant_thresh, lin_thresh]), 0.0, 1.0))
    combined.sort()
    return combined


def compute_business_impact(
    df: pd.DataFrame,
    models: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculate business impact metrics across model score distributions and rule baseline."""
    if "isFraud" not in df.columns or "amount" not in df.columns:
        raise ValueError("DataFrame must contain 'isFraud' and 'amount' columns")

    y_true = df["isFraud"].to_numpy(dtype=int)
    amounts = df["amount"].to_numpy(dtype=float)
    total_rows = len(df)

    fraud_mask = y_true == 1
    fraud_value_total = float(np.sum(amounts[fraud_mask]))
    fraud_rows = int(np.sum(fraud_mask))

    if models is None:
        models = [
            col.replace("score_", "")
            for col in df.columns
            if col.startswith("score_") and col != "score_m0_rule"
        ]

    impact_results: Dict[str, Any] = {
        "currency": "PaySim simulated currency units",
        "split": "time",
        "test_rows": total_rows,
        "fraud_rows": fraud_rows,
        "fraud_value_total": round(fraud_value_total, 2),
        "models": {},
    }

    # Evaluate rule baseline
    rule_score_col = "score_m0_rule" if "score_m0_rule" in df.columns else None
    if rule_score_col is not None:
        rule_scores = df[rule_score_col].to_numpy(dtype=float)
        rule_metrics = compute_impact_for_threshold(
            y_true=y_true,
            scores=rule_scores,
            amounts=amounts,
            threshold=0.5,
            fraud_value_total=fraud_value_total,
            total_rows=total_rows,
        )
        rule_metrics["rule_name"] = "isFlaggedFraud (transfer > 200,000)"
        impact_results["rule_baseline"] = rule_metrics
    elif "isFlaggedFraud" in df.columns:
        rule_scores = df["isFlaggedFraud"].to_numpy(dtype=float)
        rule_metrics = compute_impact_for_threshold(
            y_true=y_true,
            scores=rule_scores,
            amounts=amounts,
            threshold=0.5,
            fraud_value_total=fraud_value_total,
            total_rows=total_rows,
        )
        rule_metrics["rule_name"] = "isFlaggedFraud (transfer > 200,000)"
        impact_results["rule_baseline"] = rule_metrics

    # Evaluate each ML model
    for m in models:
        score_col = f"score_{m}" if f"score_{m}" in df.columns else m
        if score_col not in df.columns:
            continue

        scores = df[score_col].to_numpy(dtype=float)
        thresholds = generate_threshold_grid(scores, num_points=100)

        grid_points = []
        for th in thresholds:
            point = compute_impact_for_threshold(
                y_true=y_true,
                scores=scores,
                amounts=amounts,
                threshold=th,
                fraud_value_total=fraud_value_total,
                total_rows=total_rows,
            )
            grid_points.append(point)

        impact_results["models"][m] = {
            "num_thresholds": len(grid_points),
            "grid": grid_points,
        }

    return impact_results


def run_business_impact_analysis(
    parquet_path: Path,
    output_dir: Path,
) -> Path:
    """Execute impact analysis reading parquet and writing business_impact.json."""
    parquet_path = Path(parquet_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not parquet_path.exists():
        raise FileNotFoundError(f"Scored time split parquet not found: {parquet_path}")

    LOGGER.info("Loading scored time split from %s", parquet_path)
    df = pd.read_parquet(parquet_path)

    results = compute_business_impact(df)
    out_file = output_dir / "business_impact.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    LOGGER.info("Wrote business impact table to %s", out_file)
    return out_file


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Generate business impact table from scored test set.")
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=Path("outputs/scored_time.parquet"),
        help="Path to scored_time.parquet directory or file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to write business_impact.json.",
    )
    args = parser.parse_args()
    run_business_impact_analysis(args.parquet_path, args.output_dir)


if __name__ == "__main__":
    main()
