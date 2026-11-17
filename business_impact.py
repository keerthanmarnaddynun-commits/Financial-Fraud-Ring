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

DEFAULT_REVIEW_COST = 10000.0
DEFAULT_REVIEW_COST_LABEL = "assumption"
SENSITIVITY_COSTS = [1000.0, 10000.0, 100000.0]


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


def load_validation_thresholds(evaluation_path: Optional[Path]) -> Dict[str, float]:
    """Load validation thresholds from evaluation.json for the time split."""
    thresholds: Dict[str, float] = {}
    if evaluation_path is not None and Path(evaluation_path).exists():
        try:
            with open(evaluation_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            time_split = data.get("splits", {}).get("time", {})
            for m_key, m_val in time_split.items():
                if isinstance(m_val, dict) and "threshold" in m_val:
                    thresholds[m_key] = float(m_val["threshold"])
        except Exception as exc:
            LOGGER.warning("Could not read validation thresholds from %s: %s", evaluation_path, exc)
    return thresholds


def compute_business_impact(
    df: pd.DataFrame,
    models: Optional[List[str]] = None,
    evaluation_path: Optional[Path] = None,
    default_review_cost: float = DEFAULT_REVIEW_COST,
    sensitivity_costs: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Calculate business impact metrics across model score distributions and rule baseline."""
    if "isFraud" not in df.columns or "amount" not in df.columns:
        raise ValueError("DataFrame must contain 'isFraud' and 'amount' columns")

    if sensitivity_costs is None:
        sensitivity_costs = list(SENSITIVITY_COSTS)

    y_true = df["isFraud"].to_numpy(dtype=int)
    amounts = df["amount"].to_numpy(dtype=float)
    total_rows = len(df)

    fraud_mask = y_true == 1
    fraud_value_total = float(np.sum(amounts[fraud_mask]))
    fraud_rows = int(np.sum(fraud_mask))

    val_thresholds = load_validation_thresholds(evaluation_path)

    if models is None:
        models = [
            col.replace("score_", "")
            for col in df.columns
            if col.startswith("score_") and col != "score_m0_rule"
        ]

    impact_results: Dict[str, Any] = {
        "currency": "PaySim simulated currency units",
        "split": "time",
        "default_review_cost": default_review_cost,
        "review_cost_label": DEFAULT_REVIEW_COST_LABEL,
        "sensitivity_review_costs": sensitivity_costs,
        "test_rows": total_rows,
        "fraud_rows": fraud_rows,
        "fraud_value_total": round(fraud_value_total, 2),
        "headline_operating_table": [],
        "sensitivity_table": [],
        "models": {},
    }

    # 1. Rule baseline evaluation
    rule_metrics: Optional[Dict[str, Any]] = None
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
        rule_metrics["threshold_source"] = "rule_definition"
        rule_metrics["net_financial_saving_default"] = round(
            rule_metrics["fraud_value_caught"] - default_review_cost * rule_metrics["alerts"], 2
        )
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
        rule_metrics["threshold_source"] = "rule_definition"
        rule_metrics["net_financial_saving_default"] = round(
            rule_metrics["fraud_value_caught"] - default_review_cost * rule_metrics["alerts"], 2
        )
        impact_results["rule_baseline"] = rule_metrics

    if rule_metrics is not None:
        impact_results["headline_operating_table"].append({
            "model_key": "rule_baseline",
            "display_name": "isFlaggedFraud (Amount > 200k)",
            "operating_threshold": 0.5,
            "threshold_source": "rule_definition",
            "alerts": rule_metrics["alerts"],
            "true_alerts": rule_metrics["true_alerts"],
            "false_alerts": rule_metrics["false_alerts"],
            "alerts_per_10k": rule_metrics["alerts_per_10k"],
            "fraud_value_caught": rule_metrics["fraud_value_caught"],
            "fraud_value_missed": rule_metrics["fraud_value_missed"],
            "share_fraud_value_caught": rule_metrics["share_fraud_value_caught"],
            "net_financial_saving": rule_metrics["net_financial_saving_default"],
        })

    # 2. Machine learning models evaluation
    for m in models:
        score_col = f"score_{m}" if f"score_{m}" in df.columns else m
        if score_col not in df.columns:
            continue

        scores = df[score_col].to_numpy(dtype=float)
        thresholds = generate_threshold_grid(scores, num_points=100)

        # Headline operating point: MUST be validation threshold from evaluation.json
        val_th = val_thresholds.get(m, 0.5)
        headline_pt = compute_impact_for_threshold(
            y_true=y_true,
            scores=scores,
            amounts=amounts,
            threshold=val_th,
            fraud_value_total=fraud_value_total,
            total_rows=total_rows,
        )
        headline_net = round(
            headline_pt["fraud_value_caught"] - default_review_cost * headline_pt["alerts"], 2
        )
        headline_pt["net_financial_saving_default"] = headline_net
        headline_pt["threshold_source"] = "validation"

        impact_results["headline_operating_table"].append({
            "model_key": m,
            "display_name": m,
            "operating_threshold": round(val_th, 6),
            "threshold_source": "validation",
            "alerts": headline_pt["alerts"],
            "true_alerts": headline_pt["true_alerts"],
            "false_alerts": headline_pt["false_alerts"],
            "alerts_per_10k": headline_pt["alerts_per_10k"],
            "fraud_value_caught": headline_pt["fraud_value_caught"],
            "fraud_value_missed": headline_pt["fraud_value_missed"],
            "share_fraud_value_caught": headline_pt["share_fraud_value_caught"],
            "net_financial_saving": headline_net,
        })

        # Grid points across threshold spectrum
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
            "validation_threshold": round(val_th, 6),
            "headline_operating_point": headline_pt,
            "num_thresholds": len(grid_points),
            "grid": grid_points,
        }

    # 3. Sensitivity table at review costs [1000, 10000, 100000]
    for cost in sensitivity_costs:
        # Rule baseline
        if rule_metrics is not None:
            rule_alerts = rule_metrics["alerts"]
            rule_caught = rule_metrics["fraud_value_caught"]
            rule_net = round(rule_caught - cost * rule_alerts, 2)
            impact_results["sensitivity_table"].append({
                "review_cost": cost,
                "model_key": "rule_baseline",
                "mode": "fixed_rule",
                "threshold": 0.5,
                "alerts": rule_alerts,
                "fraud_value_caught": rule_caught,
                "net_financial_saving": rule_net,
            })

        for m in models:
            if m not in impact_results["models"]:
                continue
            m_info = impact_results["models"][m]
            val_pt = m_info["headline_operating_point"]
            val_th = float(val_pt["threshold"])
            val_alerts = int(val_pt["alerts"])
            val_caught = float(val_pt["fraud_value_caught"])
            val_net = round(val_caught - cost * val_alerts, 2)

            impact_results["sensitivity_table"].append({
                "review_cost": cost,
                "model_key": m,
                "mode": "validation_threshold",
                "threshold": round(val_th, 6),
                "alerts": val_alerts,
                "fraud_value_caught": val_caught,
                "net_financial_saving": val_net,
            })

            # Hindsight best on test grid for this cost
            grid = m_info["grid"]
            best_point = max(
                grid,
                key=lambda p: float(p.get("fraud_value_caught", 0.0)) - cost * float(p.get("alerts", 0)),
            )
            best_th = float(best_point.get("threshold", 0.5))
            best_alerts = int(best_point.get("alerts", 0))
            best_caught = float(best_point.get("fraud_value_caught", 0.0))
            best_net = round(best_caught - cost * best_alerts, 2)

            impact_results["sensitivity_table"].append({
                "review_cost": cost,
                "model_key": m,
                "mode": "hindsight_best_test",
                "threshold": round(best_th, 6),
                "alerts": best_alerts,
                "fraud_value_caught": best_caught,
                "net_financial_saving": best_net,
            })

    return impact_results


def run_business_impact_analysis(
    parquet_path: Path,
    output_dir: Path,
    evaluation_path: Optional[Path] = None,
    default_review_cost: float = DEFAULT_REVIEW_COST,
) -> Path:
    """Execute impact analysis reading parquet and writing business_impact.json."""
    parquet_path = Path(parquet_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if evaluation_path is None:
        candidate_eval = output_dir / "evaluation.json"
        if candidate_eval.exists():
            evaluation_path = candidate_eval

    if not parquet_path.exists():
        raise FileNotFoundError(f"Scored time split parquet not found: {parquet_path}")

    LOGGER.info("Loading scored time split from %s", parquet_path)
    df = pd.read_parquet(parquet_path)

    results = compute_business_impact(
        df,
        evaluation_path=evaluation_path,
        default_review_cost=default_review_cost,
    )
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
    parser.add_argument(
        "--evaluation-path",
        type=Path,
        default=None,
        help="Path to evaluation.json to load validation thresholds.",
    )
    parser.add_argument(
        "--default-review-cost",
        type=float,
        default=DEFAULT_REVIEW_COST,
        help="Default manual review cost per alert in simulated currency units (default: 10,000.0).",
    )
    args = parser.parse_args()
    run_business_impact_analysis(
        args.parquet_path,
        args.output_dir,
        args.evaluation_path,
        args.default_review_cost,
    )


if __name__ == "__main__":
    main()
