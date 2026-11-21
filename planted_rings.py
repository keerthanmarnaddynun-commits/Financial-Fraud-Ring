"""Planted-ring validation using synthetic injection with camouflage on a 200k PaySim sample."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import psutil
from pyspark.sql import functions as F
from sklearn.metrics import completeness_score, homogeneity_score, v_measure_score

from data_ingestion import PAYMENTS_SCHEMA, ensure_dataset
from feature_engineering import compute_connected_components, compute_label_propagation
from spark_session import create_spark_session

LOGGER = logging.getLogger("planted_rings")


def check_ram_preflight(min_gb_normal: float = 3.0, min_gb_fallback: float = 1.5) -> Dict[str, Any]:
    """Check system available memory and return appropriate driver settings."""
    avail_bytes = psutil.virtual_memory().available
    avail_gb = avail_bytes / (1024**3)
    LOGGER.info("Pre-flight check: Available RAM = %.2f GB", avail_gb)

    driver_mem = os.getenv("SPARK_DRIVER_MEMORY", "3g")
    shuffle_parts = os.getenv("SPARK_SHUFFLE_PARTITIONS", "8")

    if avail_gb < min_gb_fallback:
        raise MemoryError(
            f"Available RAM ({avail_gb:.2f} GB) is below critical threshold ({min_gb_fallback:.2f} GB). Aborting."
        )

    if avail_gb < min_gb_normal:
        LOGGER.warning(
            "Available RAM (%.2f GB) is below %.1f GB; setting low-memory fallback (2g driver, 8 partitions).",
            avail_gb,
            min_gb_normal,
        )
        return {"driver_memory": "2g", "shuffle_partitions": "8", "low_memory": True}

    return {"driver_memory": driver_mem, "shuffle_partitions": shuffle_parts, "low_memory": False}


def generate_planted_rings(
    normal_accounts: List[str],
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Generate 60 planted rings (20 chains, 20 funnels, 20 cycles) across 3 camouflage levels.

    Returns:
        (planted_transactions_df, ground_truth_dict)
    """
    rng = np.random.default_rng(seed)
    patterns = ["chain", "funnel", "cycle"]
    camo_levels = [0.0, 0.20, 0.50]  # 0%, 20%, 50%

    transactions: List[Dict[str, Any]] = []
    ground_truth: Dict[str, Dict[str, Any]] = {}

    ring_idx = 0
    for pattern in patterns:
        for i in range(20):
            ring_idx += 1
            ring_id = f"CPLANT_R{ring_idx:03d}_{pattern.upper()}"
            # Assign camouflage level: 7 at 0%, 7 at 20%, 6 at 50%
            if i < 7:
                camo = 0.0
            elif i < 14:
                camo = 0.20
            else:
                camo = 0.50

            ring_nodes: List[str] = []
            base_amount = rng.uniform(50_000, 300_000)

            if pattern == "chain":
                # 4-6 accounts, 5% fee per hop, final cash-out
                num_hops = rng.integers(4, 7)
                chain_accounts = [f"{ring_id}_N{k}" for k in range(num_hops)]
                ring_nodes.extend(chain_accounts)
                curr_amt = base_amount

                for h in range(num_hops - 1):
                    src = chain_accounts[h]
                    dst = chain_accounts[h + 1]
                    tx_type = "TRANSFER" if h < num_hops - 2 else "CASH_OUT"
                    transactions.append({
                        "nameOrig": src,
                        "nameDest": dst,
                        "amount": round(curr_amt, 2),
                        "type": tx_type,
                        "step": 100 + h * 2,
                        "isFraud": 1,
                    })
                    curr_amt *= 0.95  # 5% fee deduction per hop

            elif pattern == "funnel":
                # 8-15 feeders into one hub, then cash-out
                num_feeders = rng.integers(8, 16)
                hub = f"{ring_id}_HUB"
                feeder_accounts = [f"{ring_id}_F{k}" for k in range(num_feeders)]
                cashout_dest = f"{ring_id}_CASHOUT"
                ring_nodes.extend(feeder_accounts + [hub, cashout_dest])

                feeder_amt = base_amount / num_feeders
                for f_acc in feeder_accounts:
                    transactions.append({
                        "nameOrig": f_acc,
                        "nameDest": hub,
                        "amount": round(feeder_amt, 2),
                        "type": "TRANSFER",
                        "step": 150,
                        "isFraud": 1,
                    })
                # Final cash-out from hub
                transactions.append({
                    "nameOrig": hub,
                    "nameDest": cashout_dest,
                    "amount": round(base_amount * 0.98, 2),
                    "type": "CASH_OUT",
                    "step": 155,
                    "isFraud": 1,
                })

            elif pattern == "cycle":
                # 3-5 accounts in a directed loop
                cycle_len = rng.integers(3, 6)
                cycle_accounts = [f"{ring_id}_C{k}" for k in range(cycle_len)]
                ring_nodes.extend(cycle_accounts)

                for c_idx in range(cycle_len):
                    src = cycle_accounts[c_idx]
                    dst = cycle_accounts[(c_idx + 1) % cycle_len]
                    transactions.append({
                        "nameOrig": src,
                        "nameDest": dst,
                        "amount": round(base_amount, 2),
                        "type": "TRANSFER",
                        "step": 200 + c_idx,
                        "isFraud": 1,
                    })

            # Add Camouflage transactions if applicable
            if camo > 0 and normal_accounts:
                num_camo_nodes = int(np.ceil(len(ring_nodes) * camo))
                camo_targets = rng.choice(ring_nodes, size=num_camo_nodes, replace=False)
                for c_node in camo_targets:
                    num_txs = rng.integers(1, 4)
                    for t in range(num_txs):
                        partner = rng.choice(normal_accounts)
                        direction = rng.choice(["out", "in"])
                        src = c_node if direction == "out" else partner
                        dst = partner if direction == "out" else c_node
                        transactions.append({
                            "nameOrig": src,
                            "nameDest": dst,
                            "amount": round(rng.uniform(100, 5000), 2),
                            "type": "PAYMENT",
                            "step": rng.integers(50, 300),
                            "isFraud": 0,
                        })

            ground_truth[ring_id] = {
                "ring_id": ring_id,
                "pattern": pattern,
                "camouflage": camo,
                "accounts": sorted(list(set(ring_nodes))),
            }

    df_planted = pd.DataFrame(transactions)
    return df_planted, ground_truth


def evaluate_ring_recovery(
    ground_truth: Dict[str, Dict[str, Any]],
    account_to_community: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate ring recovery, ring contained rate, ring purity, and clustering metrics."""
    # Build reverse community to accounts mapping
    community_to_accounts: Dict[Any, Set[str]] = {}
    for acc, comm in account_to_community.items():
        community_to_accounts.setdefault(comm, set()).add(acc)

    pattern_camo_groups: Dict[Tuple[str, float], List[str]] = {}
    for r_id, meta in ground_truth.items():
        key = (meta["pattern"], meta["camouflage"])
        pattern_camo_groups.setdefault(key, []).append(r_id)

    total_planted = len(ground_truth)
    total_recovered = 0

    detailed_eval: List[Dict[str, Any]] = []

    for (pattern, camo), ring_ids in sorted(pattern_camo_groups.items()):
        recovered_in_group = 0
        y_true_planted: List[str] = []
        y_pred_comm: List[str] = []
        contaminations: List[float] = []
        contained_rates: List[float] = []
        purities: List[float] = []
        comms_with_planted_accounts: Set[Any] = set()

        for r_id in ring_ids:
            ring_nodes = set(ground_truth[r_id]["accounts"])
            comm_counts: Dict[Any, int] = {}
            for node in ring_nodes:
                c = account_to_community.get(node)
                if c is not None:
                    comm_counts[c] = comm_counts.get(c, 0) + 1
                    y_true_planted.append(r_id)
                    y_pred_comm.append(str(c))
                    comms_with_planted_accounts.add(c)

            # Find dominant component for this ring
            best_comm = None
            best_overlap = 0
            for comm, overlap in comm_counts.items():
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_comm = comm

            if best_comm is not None:
                comm_nodes = community_to_accounts.get(best_comm, set())
                comm_size = len(comm_nodes)
                ring_contained_frac = best_overlap / max(1, len(ring_nodes))
                purity = best_overlap / max(1, comm_size)

                contained_rates.append(ring_contained_frac)
                purities.append(purity)

                normal_in_comm = sum(1 for n in comm_nodes if not str(n).startswith("CPLANT"))
                contam = normal_in_comm / max(1, comm_size)
                contaminations.append(contam)

                # Strict recovery criterion: at least 80% of ring nodes in community AND 80% of community is ring
                if ring_contained_frac >= 0.80 and purity >= 0.80:
                    recovered_in_group += 1
                    total_recovered += 1
            else:
                contained_rates.append(0.0)
                purities.append(0.0)
                contaminations.append(0.0)

        # Largest component size holding any planted accounts in this group
        largest_comp_size = max(
            (len(community_to_accounts.get(c, set())) for c in comms_with_planted_accounts),
            default=0,
        )

        homog = float(homogeneity_score(y_true_planted, y_pred_comm)) if y_true_planted else 0.0
        comp = float(completeness_score(y_true_planted, y_pred_comm)) if y_true_planted else 0.0
        v_meas = float(v_measure_score(y_true_planted, y_pred_comm)) if y_true_planted else 0.0

        group_summary = {
            "pattern": pattern,
            "camouflage": f"{int(camo*100)}%",
            "planted": len(ring_ids),
            "recovered": recovered_in_group,
            "recovery_rate": round(recovered_in_group / len(ring_ids), 4) if ring_ids else 0.0,
            "ring_contained_rate": round(float(np.mean(contained_rates)), 4) if contained_rates else 0.0,
            "mean_ring_purity": round(float(np.mean(purities)), 4) if purities else 0.0,
            "largest_component_size_with_planted": int(largest_comp_size),
            "homogeneity": round(homog, 4),
            "completeness": round(comp, 4),
            "v_measure": round(v_meas, 4),
            "contamination": round(float(np.mean(contaminations)), 4) if contaminations else 0.0,
        }
        detailed_eval.append(group_summary)

    return {
        "label": "synthetic injection for validation",
        "total_planted_rings": total_planted,
        "total_recovered_rings": total_recovered,
        "overall_recovery_rate": round(total_recovered / total_planted, 4) if total_planted > 0 else 0.0,
        "results_by_pattern_and_camouflage": detailed_eval,
    }


def run_planted_ring_validation(
    data_path: Optional[Path] = None,
    output_dir: Path = Path("outputs"),
    sample_rows: int = 200_000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run full planted ring injection and recovery benchmark on 200k PaySim sample using CC and LPA."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ram_conf = check_ram_preflight()
    csv_path = data_path or ensure_dataset()

    LOGGER.info("Loading %d uniform row sample from %s", sample_rows, csv_path)
    df_sample = pd.read_csv(csv_path, nrows=sample_rows * 2)
    df_sample = df_sample.sample(n=min(sample_rows, len(df_sample)), random_state=seed)

    normal_accounts = list(set(df_sample["nameOrig"]).union(set(df_sample["nameDest"])))
    LOGGER.info("Sample accounts: %d. Injecting 60 planted rings...", len(normal_accounts))

    df_planted_txs, ground_truth = generate_planted_rings(normal_accounts, seed=seed)

    # Combine sample with planted transactions
    sample_edges = df_sample[["nameOrig", "nameDest"]].copy()
    planted_edges = df_planted_txs[["nameOrig", "nameDest"]].copy()
    all_edges_pdf = pd.concat([sample_edges, planted_edges], ignore_index=True).drop_duplicates()

    all_nodes = list(set(all_edges_pdf["nameOrig"]).union(set(all_edges_pdf["nameDest"])))
    all_nodes_pdf = pd.DataFrame({"id": all_nodes})

    spark = create_spark_session(
        app_name="PaySimPlantedRingValidation",
        driver_memory=ram_conf["driver_memory"],
        shuffle_partitions=ram_conf["shuffle_partitions"],
    )

    try:
        LOGGER.info(
            "Graph initialized: %d nodes, %d edges. Running Connected Components...",
            len(all_nodes),
            len(all_edges_pdf),
        )
        vertices_df = spark.createDataFrame(all_nodes_pdf).cache()
        edges_df = spark.createDataFrame(
            all_edges_pdf.rename(columns={"nameOrig": "src", "nameDest": "dst"})
        ).cache()

        # 1. Connected Components
        comp_df, cc_info = compute_connected_components(vertices_df, edges_df, max_iter=15)
        comp_pdf = comp_df.toPandas()
        account_to_cc = dict(zip(comp_pdf["id"], comp_pdf["component_id"]))
        LOGGER.info("Connected components finished. Info: %s", cc_info)

        # 2. Label Propagation (LPA)
        LOGGER.info("Running Label Propagation Algorithm (LPA)...")
        lpa_df, lpa_info = compute_label_propagation(vertices_df, edges_df, max_iter=5)
        lpa_pdf = lpa_df.toPandas()
        account_to_lpa = dict(zip(lpa_pdf["id"], lpa_pdf["community_id"]))
        LOGGER.info("Label propagation finished. Info: %s", lpa_info)

    finally:
        spark.stop()

    LOGGER.info("Evaluating Connected Components recovery against ground truth planted rings...")
    eval_cc = evaluate_ring_recovery(ground_truth, account_to_cc)

    LOGGER.info("Evaluating Label Propagation recovery against ground truth planted rings...")
    eval_lpa = evaluate_ring_recovery(ground_truth, account_to_lpa)

    eval_report = {
        "label": "synthetic injection for validation",
        "sample_size": sample_rows,
        "total_planted_rings": 60,
        "spark_config": {
            "driver_memory": ram_conf["driver_memory"],
            "shuffle_partitions": ram_conf["shuffle_partitions"],
        },
        "algorithms": {
            "connected_components": eval_cc,
            "label_propagation": eval_lpa,
        },
        "results_by_pattern_and_camouflage": eval_cc["results_by_pattern_and_camouflage"],
        "results_by_pattern_and_camouflage_lpa": eval_lpa["results_by_pattern_and_camouflage"],
        "total_recovered_rings": eval_cc["total_recovered_rings"],
        "overall_recovery_rate": eval_cc["overall_recovery_rate"],
    }

    # Save validation report
    report_file = output_dir / "planted_ring_validation.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)

    LOGGER.info("Wrote planted ring validation report to %s", report_file)
    return eval_report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run synthetic planted-ring validation with camouflage.")
    parser.add_argument("--data", type=Path, default=None, help="Path to PaySim CSV file.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory.")
    parser.add_argument("--sample-rows", type=int, default=200_000, help="Number of sample rows.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    run_planted_ring_validation(args.data, args.output_dir, args.sample_rows, args.seed)


if __name__ == "__main__":
    main()
