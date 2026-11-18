"""Aggregate scaling_runs.jsonl into scaling_runs.json, CSV, CARD.md, and SHA256SUMS.txt."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from provenance import make_readonly, write_card, write_manifest

STAGE_KEYS = [
    "ingest_validate",
    "graph_build",
    "degrees",
    "pagerank",
    "triangles",
    "components",
    "lpa",
    "behaviour_features",
    "feature_join",
    "total",
]

DEFAULT_CAVEATS = [
    "Single machine (Windows 11 laptop, 15.37 GiB shared RAM); results do not represent a multi-node cluster.",
    "Hybrid CPU architecture (Intel Core Ultra 7 155H, 16 cores / 22 logical threads mixing performance and efficiency cores); core scaling is uneven across thread counts.",
    "Shared RAM and local NTFS disk I/O with OS background services.",
    "Uniform random transaction sampling (seed=42) changes graph edge density and connected-component structure across fractions; node and edge counts are reported per run.",
    "2 repeats per configuration; both individual repeat values and means are reported.",
    "Default Spark settings used across all runs (spark.driver.memory=6g, spark.sql.shuffle.partitions=48).",
]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _mean_or_none(vals: List[float]) -> float | None:
    return round(statistics.mean(vals), 4) if vals else None


def aggregate_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build structured scaling summary preserving every individual run and non-ok status."""
    data_groups: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    core_groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    non_ok_runs: List[Dict[str, Any]] = []
    total_wall = 0.0

    for rec in records:
        total_wall += float(rec.get("wall_seconds") or (rec.get("seconds") or {}).get("total") or 0.0)
        if rec.get("status") != "ok":
            non_ok_runs.append(rec)
        suite = rec.get("suite")
        if suite == "data_scaling":
            data_groups[round(float(rec["fraction"]), 4)].append(rec)
        elif suite == "core_scaling":
            core_groups[int(rec["cores"])].append(rec)

    data_size_summary: List[Dict[str, Any]] = []
    for frac in sorted(data_groups.keys()):
        grp = sorted(data_groups[frac], key=lambda r: int(r.get("repeat", 0)))
        ok_grp = [r for r in grp if r.get("status") == "ok"]
        totals = [float(r["seconds"]["total"]) for r in ok_grp if isinstance(r.get("seconds"), dict) and "total" in r["seconds"]]
        stage_means: Dict[str, float | None] = {}
        for stg in STAGE_KEYS:
            vals = [
                float(r["seconds"][stg])
                for r in ok_grp
                if isinstance(r.get("seconds"), dict) and stg in r["seconds"] and r["seconds"][stg] is not None
            ]
            stage_means[stg] = _mean_or_none(vals)
        min_rams = [int(r["min_available_ram_bytes"]) for r in grp if r.get("min_available_ram_bytes") is not None]
        peak_rss_vals = [int(r["peak_rss_bytes"]) for r in ok_grp if r.get("peak_rss_bytes") is not None]
        shuffle_vals = [int(r["shuffle_bytes"]) for r in ok_grp if r.get("shuffle_bytes") is not None]
        data_size_summary.append(
            {
                "fraction": frac,
                "master": grp[0].get("master") if grp else "local[*]",
                "ok_count": len(ok_grp),
                "total_count": len(grp),
                "statuses": [r.get("status") for r in grp],
                "mean_rows": round(statistics.mean([int(r["rows"]) for r in ok_grp]), 1) if ok_grp else None,
                "mean_nodes": round(statistics.mean([int(r["nodes"]) for r in ok_grp]), 1) if ok_grp else None,
                "mean_edges": round(statistics.mean([int(r["edges"]) for r in ok_grp]), 1) if ok_grp else None,
                "individual_total_seconds": [round(v, 4) for v in totals],
                "mean_total_seconds": _mean_or_none(totals),
                "mean_stage_seconds": stage_means,
                "mean_peak_rss_bytes": int(round(statistics.mean(peak_rss_vals))) if peak_rss_vals else None,
                "min_available_ram_bytes": min(min_rams) if min_rams else None,
                "mean_shuffle_bytes": int(round(statistics.mean(shuffle_vals))) if shuffle_vals else None,
            }
        )

    # Baseline for speedup: mean total seconds of core_scaling at local[1]
    base_1_grp = [r for r in core_groups.get(1, []) if r.get("status") == "ok"]
    base_1_totals = [float(r["seconds"]["total"]) for r in base_1_grp if isinstance(r.get("seconds"), dict) and "total" in r["seconds"]]
    base_1_mean = statistics.mean(base_1_totals) if base_1_totals else None

    core_scaling_summary: List[Dict[str, Any]] = []
    for cores in sorted(core_groups.keys()):
        grp = sorted(core_groups[cores], key=lambda r: int(r.get("repeat", 0)))
        ok_grp = [r for r in grp if r.get("status") == "ok"]
        totals = [float(r["seconds"]["total"]) for r in ok_grp if isinstance(r.get("seconds"), dict) and "total" in r["seconds"]]
        mean_total = _mean_or_none(totals)
        speedup = round(base_1_mean / mean_total, 4) if (base_1_mean and mean_total and mean_total > 0) else None
        indiv_speedups = (
            [round(base_1_mean / t, 4) for t in totals if t > 0]
            if base_1_mean
            else []
        )
        efficiency = round(speedup / cores, 4) if (speedup is not None and cores > 0) else None

        stage_means = {}
        for stg in STAGE_KEYS:
            vals = [
                float(r["seconds"][stg])
                for r in ok_grp
                if isinstance(r.get("seconds"), dict) and stg in r["seconds"] and r["seconds"][stg] is not None
            ]
            stage_means[stg] = _mean_or_none(vals)

        min_rams = [int(r["min_available_ram_bytes"]) for r in grp if r.get("min_available_ram_bytes") is not None]
        peak_rss_vals = [int(r["peak_rss_bytes"]) for r in ok_grp if r.get("peak_rss_bytes") is not None]
        shuffle_vals = [int(r["shuffle_bytes"]) for r in ok_grp if r.get("shuffle_bytes") is not None]
        core_scaling_summary.append(
            {
                "cores": cores,
                "master": f"local[{cores}]",
                "fraction": round(float(grp[0]["fraction"]), 4) if grp else 0.10,
                "ok_count": len(ok_grp),
                "total_count": len(grp),
                "statuses": [r.get("status") for r in grp],
                "mean_rows": round(statistics.mean([int(r["rows"]) for r in ok_grp]), 1) if ok_grp else None,
                "mean_nodes": round(statistics.mean([int(r["nodes"]) for r in ok_grp]), 1) if ok_grp else None,
                "mean_edges": round(statistics.mean([int(r["edges"]) for r in ok_grp]), 1) if ok_grp else None,
                "individual_total_seconds": [round(v, 4) for v in totals],
                "mean_total_seconds": mean_total,
                "speedup": speedup,
                "individual_speedups": indiv_speedups,
                "efficiency": efficiency,
                "mean_stage_seconds": stage_means,
                "mean_peak_rss_bytes": int(round(statistics.mean(peak_rss_vals))) if peak_rss_vals else None,
                "min_available_ram_bytes": min(min_rams) if min_rams else None,
                "mean_shuffle_bytes": int(round(statistics.mean(shuffle_vals))) if shuffle_vals else None,
            }
        )

    return {
        "benchmark": "spark_scaling",
        "generated_at": datetime.datetime.now().isoformat(),
        "total_runs_recorded": len(records),
        "ok_runs_count": len(records) - len(non_ok_runs),
        "non_ok_runs_count": len(non_ok_runs),
        "total_benchmark_wall_seconds": round(total_wall, 4),
        "core_scaling_fraction": core_scaling_summary[0]["fraction"] if core_scaling_summary else 0.10,
        "core_scaling_baseline_local1_seconds": round(base_1_mean, 4) if base_1_mean else None,
        "data_size_summary": data_size_summary,
        "core_scaling_summary": core_scaling_summary,
        "non_ok_runs": non_ok_runs,
        "runs": records,
        "caveats": DEFAULT_CAVEATS,
    }


def _write_csv(records: List[Dict[str, Any]], csv_path: Path) -> None:
    fieldnames = [
        "suite",
        "fraction",
        "master",
        "cores",
        "repeat",
        "status",
        "rows",
        "nodes",
        "edges",
        "wall_seconds",
        "total_seconds",
        "ingest_validate",
        "graph_build",
        "degrees",
        "pagerank",
        "triangles",
        "components",
        "lpa",
        "behaviour_features",
        "feature_join",
        "peak_rss_bytes",
        "min_available_ram_bytes",
        "shuffle_read_bytes",
        "shuffle_write_bytes",
        "shuffle_bytes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            secs = rec.get("seconds") or {}
            writer.writerow(
                {
                    "suite": rec.get("suite"),
                    "fraction": rec.get("fraction"),
                    "master": rec.get("master"),
                    "cores": rec.get("cores"),
                    "repeat": rec.get("repeat"),
                    "status": rec.get("status"),
                    "rows": rec.get("rows"),
                    "nodes": rec.get("nodes"),
                    "edges": rec.get("edges"),
                    "wall_seconds": rec.get("wall_seconds"),
                    "total_seconds": secs.get("total"),
                    "ingest_validate": secs.get("ingest_validate"),
                    "graph_build": secs.get("graph_build"),
                    "degrees": secs.get("degrees"),
                    "pagerank": secs.get("pagerank"),
                    "triangles": secs.get("triangles"),
                    "components": secs.get("components"),
                    "lpa": secs.get("lpa"),
                    "behaviour_features": secs.get("behaviour_features"),
                    "feature_join": secs.get("feature_join"),
                    "peak_rss_bytes": rec.get("peak_rss_bytes"),
                    "min_available_ram_bytes": rec.get("min_available_ram_bytes"),
                    "shuffle_read_bytes": rec.get("shuffle_read_bytes"),
                    "shuffle_write_bytes": rec.get("shuffle_write_bytes"),
                    "shuffle_bytes": rec.get("shuffle_bytes"),
                }
            )


def aggregate_run_dir(
    run_dir: Path,
    copy_to_outputs: bool = True,
    finalize_readonly: bool = False,
) -> Dict[str, Any]:
    """Read scaling_runs.jsonl in run_dir and write JSON, CSV, CARD.md, SHA256SUMS.txt."""
    jsonl_path = run_dir / "scaling_runs.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"scaling_runs.jsonl not found in {run_dir}")

    records = _load_jsonl(jsonl_path)
    summary = aggregate_records(records)

    out_json = run_dir / "scaling_runs.json"
    out_csv = run_dir / "scaling_runs.csv"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(records, out_csv)

    key_nums = [
        ("Total Runs Recorded", "scaling_runs.json", "total_runs_recorded"),
        ("OK Runs Count", "scaling_runs.json", "ok_runs_count"),
        ("Non-OK Runs Count", "scaling_runs.json", "non_ok_runs_count"),
        ("Total Benchmark Wall Seconds", "scaling_runs.json", "total_benchmark_wall_seconds"),
        ("Core Scaling Fraction", "scaling_runs.json", "core_scaling_fraction"),
        ("Core Scaling local[1] Mean Seconds", "scaling_runs.json", "core_scaling_baseline_local1_seconds"),
    ]
    for idx, row in enumerate(summary["data_size_summary"]):
        key_nums.append(
            (
                f"Data Fraction {row['fraction']} Mean Total Seconds",
                "scaling_runs.json",
                f"data_size_summary.{idx}.mean_total_seconds",
            )
        )
    for idx, row in enumerate(summary["core_scaling_summary"]):
        key_nums.append(
            (
                f"Cores local[{row['cores']}] Mean Seconds",
                "scaling_runs.json",
                f"core_scaling_summary.{idx}.mean_total_seconds",
            )
        )
        key_nums.append(
            (
                f"Cores local[{row['cores']}] Speedup",
                "scaling_runs.json",
                f"core_scaling_summary.{idx}.speedup",
            )
        )

    write_card(
        run_dir=run_dir,
        run_id=run_dir.name,
        purpose="Measure Spark graph-stage execution time, memory footprint, and shuffle volume across PaySim data fractions (0.10, 0.25, 0.50, 1.00) and local core counts (1, 2, 4, 8, 16 at fraction 0.10) with 2 repeats each.",
        command=f"python benchmark_scaling.py --run-dir {run_dir}",
        inputs=[("PaySim Parquet (partitioned by type)", "data/parquet/paysim")],
        outputs=[
            ("scaling_runs.jsonl", "Per-run append-only raw benchmark records"),
            ("scaling_runs.json", "Aggregated per-configuration summaries and all individual runs"),
            ("scaling_runs.csv", "Tabular export of all scaling benchmark runs"),
        ],
        key_numbers=key_nums,
        caveats=DEFAULT_CAVEATS,
        not_shown=[
            "Does not measure distributed multi-node network shuffle or YARN/Kubernetes cluster scheduling.",
            "Does not include scikit-learn/XGBoost model training or distributed batch scoring time.",
        ],
    )
    write_manifest(run_dir)
    if finalize_readonly:
        make_readonly(run_dir)

    if copy_to_outputs:
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_json, outputs_dir / "scaling_runs.json")
        shutil.copy2(out_csv, outputs_dir / "scaling_runs.csv")
        snap_dir = Path("results_snapshot")
        if snap_dir.exists():
            shutil.copy2(out_json, snap_dir / "scaling_runs.json")
            shutil.copy2(out_csv, snap_dir / "scaling_runs.csv")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate scaling benchmark JSONL records.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Scaling benchmark run directory.")
    parser.add_argument("--no-copy-outputs", action="store_true", help="Do not copy outputs to outputs/.")
    parser.add_argument("--finalize", action="store_true", help="Set run folder files read-only after manifest.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = aggregate_run_dir(
        run_dir=args.run_dir,
        copy_to_outputs=not args.no_copy_outputs,
        finalize_readonly=args.finalize,
    )
    print(
        json.dumps(
            {
                "total_runs_recorded": summary["total_runs_recorded"],
                "ok_runs_count": summary["ok_runs_count"],
                "non_ok_runs_count": summary["non_ok_runs_count"],
                "total_benchmark_wall_seconds": summary["total_benchmark_wall_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
