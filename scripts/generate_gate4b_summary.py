"""Programmatically generate logs/GATE_4B_SUMMARY.txt summarizing STEP 4.1b doc fixes."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from provenance import sha256


def md_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def main() -> None:
    root = Path(".")
    progress_lines = [
        ln.strip()
        for ln in (root / "PROGRESS.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip().startswith(("STEP 4.", "GATE 4"))
    ]
    check_docs_raw = (root / "logs" / "raw" / "20261118_4.1b_check_docs.txt").read_text(encoding="utf-8-sig").strip()

    bi = json.loads((root / "outputs" / "business_impact.json").read_text(encoding="utf-8"))
    sc = json.loads((root / "outputs" / "scalability.json").read_text(encoding="utf-8"))
    ri = json.loads((root / "outputs" / "run_info.json").read_text(encoding="utf-8"))
    sr = json.loads((root / "outputs" / "scaling_runs.json").read_text(encoding="utf-8"))
    pvs = json.loads((root / "outputs" / "pandas_vs_spark.json").read_text(encoding="utf-8"))
    st = json.loads((root / "outputs" / "storage_benchmark.json").read_text(encoding="utf-8"))

    # Key script-extracted check values for the summary
    ds_025 = next((r for r in sr.get("data_size_summary", []) if round(float(r.get("fraction", 0)), 2) == 0.25), {})
    min_ram_mib = float(ds_025.get("min_available_ram_bytes", 0)) / (1024 ** 2)
    sp_deg_4m = next((r for r in pvs["summary"] if r["tool"] == "spark" and r["task"] == "degree" and r["rows"] == 4_000_000), {})
    pd_deg_4m = next((r for r in pvs["summary"] if r["tool"] == "pandas_networkx" and r["task"] == "degree" and r["rows"] == 4_000_000), {})
    csv_cnt_reps = st["formats"]["csv"]["operations"]["count_rows"]["repeats_seconds"]
    csv_cnt_warm = sum(csv_cnt_reps[1:]) / len(csv_cnt_reps[1:])
    unp_cnt_reps = st["formats"]["parquet_unpartitioned"]["operations"]["count_rows"]["repeats_seconds"]
    unp_cnt_warm = sum(unp_cnt_reps[1:]) / len(unp_cnt_reps[1:])
    warm_sp_unp_cnt = csv_cnt_warm / unp_cnt_warm

    doc_rows: List[List[str]] = []
    for rel in (
        "make_report.py",
        "docs/templates/README.md.tmpl",
        "README.md",
        "docs/RESULTS_TABLES.md",
        "docs/FACTS_FOR_PAPER.md",
    ):
        p = root / rel
        doc_rows.append([rel, f"{p.stat().st_size:,}", sha256(p)[:16] + "..."])
    docs_table = md_table(["Updated File", "Size (Bytes)", "SHA-256 (prefix)"], doc_rows)

    changes_table = md_table(
        ["Item", "Document Locations", "Summary of Script-Generated Update"],
        [
            ["1", "README Section 1 item 2", "Removed 'leakage-free'; stated validation-only thresholds, per-split training, and label-free transductive graph features."],
            ["2", "README 6.8, FACTS 8, RESULTS 14", f"Added warm mean (repeats 2-3) and warm speedup vs CSV plus cold-cache CSV note (e.g., CSV count_rows warm mean {csv_cnt_warm:.4f}s vs unpartitioned Parquet {unp_cnt_warm:.4f}s = {warm_sp_unp_cnt:.2f}x warm speedup)."],
            ["3", "README 6.7, FACTS 7, RESULTS 13", f"Added reading guide: pandas faster on components (Spark timeout at 2M); at 4M degree compute is {sp_deg_4m['mean_compute_seconds']:.2f}s (Spark) vs {pd_deg_4m['mean_compute_seconds']:.2f}s (pandas) with Spark load separate; Spark not run at 6,362,620 rows."],
            ["4", "README Section 6.0", f"Added 'What the results do and do not show' from outputs/ (m1/m2/m3 row differences, m1_tabular_noerr drop, 0 real rings >=4, components {sc['timings_seconds']['components']:.2f}s of {sc['timings_seconds']['graph_feature_engineering']:.2f}s, 6 or 7 rings/cell, >=80% strict recovery definition)."],
            ["5", "README 7, FACTS 11", f"Added single training run / no CIs (random_state=42), training sampling (max_training_rows={ri['config']['max_training_rows']:,}, sampled_training_rows={sc['sampled_training_rows']:,}), and 0.2 GiB/120s memory guard note (ok min dip {min_ram_mib:.2f} MiB at f=0.25)."],
            ["6", "README 6.1, FACTS 2a-2c, RESULTS 3-5", "Changed m0_rule threshold source column to rule_definition in every model table."],
            ["7", "README 4 & 7, FACTS 9c & 11, RESULTS 16", "Removed 'Intel Core Ultra 7 155H' and 'Windows 11' (absent from outputs/run_info.json); environment table only emits CPU/OS rows if keys exist in run_info.json."],
            ["8", "README Section 6.2", f"Formatted total fraud value with thousands separators ({bi['fraud_value_total']:,.2f})."],
        ],
    )

    steps_block = "\n".join(f"- {ln}" for ln in progress_lines)

    out = f"""# GATE 4B SUMMARY (STEP 4.1b - Chunk 4 Doc Fixes)

Generated programmatically by `scripts/generate_gate4b_summary.py`. Docs only (zero Spark runs, zero full pytest runs).

## 1. Changes Applied (All 8 Items)
{changes_table}

## 2. Updated Files
{docs_table}

## 3. Chunk 4 Status (from `PROGRESS.txt`)
{steps_block}

## 4. Read-Only Consistency Check (`python check_docs.py`)
Raw output (`logs/raw/20261118_4.1b_check_docs.txt`):
```text
{check_docs_raw}
```

---
Stopped at GATE 4B. `STEP 4.8` remains `PENDING` until plugged in at home.
"""

    out_path = root / "logs" / "GATE_4B_SUMMARY.txt"
    out_path.write_text(out, encoding="utf-8", newline="\n")
    print(f"Wrote {out_path} ({len(out)} chars)")


if __name__ == "__main__":
    main()
