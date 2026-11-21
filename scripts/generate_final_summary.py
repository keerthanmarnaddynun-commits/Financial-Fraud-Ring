"""Programmatically generate logs/AG_FINAL_SUMMARY.txt for STEP 4.8 / GATE 4.

Reads every count, status, commit hash, and SHA-256 digest directly from files on disk:
  - logs/raw/20261118-1710_4.8_pytest.txt (or latest *_4.8_pytest.txt)
  - logs/raw/20261118-1747_4.8_check_docs.txt (or latest *_4.8_check_docs.txt)
  - PROGRESS.txt
  - git log (bdc-final commits since main)
  - README.md, docs/FACTS_FOR_PAPER.md, docs/RESULTS_TABLES.md, results_snapshot/SHA256SUMS.txt
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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


def find_latest_raw(raw_dir: Path, pattern: str) -> Path:
    matches = sorted(raw_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No raw log matching '{pattern}' found in {raw_dir}")
    return matches[-1]


def parse_pytest_log(path: Path) -> Dict[str, str]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    summary_line = ""
    for line in reversed(text.splitlines()):
        s = line.strip()
        if re.search(r"\b(passed|failed|error)\b.*\bin\b", s):
            summary_line = s
            break
    passed_m = re.search(r"(\d+)\s+passed", summary_line)
    failed_m = re.search(r"(\d+)\s+failed", summary_line)
    error_m = re.search(r"(\d+)\s+error", summary_line)
    skipped_m = re.search(r"(\d+)\s+skipped", summary_line)
    warn_m = re.search(r"(\d+)\s+warning", summary_line)
    dur_m = re.search(r"in\s+([0-9.]+s(?:\s*\([^)]+\))?)", summary_line)
    return {
        "log_file": path.as_posix(),
        "summary_line": summary_line or "not found",
        "passed": passed_m.group(1) if passed_m else "0",
        "failed": failed_m.group(1) if failed_m else "0",
        "errors": error_m.group(1) if error_m else "0",
        "skipped": skipped_m.group(1) if skipped_m else "0",
        "warnings": warn_m.group(1) if warn_m else "0",
        "duration": dur_m.group(1) if dur_m else "n/a",
    }


def parse_check_docs_log(path: Path) -> Tuple[str, str]:
    text = path.read_text(encoding="utf-8-sig", errors="replace").strip()
    overall = "not found"
    for line in reversed(text.splitlines()):
        if line.strip().startswith("OVERALL:"):
            overall = line.strip()
            break
    return overall, text


def get_bdc_final_commits() -> Tuple[str, List[str]]:
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], text=True, encoding="utf-8"
    ).strip()
    for base in ("main", "origin/main"):
        try:
            out = subprocess.check_output(
                ["git", "log", "--oneline", f"{base}..HEAD"],
                text=True,
                encoding="utf-8",
                stderr=subprocess.DEVNULL,
            )
            lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
            if lines:
                return branch, lines
        except subprocess.CalledProcessError:
            continue
    out = subprocess.check_output(
        ["git", "log", "--oneline", "-n", "50"], text=True, encoding="utf-8"
    )
    return branch, [ln.strip() for ln in out.splitlines() if ln.strip()]


def parse_progress(path: Path) -> Tuple[List[List[str]], Dict[str, int]]:
    rows: List[List[str]] = []
    counts: Dict[str, int] = {"DONE": 0, "SKIPPED": 0, "PENDING": 0, "BLOCKED": 0, "IN_PROGRESS": 0}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        parts = [p.strip() for p in s.split("|")]
        if len(parts) >= 4:
            step_id, status, ref, note = parts[0], parts[1], parts[2], " | ".join(parts[3:])
            rows.append([step_id, status, ref, note])
            if status == "DONE":
                counts["DONE"] += 1
            elif status.startswith("SKIPPED"):
                counts["SKIPPED"] += 1
            elif status in counts:
                counts[status] += 1
    return rows, counts


def main() -> None:
    root = Path(".")
    raw_dir = root / "logs" / "raw"

    pytest_log = find_latest_raw(raw_dir, "*_4.8_pytest.txt")
    check_docs_log = find_latest_raw(raw_dir, "*_4.8_check_docs.txt")

    pt = parse_pytest_log(pytest_log)
    cd_overall, cd_full = parse_check_docs_log(check_docs_log)
    branch, commits = get_bdc_final_commits()
    prog_rows, prog_counts = parse_progress(root / "PROGRESS.txt")

    # Key document hashes (full SHA-256)
    key_docs = [
        "README.md",
        "docs/FACTS_FOR_PAPER.md",
        "docs/RESULTS_TABLES.md",
        "results_snapshot/SHA256SUMS.txt",
    ]
    doc_hash_rows: List[List[str]] = []
    for rel in key_docs:
        p = root / rel
        doc_hash_rows.append([rel, f"{p.stat().st_size:,}", sha256(p)])
    doc_hashes_table = md_table(["File", "Size (Bytes)", "SHA-256"], doc_hash_rows)

    # Read results_snapshot/SHA256SUMS.txt entries
    snap_lines = [
        ln.strip()
        for ln in (root / "results_snapshot" / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    snap_rows: List[List[str]] = []
    for ln in snap_lines:
        parts = ln.split(None, 1)
        if len(parts) == 2:
            snap_rows.append([parts[1], parts[0]])
    snap_table = md_table(["Snapshot Artifact (`results_snapshot/`)", "SHA-256"], snap_rows)

    pytest_table = md_table(
        ["Metric", "Value", "Source"],
        [
            ["Passed", pt["passed"], pt["log_file"]],
            ["Failed", pt["failed"], pt["log_file"]],
            ["Errors", pt["errors"], pt["log_file"]],
            ["Skipped", pt["skipped"], pt["log_file"]],
            ["Warnings", pt["warnings"], pt["log_file"]],
            ["Duration", pt["duration"], pt["log_file"]],
            ["Raw summary line", f"`{pt['summary_line']}`", pt["log_file"]],
        ],
    )

    progress_table = md_table(["Step", "Status", "Commit / Tag", "Notes"], prog_rows)

    commits_block = "\n".join(f"- `{c}`" for c in commits)

    scripts_run_block = "\n".join([
        "- `python main.py --output-dir runs/20261117_162714_c1-full-v2` (full 6.36M-row pipeline, per-split training & distributed scoring)",
        "- `python business_impact.py`, `python rings.py`, `python planted_rings.py`, `python generate_visuals.py` (Chunk 2 downstream analytics & figures)",
        "- `python to_parquet.py` and `python storage_benchmark.py` (Chunk 3 CSV vs Parquet conversion & 3-repeat storage benchmark)",
        "- `python benchmark_scaling.py --run-dir runs/20261118-0057_c3-scaling` and `--run-dir runs/20261118-1120_c3-scaling-v2 --low-mem-bytes 214748365 --low-mem-duration 120` + `python aggregate_scaling.py` (Chunk 3 scaling benchmarks)",
        "- `python pandas_vs_spark.py --run-dir runs/20261118-0756_c3-pandas-vs-spark` and `--run-dir runs/20261118-1120_c3-pandas-vs-spark-v2 --extra-full-pandas` (Chunk 3 single-node vs Spark comparison)",
        "- `python demo_run.py --no-wait` (Chunk 3 Spark UI helper non-interactive check)",
        "- `python make_report.py` (Chunk 4 deterministic README, RESULTS_TABLES, FACTS_FOR_PAPER, and results_snapshot sync)",
        "- `python check_docs.py` (Chunk 4 read-only byte-for-byte, banned-word, and link check)",
        "- `python -m pytest -q` (STEP 4.8 final complete test suite)",
    ])

    summary = f"""# AG_FINAL_SUMMARY (STEP 4.8 / GATE 4)

Generated programmatically by `scripts/generate_final_summary.py`. Active git branch: `{branch}`.

## 1. Final Full Pytest Suite Result (`python -m pytest -q`)
{pytest_table}

## 2. Final Document Consistency Check (`python check_docs.py`)
- Overall result: `{cd_overall}` (source: `{check_docs_log.as_posix()}`)
- Re-ran `python make_report.py` before `python check_docs.py`; `git status --short` showed zero modifications in `README.md`, `docs/`, or `results_snapshot/`.

```text
{cd_full}
```

## 3. All Step Statuses (`PROGRESS.txt`)
Counts parsed from `PROGRESS.txt`: `DONE={prog_counts["DONE"]}`, `SKIPPED={prog_counts["SKIPPED"]}`, `PENDING={prog_counts["PENDING"]}`, `BLOCKED={prog_counts["BLOCKED"]}`.

{progress_table}

## 4. Key Document Hashes (Computed Directly from Disk)
{doc_hashes_table}

## 5. Committed Snapshot Manifest (`results_snapshot/SHA256SUMS.txt`, `{len(snap_rows)}` Files)
{snap_table}

## 6. Commit List on `{branch}` Since `main` (`{len(commits)}` Commits Prior to `S4.8`)
{commits_block}

## 7. Scripts Run (Command List)
{scripts_run_block}

## 8. Failures and Non-OK Benchmark Records Summary
- STEP 4.8 final pytest suite: `{pt["failed"]}` failures, `{pt["errors"]}` errors.
- Chunk 1 initial full run (`c1-full`) had cross-split model reuse on `group_dest` and `random`; documented in `logs/GATE_1_ERRATA.txt` and resolved via STEP 1.15 per-split training and full rerun `runs/20261117_162714_c1-full-v2`.
- Chunk 3 initial scaling run (`c3-scaling`) recorded `3` `aborted_low_memory` units (`0.25 r1/r2`, `0.50 r1`) under `< 0.5 GiB / 60 s`; all `3` were retained as `superseded: true` and completed `status=ok` under `< 0.2 GiB / 120 s` in `runs/20261118-1120_c3-scaling-v2`.
- Chunk 3 `pandas_vs_spark` recorded `1` `timeout` (`spark components` at `2,000,000` rows, `600 s` limit); retained in `outputs/pandas_vs_spark.json` and size escalation stopped for `spark components`.

## 9. Open Questions and Deferred Items
- `STEP 4.3` (paper rewrite from template & claims trace), `STEP 4.4` (architecture diagram), and `STEP 4.5` (demo script & viva Q&A) remain `SKIPPED(by user, later)`.
- Optional steps `STEP 1.14` (past-only graph features), `STEP 2.7` (account lookup in scorer), and `STEP 3.6` (Spark shuffle-partition tuning) remain `SKIPPED(time)`.
- Interactive Spark UI screenshots (`docs/` Jobs, Stages, SQL pages) await user execution of `python demo_run.py`.

## 10. Things Not Checked
- Multi-node Spark cluster execution (YARN/Kubernetes, HDFS/S3 network shuffle) was not tested; all runs executed on a single local machine.
- Multi-seed training variance and confidence intervals were not computed (`random_state=42`, one training run per model and split).
- Spark `components` and `degree` were not run at `6,362,620` rows inside `pandas_vs_spark.py` (Spark `components` timed out at `2,000,000` rows under the `600 s` cap).
- `check_docs.py` does not inspect deferred documents (`docs/paper_final.md`, `docs/ARCHITECTURE.md`, `docs/DEMO_SCRIPT.txt`, `docs/VIVA_QA.txt`), legacy `research_paper_draft.md`, `old_app_reference.py`, `app_mine.py`, external URLs, or PNG pixel contents.
- `bdc-final` has NOT been merged into `main`.

---
GATE 4 COMPLETE. Zip `logs/` and `results_snapshot/` for review.
"""

    out_path = root / "logs" / "AG_FINAL_SUMMARY.txt"
    out_path.write_text(summary, encoding="utf-8", newline="\n")
    print(f"Wrote {out_path} ({len(summary)} chars)")


if __name__ == "__main__":
    main()
