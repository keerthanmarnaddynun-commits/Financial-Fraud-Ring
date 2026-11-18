"""Programmatically generate logs/GATE_4A_SUMMARY.txt from Chunk 4a files, PROGRESS.txt, and check_docs output."""

from __future__ import annotations

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
        line.strip()
        for line in (root / "PROGRESS.txt").read_text(encoding="utf-8").splitlines()
        if line.strip().startswith(("STEP 4.", "GATE 4"))
    ]
    git_lines = [
        line.strip()
        for line in subprocess.check_output(["git", "log", "--oneline", "-n", "12"], text=True, encoding="utf-8").splitlines()
        if "S4." in line
    ]
    check_docs_raw = (root / "logs" / "raw" / "20261118_4.7_check_docs.txt").read_text(encoding="utf-8-sig").strip()

    doc_targets = [
        "README.md",
        "docs/RESULTS_TABLES.md",
        "docs/FACTS_FOR_PAPER.md",
        "docs/templates/README.md.tmpl",
        "results_snapshot/SHA256SUMS.txt",
    ]
    doc_rows: List[List[str]] = []
    for rel in doc_targets:
        p = root / rel
        if p.exists():
            doc_rows.append([rel, f"{p.stat().st_size:,}", sha256(p)[:16] + "..."])
        else:
            doc_rows.append([rel, "missing", "n/a"])
    docs_table = md_table(["Artifact", "Size (Bytes)", "SHA-256 (prefix)"], doc_rows)

    snap_entries = [
        ln.strip()
        for ln in (root / "results_snapshot" / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    archive_items = sorted(
        p.relative_to(root).as_posix()
        for p in (root / "archive" / "web_legacy").rglob("*")
        if p.is_file()
    )

    steps_block = "\n".join(f"- {ln}" for ln in progress_lines)
    commits_block = "\n".join(f"- `{ln}`" for ln in git_lines)
    archive_block = "\n".join(f"- `{item}`" for item in archive_items)

    summary = f"""# GATE 4A SUMMARY (Chunk 4 Battery Scope - Pre-Final Tests)

Generated programmatically by `scripts/generate_gate4a_summary.py` from `PROGRESS.txt`, `git log`, `results_snapshot/SHA256SUMS.txt`, and `logs/raw/20261118_4.7_check_docs.txt`.

## 1. Chunk 4 Steps Status (from `PROGRESS.txt`)
{steps_block}

### Chunk 4 Commits (from `git log`)
{commits_block}

## 2. Generated Documentation and Snapshot Artifacts
`results_snapshot/SHA256SUMS.txt` covers `{len(snap_entries)}` snapshot files.

{docs_table}

## 3. Archived Legacy Web Dashboard (Disk Copy Under Gitignored `archive/`)
Untracked from git via `git rm --cached -r web web_app.py`; files preserved on disk:
{archive_block}

## 4. Read-Only Consistency Check (`python check_docs.py`)
Raw output (`logs/raw/20261118_4.7_check_docs.txt`):
```text
{check_docs_raw}
```

## 5. Battery Constraints & Deferred Items
- Executed on battery: zero Spark runs and zero full pytest runs executed during Chunk 4a.
- `STEP 4.3` (paper), `STEP 4.4` (architecture diagram), and `STEP 4.5` (demo script and viva Q&A) logged as `SKIPPED(by user, later)`.
- `STEP 4.8` (final full pytest suite and `logs/AG_FINAL_SUMMARY.txt`) remains `PENDING` until plugged in at home.

---
READY FOR FINAL TESTS AT HOME
"""

    out_path = root / "logs" / "GATE_4A_SUMMARY.txt"
    out_path.write_text(summary, encoding="utf-8", newline="\n")
    print(f"Wrote {out_path} ({len(summary)} chars)")


if __name__ == "__main__":
    main()
