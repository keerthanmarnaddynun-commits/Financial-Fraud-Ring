"""Strict read-only consistency, banned-word, and relative-link checker for project docs.

Checks performed (never modifies any repository file):
  (a) Regenerates all documents via make_report.py into a temporary directory and compares
      byte-for-byte against the workspace copies.
  (b) Scans README.md, docs/ text/markdown/template files, app.py string literals, and
      Python comments across active project modules for Appendix A banned words plus
      "verified", "confirmed", and "audited".
  (c) Scans README.md and docs/**/*.md for broken relative Markdown links.
  (d) Prints explicit COVERAGE detailing what is checked and what is NOT checked.
"""

from __future__ import annotations

import io
import re
import sys
import tempfile
import tokenize
from pathlib import Path
from typing import Dict, List, Tuple

import make_report

BANNED_WORDS = [
    "optimal",
    "sweet spot",
    "robust",
    "decisively",
    "catastrophic",
    "proves",
    "guarantees",
    "perfect",
    "state-of-the-art",
    "outperforms",
    "verified",
    "confirmed",
    "audited",
]

BANNED_RE = re.compile(
    r"(?i)\b(" + "|".join(re.escape(w) for w in BANNED_WORDS) + r")\b"
)

MD_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")

ACTIVE_PYTHON_MODULES = [
    "main.py",
    "graph_pipeline.py",
    "spark_session.py",
    "data_ingestion.py",
    "graph_construction.py",
    "feature_engineering.py",
    "feature_sets.py",
    "splits.py",
    "models.py",
    "evaluation.py",
    "stage_timer.py",
    "run_info.py",
    "provenance.py",
    "business_impact.py",
    "rings.py",
    "planted_rings.py",
    "to_parquet.py",
    "storage_benchmark.py",
    "benchmark_scaling.py",
    "aggregate_scaling.py",
    "pandas_vs_spark.py",
    "generate_visuals.py",
    "make_report.py",
    "demo_run.py",
    "app.py",
]


def check_regenerated_docs(project_root: Path) -> Tuple[List[str], List[str]]:
    """Regenerate documents into a temp directory and compare byte-for-byte."""
    checked: List[str] = []
    mismatches: List[str] = []

    with tempfile.TemporaryDirectory(prefix="check_docs_") as tmp_dir:
        generated = make_report.render_all(
            outputs_dir=project_root / "outputs",
            templates_dir=project_root / "docs" / "templates",
            dest_root=Path(tmp_dir),
            project_root=project_root,
            update_snapshot=False,
        )
        for rel_path, gen_path in sorted(generated.items()):
            checked.append(rel_path)
            committed_path = project_root / rel_path
            if not committed_path.exists():
                mismatches.append(f"MISSING committed file: {rel_path}")
                continue
            gen_bytes = gen_path.read_bytes()
            com_bytes = committed_path.read_bytes()
            if gen_bytes != com_bytes:
                mismatches.append(
                    f"MISMATCH {rel_path}: workspace={len(com_bytes)}B vs regenerated={len(gen_bytes)}B"
                )
    return checked, mismatches


def _scan_text_lines(label: str, text: str) -> List[str]:
    hits: List[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for m in BANNED_RE.finditer(line):
            excerpt = line.strip()[:100]
            hits.append(f"{label}:{lineno} matched '{m.group(0)}' -> {excerpt}")
    return hits


def _scan_python_comments_and_strings(py_path: Path, rel_label: str, include_strings: bool) -> List[str]:
    hits: List[str] = []
    src = py_path.read_text(encoding="utf-8", errors="replace")
    try:
        tokens = tokenize.generate_tokens(io.StringIO(src).readline)
        for tok in tokens:
            if tok.type == tokenize.COMMENT or (include_strings and tok.type == tokenize.STRING):
                for m in BANNED_RE.finditer(tok.string):
                    excerpt = tok.string.replace("\n", " ").strip()[:100]
                    kind = "comment" if tok.type == tokenize.COMMENT else "string"
                    hits.append(
                        f"{rel_label}:{tok.start[0]} ({kind}) matched '{m.group(0)}' -> {excerpt}"
                    )
    except tokenize.TokenError as exc:
        hits.append(f"{rel_label}: tokenize error: {exc}")
    return hits


def check_banned_words(project_root: Path) -> Tuple[List[str], List[str]]:
    """Search README.md, docs/, app.py strings, and active Python code comments."""
    scanned: List[str] = []
    hits: List[str] = []

    # 1. README.md
    readme = project_root / "README.md"
    if readme.exists():
        scanned.append("README.md")
        hits.extend(_scan_text_lines("README.md", readme.read_text(encoding="utf-8", errors="replace")))

    # 2. docs/ markdown, text, and template files
    docs_dir = project_root / "docs"
    if docs_dir.exists():
        for p in sorted(docs_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in (".md", ".txt", ".tmpl"):
                rel = p.relative_to(project_root).as_posix()
                scanned.append(rel)
                hits.extend(_scan_text_lines(rel, p.read_text(encoding="utf-8", errors="replace")))

    # 3. Active Python modules (comments everywhere; strings in app.py)
    for mod in ACTIVE_PYTHON_MODULES:
        p = project_root / mod
        if p.exists():
            include_strings = mod == "app.py"
            label = f"{mod} ({'comments+strings' if include_strings else 'comments'})"
            scanned.append(label)
            hits.extend(_scan_python_comments_and_strings(p, mod, include_strings=include_strings))

    return scanned, hits


def check_relative_links(project_root: Path) -> Tuple[List[str], List[str]]:
    """List broken relative Markdown links in README.md and docs/**/*.md."""
    md_files: List[Path] = []
    readme = project_root / "README.md"
    if readme.exists():
        md_files.append(readme)
    docs_dir = project_root / "docs"
    if docs_dir.exists():
        md_files.extend(sorted(docs_dir.rglob("*.md")))

    checked: List[str] = []
    broken: List[str] = []
    for md_path in md_files:
        rel_md = md_path.relative_to(project_root).as_posix()
        checked.append(rel_md)
        in_code_fence = False
        for lineno, line in enumerate(md_path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code_fence = not in_code_fence
                continue
            if in_code_fence:
                continue
            for m in MD_LINK_RE.finditer(line):
                raw_target = m.group(1).strip()
                # Strip optional markdown link title
                target = raw_target.split()[0] if raw_target else ""
                if (
                    not target
                    or target.startswith(("http://", "https://", "mailto:", "#", "file://"))
                ):
                    continue
                target_no_anchor = target.split("#", 1)[0]
                if not target_no_anchor:
                    continue
                resolved = (md_path.parent / target_no_anchor).resolve()
                if not resolved.exists():
                    broken.append(f"{rel_md}:{lineno} -> broken link '{target}'")
    return checked, broken


def main() -> None:
    project_root = Path(__file__).resolve().parent

    print("=== CHECK_DOCS (READ-ONLY CONSISTENCY CHECK) ===")
    print()

    # (a) Byte-for-byte regeneration check
    regen_checked, regen_mismatches = check_regenerated_docs(project_root)
    print(f"[A] Byte-for-byte template regeneration ({len(regen_checked)} files):")
    for f in regen_checked:
        print(f"    - {f}")
    if regen_mismatches:
        print(f"    STATUS: FAIL ({len(regen_mismatches)} mismatches)")
        for item in regen_mismatches:
            print(f"      * {item}")
    else:
        print("    STATUS: PASS (0 byte differences)")
    print()

    # (b) Banned words check
    bw_scanned, bw_hits = check_banned_words(project_root)
    print(f"[B] Banned vocabulary scan ({len(bw_scanned)} targets):")
    if bw_hits:
        print(f"    STATUS: FAIL ({len(bw_hits)} hits)")
        for h in bw_hits:
            print(f"      * {h}")
    else:
        print("    STATUS: PASS (0 banned-word hits)")
    print()

    # (c) Broken relative links check
    link_files, broken_links = check_relative_links(project_root)
    print(f"[C] Relative Markdown link check ({len(link_files)} files):")
    if broken_links:
        print(f"    STATUS: FAIL ({len(broken_links)} broken links)")
        for b in broken_links:
            print(f"      * {b}")
    else:
        print("    STATUS: PASS (0 broken relative links)")
    print()

    # (d) Explicit COVERAGE disclosure
    print("[D] COVERAGE")
    print("  Checked:")
    print("    1. Byte-for-byte identity between make_report.py temp regeneration and workspace files: " + ", ".join(regen_checked))
    print("    2. Banned words (" + ", ".join(BANNED_WORDS) + ") in README.md, docs/*.(md|txt|tmpl), app.py string literals, and Python comments across active project modules.")
    print("    3. Local existence of all relative Markdown links in README.md and docs/**/*.md.")
    print("  NOT Checked:")
    print("    1. Does NOT re-run Spark pipelines, benchmarks, or pytest tests to recompute JSON values in outputs/.")
    print("    2. Does NOT inspect deferred/unwritten docs (docs/paper_final.md, docs/ARCHITECTURE.md, docs/DEMO_SCRIPT.txt, docs/VIVA_QA.txt) or legacy research_paper_draft.md.")
    print("    3. Does NOT inspect check_docs.py itself, tests/, scratch/, archive/, runs/, logs/, old_app_reference.py, or app_mine.py.")
    print("    4. Does NOT check external HTTP/HTTPS URLs or PNG chart pixel contents.")
    print("    5. Does NOT flag standard technical terms 'validation'/'validate' in code identifiers or filenames.")
    print()

    total_issues = len(regen_mismatches) + len(bw_hits) + len(broken_links)
    print(f"OVERALL: {'PASS' if total_issues == 0 else 'FAIL'} (total_issues={total_issues})")
    sys.exit(0 if total_issues == 0 else 1)


if __name__ == "__main__":
    main()
