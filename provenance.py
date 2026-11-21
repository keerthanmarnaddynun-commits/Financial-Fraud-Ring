"""Provenance and artifact tracking utilities."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


def sha256(path: Union[str, Path]) -> str:
    """Calculate the SHA-256 hex digest of a file in 8MB chunks."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found for hashing: {file_path}")
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024 * 8):
            hasher.update(chunk)
    return hasher.hexdigest()


def make_readonly(path: Union[str, Path]) -> None:
    """Set read-only attributes on a file or folder tree."""
    target = Path(path)
    if target.is_file():
        try:
            target.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
        except OSError:
            pass
    elif target.is_dir():
        for child in target.rglob("*"):
            if child.is_file():
                try:
                    child.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
                except OSError:
                    pass


def _lookup_key(data: Any, path_expr: Union[str, List[str]]) -> Any:
    """Traverse nested dict/list using dot-separated keys or key sequence."""
    if isinstance(path_expr, str):
        keys = path_expr.split(".")
    else:
        keys = list(path_expr)
    current = data
    for k in keys:
        if isinstance(current, dict):
            if k in current:
                current = current[k]
            else:
                return "n/a"
        elif isinstance(current, (list, tuple)):
            try:
                idx = int(k)
                current = current[idx]
            except (ValueError, IndexError):
                return "n/a"
        else:
            return "n/a"
    return current


def write_manifest(run_dir: Union[str, Path]) -> Path:
    """Generate SHA256SUMS.txt for all files in run_dir."""
    directory = Path(run_dir)
    manifest_path = directory / "SHA256SUMS.txt"
    entries: List[str] = []
    for item in sorted(directory.rglob("*")):
        if item.is_file() and item.name != "SHA256SUMS.txt":
            rel_path = item.relative_to(directory).as_posix()
            file_hash = sha256(item)
            entries.append(f"{file_hash}  {rel_path}")
    manifest_content = "\n".join(entries) + ("\n" if entries else "")
    manifest_path.write_text(manifest_content, encoding="utf-8")
    return manifest_path


def write_card(
    run_dir: Union[str, Path],
    run_id: str,
    purpose: str,
    command: str,
    inputs: List[Tuple[str, str]],
    outputs: List[Tuple[str, str]],
    key_numbers: List[Tuple[str, str, Union[str, List[str]]]],
    caveats: List[str],
    not_shown: List[str],
) -> Path:
    """Generate CARD.md describing an experiment run and its key findings.

    Args:
        run_dir: Directory where the run artifacts are located.
        run_id: Identifier of the run.
        purpose: Concise purpose of the run.
        command: Command executed to produce the run.
        inputs: List of (name, path_or_hash).
        outputs: List of (name, path_or_description).
        key_numbers: List of (label, file_in_run_dir, json_path) read programmatically.
        caveats: Structural caveats, limitations, noise notes.
        not_shown: Explicit statements of what the result does NOT show.
    """
    directory = Path(run_dir)
    card_path = directory / "CARD.md"

    # Programmatically read key numbers
    key_num_rows: List[str] = []
    for label, file_name, path_expr in key_numbers:
        f_path = directory / file_name
        if not f_path.exists():
            val = "n/a (file missing)"
        else:
            try:
                with f_path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                val = _lookup_key(data, path_expr)
            except Exception as exc:
                val = f"error: {exc}"
        key_num_rows.append(f"| {label} | `{val}` | `{file_name}` | `{path_expr}` |")

    inputs_rows = "\n".join(f"- **{name}**: `{spec}`" for name, spec in inputs)
    outputs_rows = "\n".join(f"- **{name}**: `{spec}`" for name, spec in outputs)
    caveats_rows = "\n".join(f"- {c}" for c in caveats)
    not_shown_rows = "\n".join(f"- {ns}" for ns in not_shown)
    table_header = "| Metric / Label | Value | Source File | JSON Path |\n| --- | --- | --- | --- |"
    table_body = "\n".join(key_num_rows) if key_num_rows else "| (none) | - | - | - |"

    content = f"""# Results Card: {run_id}

## Purpose
{purpose}

## Execution Command
```bash
{command}
```

## Inputs
{inputs_rows}

## Outputs
{outputs_rows}

## Key Measured Numbers (Read from Output Artifacts)
{table_header}
{table_body}

## Caveats and Limitations
{caveats_rows}

## What This Result Does NOT Show
{not_shown_rows}
"""
    card_path.write_text(content, encoding="utf-8")
    return card_path
