"""System and execution environment metadata collector."""

from __future__ import annotations

import datetime
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import psutil
import pyspark
import sklearn
import xgboost as xgb

from provenance import sha256


def get_java_version() -> str:
    """Retrieve the active JVM version string."""
    try:
        proc = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = (proc.stdout + proc.stderr).strip()
        first_line = output.splitlines()[0] if output else "unknown"
        return first_line
    except Exception as exc:
        return f"java_query_error: {exc}"


def get_git_info() -> Dict[str, str]:
    """Retrieve current Git commit and branch."""
    commit = "unknown"
    branch = "unknown"
    try:
        proc_c = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc_c.returncode == 0:
            commit = proc_c.stdout.strip()

        proc_b = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc_b.returncode == 0:
            branch = proc_b.stdout.strip()
    except Exception:
        pass
    return {"commit": commit, "branch": branch}


def collect_run_info(
    dataset_path: Optional[Path | str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Collect non-empty environment, hardware, version, and input dataset metadata."""
    git_info = get_git_info()
    v_mem = psutil.virtual_memory()

    # Dataset details
    ds_file = "unknown"
    ds_size = 0
    ds_hash = "unknown"
    if dataset_path:
        ds_p = Path(dataset_path)
        if ds_p.exists():
            ds_file = ds_p.name
            ds_size = ds_p.stat().st_size
            try:
                ds_hash = sha256(ds_p)
            except Exception:
                ds_hash = "hash_error"

    cfg = config.copy() if config else {}
    cfg.setdefault("SPARK_DRIVER_MEMORY", os.getenv("SPARK_DRIVER_MEMORY", "6g"))
    cfg.setdefault("SPARK_SHUFFLE_PARTITIONS", os.getenv("SPARK_SHUFFLE_PARTITIONS", "48"))

    info: Dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "java": get_java_version(),
        "pyspark": pyspark.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "xgboost": xgb.__version__,
        "git_commit": git_info["commit"],
        "git_branch": git_info["branch"],
        "cores": {
            "logical": os.cpu_count() or 1,
            "physical": psutil.cpu_count(logical=False) or 1,
        },
        "ram_gb": {
            "total": round(v_mem.total / (1024**3), 2),
            "available": round(v_mem.available / (1024**3), 2),
        },
        "config": cfg,
        "dataset": {
            "file": ds_file,
            "size_bytes": ds_size,
            "sha256": ds_hash,
        },
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    return info


def write_run_info(
    output_path: Path | str,
    dataset_path: Optional[Path | str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Capture and write run_info.json."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    info = collect_run_info(dataset_path=dataset_path, config=config)
    with out.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return out


if __name__ == "__main__":
    test_out = Path("outputs_test") / "run_info.json"
    ds_csv = next(iter(sorted(Path("data").glob("*.csv"))), None)
    write_run_info(test_out, dataset_path=ds_csv, config={"mode": "test"})
    print(f"Generated {test_out}")
