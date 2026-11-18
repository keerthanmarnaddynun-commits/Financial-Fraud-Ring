"""Resumable distributed scaling benchmark (data-size scaling and core scaling).

Runs each configuration in an isolated Python subprocess (fresh SparkSession) with
per-run timeout (default 75 minutes) and sustained low-memory guard (available RAM
below 0.5 GiB continuously for 60 seconds). Appends one JSON line per completed or
aborted run to scaling_runs.jsonl and skips already-recorded runs when restarted.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import psutil

from data_ingestion import resolve_input_path
from graph_pipeline import run_graph_stages
from run_info import write_run_info
from spark_session import create_spark_session

LOGGER = logging.getLogger("fraud_analytics.benchmark_scaling")

DEFAULT_TIMEOUT_SECONDS = 75 * 60  # 75 minutes
DEFAULT_LOW_MEM_BYTES = int(0.5 * (1024 ** 3))  # 0.5 GiB = 536,870,912 bytes
DEFAULT_LOW_MEM_DURATION_SECONDS = 60.0
DEFAULT_DRIVER_MEMORY = "6g"
DEFAULT_SHUFFLE_PARTITIONS = 48


def _get_tree_rss_bytes(pid: int) -> int:
    """Return total RSS in bytes for pid and all recursive child processes."""
    try:
        proc = psutil.Process(pid)
        total = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return int(total)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0


def _kill_process_tree(pid: int) -> None:
    """Force-terminate a process and all its descendants on Windows/POSIX."""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for ch in children:
            try:
                ch.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        try:
            parent.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        psutil.wait_procs(children + [parent], timeout=10)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
            )
        except Exception:
            pass


def _run_key(rec: Dict[str, Any]) -> Tuple[str, float, str, int]:
    return (
        str(rec["suite"]),
        round(float(rec["fraction"]), 4),
        str(rec["master"]),
        int(rec["repeat"]),
    )


def _load_completed_keys(jsonl_path: Path) -> Set[Tuple[str, float, str, int]]:
    keys: Set[Tuple[str, float, str, int]] = set()
    if not jsonl_path.exists():
        return keys
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                keys.add(_run_key(rec))
            except Exception:
                continue
    return keys


def _append_jsonl(jsonl_path: Path, record: Dict[str, Any]) -> None:
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_default_plan() -> List[Dict[str, Any]]:
    """Build the Chunk 3 scaling benchmark plan (smallest runs first).

    - Data-size scaling: fractions 0.10, 0.25, 0.50, 1.00; master local[*]; 2 repeats
    - Core scaling: fraction 0.10; master local[1,2,4,8,16]; 2 repeats
    """
    logical_cores = os.cpu_count() or 1
    plan: List[Dict[str, Any]] = []

    # 1. Data-size scaling smallest fraction (0.10) first
    for rep in (1, 2):
        plan.append(
            {
                "suite": "data_scaling",
                "fraction": 0.10,
                "master": "local[*]",
                "cores": logical_cores,
                "repeat": rep,
            }
        )

    # 2. Core scaling at fraction 0.10 (local[1], local[2], local[4], local[8], local[16], 2 repeats)
    for cores in (1, 2, 4, 8, 16):
        for rep in (1, 2):
            plan.append(
                {
                    "suite": "core_scaling",
                    "fraction": 0.10,
                    "master": f"local[{cores}]",
                    "cores": cores,
                    "repeat": rep,
                }
            )

    # 3. Remaining data-size scaling fractions in ascending order: 0.25, 0.50, 1.00
    for frac in (0.25, 0.50, 1.00):
        for rep in (1, 2):
            plan.append(
                {
                    "suite": "data_scaling",
                    "fraction": frac,
                    "master": "local[*]",
                    "cores": logical_cores,
                    "repeat": rep,
                }
            )

    return plan


def run_worker(args: argparse.Namespace) -> None:
    """Execute one isolated Spark graph-stages configuration and write result JSON."""
    input_path = resolve_input_path(args.data, input_format=args.input_format)
    spark = create_spark_session(
        app_name=f"Scaling_{args.suite}_f{args.fraction}_{args.master}_r{args.repeat}",
        master=args.master,
        driver_memory=args.driver_memory,
        shuffle_partitions=args.shuffle_partitions,
    )
    try:
        res = run_graph_stages(
            spark,
            df=None,
            input_path=input_path,
            input_format=args.input_format,
            sample_fraction=args.fraction if args.fraction < 1.0 else None,
            seed=42,
            materialize_enriched=True,
        )
        payload = {
            "status": "ok",
            "suite": args.suite,
            "fraction": float(args.fraction),
            "master": args.master,
            "cores": int(args.cores),
            "repeat": int(args.repeat),
            "input_format": args.input_format,
            "driver_memory": args.driver_memory,
            "shuffle_partitions": int(args.shuffle_partitions),
            "rows": int(res["rows"]),
            "nodes": int(res["nodes"]),
            "edges": int(res["edges"]),
            "account_count": int(res["account_count"]),
            "seconds": res["seconds"],
            "stages": res["stages"],
            "peak_rss_bytes": int(res["peak_rss_bytes"]),
            "min_available_ram_bytes": int(res["min_available_ram_bytes"]),
            "shuffle_read_bytes": int(res["shuffle_read_bytes"]),
            "shuffle_write_bytes": int(res["shuffle_write_bytes"]),
            "shuffle_bytes": int(res["shuffle_bytes"]),
            "engines": res["engines"],
            "algo_info": res["algo_info"],
        }
        Path(args.worker_output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    finally:
        spark.stop()


def execute_single_unit(
    unit: Dict[str, Any],
    run_dir: Path,
    input_format: str,
    data_path: Optional[Path],
    driver_memory: str,
    shuffle_partitions: int,
    timeout_seconds: int,
    low_mem_bytes: int,
    low_mem_duration: float,
) -> Dict[str, Any]:
    """Launch one worker subprocess with timeout and sustained low-memory guard."""
    unit_tag = f"{unit['suite']}_f{unit['fraction']:.2f}_{unit['master'].replace('[', '').replace(']', '').replace('*', 'all')}_r{unit['repeat']}"
    worker_out = run_dir / f"_worker_{unit_tag}.json"
    worker_log = run_dir / f"_worker_{unit_tag}.log"
    if worker_out.exists():
        worker_out.unlink()

    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        "--suite",
        str(unit["suite"]),
        "--fraction",
        str(unit["fraction"]),
        "--master",
        str(unit["master"]),
        "--cores",
        str(unit["cores"]),
        "--repeat",
        str(unit["repeat"]),
        "--input-format",
        input_format,
        "--driver-memory",
        driver_memory,
        "--shuffle-partitions",
        str(shuffle_partitions),
        "--worker-output",
        str(worker_out),
    ]
    if data_path is not None:
        cmd.extend(["--data", str(data_path)])

    started_iso = datetime.datetime.now().isoformat()
    t0 = time.perf_counter()
    min_avail_ram = psutil.virtual_memory().available
    peak_rss = 0
    low_mem_started: Optional[float] = None
    abort_status: Optional[str] = None
    abort_reason: Optional[str] = None

    LOGGER.info("Starting unit %s (timeout=%ds, low_mem<%d bytes for %.0fs)", unit_tag, timeout_seconds, low_mem_bytes, low_mem_duration)

    with worker_log.open("w", encoding="utf-8") as log_handle:
        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).resolve().parent),
        )
        while True:
            ret = proc.poll()
            now = time.perf_counter()
            elapsed = now - t0

            avail = psutil.virtual_memory().available
            if avail < min_avail_ram:
                min_avail_ram = avail

            rss = _get_tree_rss_bytes(proc.pid)
            if rss > peak_rss:
                peak_rss = rss

            if ret is not None:
                break

            # Sustained low-memory guard: abort only if available RAM < 0.5 GiB for 60s continuously
            if avail < low_mem_bytes:
                if low_mem_started is None:
                    low_mem_started = now
                    LOGGER.warning(
                        "Unit %s low available RAM detected: %.2f MiB; monitoring for %.0fs",
                        unit_tag,
                        avail / (1024 ** 2),
                        low_mem_duration,
                    )
                elif (now - low_mem_started) >= low_mem_duration:
                    abort_status = "aborted_low_memory"
                    abort_reason = (
                        f"Available RAM remained below {low_mem_bytes} bytes for "
                        f"{now - low_mem_started:.1f}s (min_available_ram_bytes={min_avail_ram})."
                    )
                    LOGGER.error("Unit %s %s: %s", unit_tag, abort_status, abort_reason)
                    _kill_process_tree(proc.pid)
                    break
            else:
                low_mem_started = None

            if elapsed >= timeout_seconds:
                abort_status = "timeout"
                abort_reason = f"Exceeded per-run timeout of {timeout_seconds}s (elapsed={elapsed:.1f}s)."
                LOGGER.error("Unit %s %s: %s", unit_tag, abort_status, abort_reason)
                _kill_process_tree(proc.pid)
                break

            time.sleep(1.0)

    wall_seconds = round(time.perf_counter() - t0, 4)
    finished_iso = datetime.datetime.now().isoformat()

    if abort_status is not None:
        record = {
            "status": abort_status,
            "reason": abort_reason,
            "suite": unit["suite"],
            "fraction": float(unit["fraction"]),
            "master": unit["master"],
            "cores": int(unit["cores"]),
            "repeat": int(unit["repeat"]),
            "input_format": input_format,
            "driver_memory": driver_memory,
            "shuffle_partitions": int(shuffle_partitions),
            "rows": None,
            "nodes": None,
            "edges": None,
            "seconds": {"total": wall_seconds},
            "peak_rss_bytes": int(peak_rss),
            "min_available_ram_bytes": int(min_avail_ram),
            "shuffle_read_bytes": None,
            "shuffle_write_bytes": None,
            "shuffle_bytes": None,
            "wall_seconds": wall_seconds,
            "started_at": started_iso,
            "finished_at": finished_iso,
            "worker_log": worker_log.name,
        }
    elif proc.returncode == 0 and worker_out.exists():
        record = json.loads(worker_out.read_text(encoding="utf-8"))
        record["peak_rss_bytes"] = int(max(peak_rss, int(record.get("peak_rss_bytes") or 0)))
        w_min = int(record.get("min_available_ram_bytes") or min_avail_ram)
        record["min_available_ram_bytes"] = int(min(min_avail_ram, w_min))
        record["wall_seconds"] = wall_seconds
        record["started_at"] = started_iso
        record["finished_at"] = finished_iso
        record["worker_log"] = worker_log.name
        try:
            worker_out.unlink()
        except OSError:
            pass
    else:
        tail_lines = ""
        try:
            lines = worker_log.read_text(encoding="utf-8", errors="replace").splitlines()
            tail_lines = "\n".join(lines[-20:])
        except Exception:
            pass
        record = {
            "status": "error",
            "returncode": proc.returncode,
            "error_tail": tail_lines,
            "suite": unit["suite"],
            "fraction": float(unit["fraction"]),
            "master": unit["master"],
            "cores": int(unit["cores"]),
            "repeat": int(unit["repeat"]),
            "input_format": input_format,
            "driver_memory": driver_memory,
            "shuffle_partitions": int(shuffle_partitions),
            "rows": None,
            "nodes": None,
            "edges": None,
            "seconds": {"total": wall_seconds},
            "peak_rss_bytes": int(peak_rss),
            "min_available_ram_bytes": int(min_avail_ram),
            "shuffle_read_bytes": None,
            "shuffle_write_bytes": None,
            "shuffle_bytes": None,
            "wall_seconds": wall_seconds,
            "started_at": started_iso,
            "finished_at": finished_iso,
            "worker_log": worker_log.name,
        }

    LOGGER.info(
        "Finished unit %s: status=%s wall=%.2fs peak_rss=%.1fMiB min_avail_ram=%.1fMiB",
        unit_tag,
        record["status"],
        wall_seconds,
        record["peak_rss_bytes"] / (1024 ** 2),
        record["min_available_ram_bytes"] / (1024 ** 2),
    )
    return record


def run_orchestrator(args: argparse.Namespace) -> None:
    """Execute the full scaling benchmark plan, skipping already-completed runs."""
    run_dir: Path = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(run_dir / "orchestrator.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)
    jsonl_path = run_dir / "scaling_runs.jsonl"

    if args.test_tiny:
        plan = [
            {
                "suite": "smoke_test",
                "fraction": 0.01,
                "master": "local[4]",
                "cores": 4,
                "repeat": 1,
            }
        ]
    else:
        plan = build_default_plan()

    run_info_path = run_dir / "run_info.json"
    if not run_info_path.exists():
        write_run_info(
            run_info_path,
            dataset_path=args.data if (args.data and args.data.is_file()) else None,
            config={
                "input_format": args.input_format,
                "driver_memory": args.driver_memory,
                "shuffle_partitions": args.shuffle_partitions,
                "timeout_seconds": args.timeout_seconds,
                "low_mem_bytes": args.low_mem_bytes,
                "low_mem_duration_seconds": args.low_mem_duration,
                "planned_runs": len(plan),
                "test_tiny": bool(args.test_tiny),
            },
        )

    completed_keys = _load_completed_keys(jsonl_path)
    LOGGER.info(
        "Scaling orchestrator starting in %s: %d/%d units already recorded.",
        run_dir,
        len(completed_keys),
        len(plan),
    )

    for idx, unit in enumerate(plan, start=1):
        key = _run_key(unit)
        if key in completed_keys:
            LOGGER.info("[%d/%d] Skipping already recorded unit: %s", idx, len(plan), key)
            continue

        LOGGER.info("[%d/%d] Launching unit: %s", idx, len(plan), key)
        rec = execute_single_unit(
            unit=unit,
            run_dir=run_dir,
            input_format=args.input_format,
            data_path=args.data,
            driver_memory=args.driver_memory,
            shuffle_partitions=args.shuffle_partitions,
            timeout_seconds=args.timeout_seconds,
            low_mem_bytes=args.low_mem_bytes,
            low_mem_duration=args.low_mem_duration,
        )
        _append_jsonl(jsonl_path, rec)
        completed_keys.add(key)

        # Brief cooldown between fresh JVM runs so OS reclaims memory cleanly
        if idx < len(plan):
            time.sleep(5.0)

    LOGGER.info("All %d planned scaling units completed or recorded in %s", len(plan), jsonl_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PaySim Spark scaling benchmark orchestrator.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Output run directory (resumable).")
    parser.add_argument(
        "--input-format",
        type=str,
        choices=["csv", "parquet"],
        default="parquet",
        help="Input storage format (default: parquet).",
    )
    parser.add_argument("--data", type=Path, default=None, help="Optional explicit input path.")
    parser.add_argument("--driver-memory", type=str, default=DEFAULT_DRIVER_MEMORY)
    parser.add_argument("--shuffle-partitions", type=int, default=DEFAULT_SHUFFLE_PARTITIONS)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--low-mem-bytes", type=int, default=DEFAULT_LOW_MEM_BYTES)
    parser.add_argument("--low-mem-duration", type=float, default=DEFAULT_LOW_MEM_DURATION_SECONDS)
    parser.add_argument(
        "--test-tiny",
        action="store_true",
        help="Run single smoke test: fraction 0.01, repeat 1, local[4].",
    )

    # Internal worker arguments
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--suite", type=str, default="data_scaling", help=argparse.SUPPRESS)
    parser.add_argument("--fraction", type=float, default=0.10, help=argparse.SUPPRESS)
    parser.add_argument("--master", type=str, default="local[*]", help=argparse.SUPPRESS)
    parser.add_argument("--cores", type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument("--repeat", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    if args.worker:
        run_worker(args)
        return

    if args.run_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        suffix = "c3-scaling-smoke" if args.test_tiny else "c3-scaling"
        args.run_dir = Path("runs") / f"{ts}_{suffix}"
    run_orchestrator(args)


if __name__ == "__main__":
    main()
