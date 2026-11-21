"""Resumable benchmark comparing pandas/NetworkX against Spark-native graph algorithms.

Sizes: 100k, 250k, 500k, 1M, 2M, 4M rows (uniform random sample, seed=42, written once
to Parquet in the run folder).
Tasks: (a) degree counts per account, (b) connected components.
Guards: 10-minute per-unit timeout, sustained available RAM < 0.5 GiB for 60s, and
process-tree RSS > 9 GiB. Stops increasing size for a (tool, task) once it reports
timeout or oom_guard.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from data_ingestion import DATA_DIR
from provenance import make_readonly, write_card, write_manifest
from run_info import write_run_info

LOGGER = logging.getLogger("fraud_analytics.pandas_vs_spark")

DEFAULT_SIZES = [100_000, 250_000, 500_000, 1_000_000, 2_000_000, 4_000_000]
DEFAULT_TIMEOUT_SECONDS = 10 * 60  # 10 minutes
DEFAULT_LOW_MEM_BYTES = int(0.5 * (1024 ** 3))  # 0.5 GiB
DEFAULT_LOW_MEM_DURATION_SECONDS = 60.0
DEFAULT_MAX_RSS_BYTES = int(9 * (1024 ** 3))  # 9 GiB
DEFAULT_DRIVER_MEMORY = "6g"
DEFAULT_SHUFFLE_PARTITIONS = 48

CAVEATS = [
    "Single Windows 11 laptop (15.37 GiB shared RAM, Intel Core Ultra 7 155H); Spark runs in local[*] mode with default 6g driver memory and 48 shuffle partitions.",
    "Load time (reading Parquet into memory) is reported separately from compute time (graph construction + algorithm execution); Spark JVM startup time is also recorded separately.",
    "Uniform random transaction sampling (seed=42) changes graph density across sizes; nodes and edges are recorded for every completed run.",
    "Each tool/task/size/repeat runs in a fresh Python subprocess with a 10-minute timeout, a 9 GiB process-tree RSS guard, and a sustained 60-second < 0.5 GiB available-RAM guard.",
    "Size escalation stops for a (tool, task) once a size records timeout or oom_guard; all non-ok runs are retained.",
]


def _get_tree_rss_bytes(pid: int) -> int:
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


def prepare_samples(
    run_dir: Path,
    sizes: List[int],
    source_parquet: Path,
    seed: int = 42,
) -> Dict[int, Path]:
    """Write uniform random Parquet samples (seed=42) once into run_dir/samples."""
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[int, Path] = {}
    missing_sizes: List[int] = []
    for n in sizes:
        p = samples_dir / f"rows_{n}.parquet"
        paths[n] = p
        if not p.exists():
            missing_sizes.append(n)

    if not missing_sizes:
        LOGGER.info("All %d sample Parquet files already exist in %s", len(sizes), samples_dir)
        return paths

    LOGGER.info("Generating %d uniform sample Parquet files from %s (seed=%d)", len(missing_sizes), source_parquet, seed)
    table = pq.read_table(
        str(source_parquet),
        columns=["step", "type", "amount", "nameOrig", "nameDest", "isFraud"],
    )
    total_rows = table.num_rows
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total_rows)

    for n in sorted(missing_sizes):
        if n > total_rows:
            raise ValueError(f"Requested sample size {n} exceeds dataset rows {total_rows}")
        idx = pa.array(perm[:n])
        sub = table.take(idx)
        out_path = paths[n]
        pq.write_table(sub, str(out_path))
        LOGGER.info("Wrote sample %s (%s rows)", out_path.name, f"{sub.num_rows:,}")

    return paths


def _run_pandas_worker(sample_path: Path, task: str) -> Dict[str, Any]:
    import networkx as nx

    t_load0 = time.perf_counter()
    df = pd.read_parquet(sample_path, columns=["nameOrig", "nameDest"])
    load_sec = time.perf_counter() - t_load0

    rows = int(len(df))
    t_comp0 = time.perf_counter()
    if task == "degree":
        out_deg = df.groupby("nameOrig", sort=False).size()
        in_deg = df.groupby("nameDest", sort=False).size()
        all_acc = pd.Index(out_deg.index).union(in_deg.index)
        deg_df = pd.DataFrame(
            {
                "in_degree": in_deg.reindex(all_acc, fill_value=0).to_numpy(dtype=np.float64),
                "out_degree": out_deg.reindex(all_acc, fill_value=0).to_numpy(dtype=np.float64),
            },
            index=all_acc,
        )
        nodes = int(len(deg_df))
        edges = rows
        extra = {"sum_in_degree": float(deg_df["in_degree"].sum()), "sum_out_degree": float(deg_df["out_degree"].sum())}
    elif task == "components":
        g = nx.Graph()
        g.add_edges_from(zip(df["nameOrig"].tolist(), df["nameDest"].tolist()))
        comps = list(nx.connected_components(g))
        nodes = int(g.number_of_nodes())
        edges = int(g.number_of_edges())
        max_comp = max((len(c) for c in comps), default=0)
        extra = {"component_count": len(comps), "max_component_size": int(max_comp)}
    else:
        raise ValueError(f"Unsupported task: {task}")
    comp_sec = time.perf_counter() - t_comp0

    return {
        "rows": rows,
        "nodes": nodes,
        "edges": edges,
        "spark_startup_seconds": 0.0,
        "load_seconds": round(load_sec, 4),
        "compute_seconds": round(comp_sec, 4),
        "seconds": round(load_sec + comp_sec, 4),
        "extra": extra,
    }


def _run_spark_worker(
    sample_path: Path,
    task: str,
    driver_memory: str,
    shuffle_partitions: int,
) -> Dict[str, Any]:
    from pyspark.sql import functions as F
    from feature_engineering import compute_connected_components, compute_degrees
    from graph_construction import build_transaction_graph
    from spark_session import create_spark_session

    t_start0 = time.perf_counter()
    spark = create_spark_session(
        app_name=f"PandasVsSpark_{task}_{sample_path.stem}",
        master="local[*]",
        driver_memory=driver_memory,
        shuffle_partitions=shuffle_partitions,
    )
    startup_sec = time.perf_counter() - t_start0

    try:
        t_load0 = time.perf_counter()
        df = spark.read.parquet(str(sample_path)).select("nameOrig", "nameDest", "amount", "type", "step", "isFraud").cache()
        rows = int(df.count())
        load_sec = time.perf_counter() - t_load0

        t_comp0 = time.perf_counter()
        graph = build_transaction_graph(df)
        vertices = graph.vertices.select("id").distinct().cache()
        edges_df = graph.edges.select("src", "dst").cache()
        nodes = int(vertices.count())
        edges = int(edges_df.count())

        if task == "degree":
            deg_df, info = compute_degrees(vertices, edges_df)
            agg = deg_df.agg(F.sum("in_degree").alias("sin"), F.sum("out_degree").alias("sout")).collect()[0]
            extra = {
                "engine": info["engine"],
                "sum_in_degree": float(agg["sin"] or 0.0),
                "sum_out_degree": float(agg["sout"] or 0.0),
            }
        elif task == "components":
            cc_df, info = compute_connected_components(vertices, edges_df, max_iter=30)
            cc_df = cc_df.cache()
            comp_count = int(cc_df.select("component_id").distinct().count())
            max_sz_row = cc_df.agg(F.max("component_size").alias("mx")).collect()[0]
            extra = {
                "engine": info["engine"],
                "iterations": info["iterations"],
                "converged": info["converged"],
                "component_count": comp_count,
                "max_component_size": int(max_sz_row["mx"] or 0),
            }
        else:
            raise ValueError(f"Unsupported task: {task}")
        comp_sec = time.perf_counter() - t_comp0

        return {
            "rows": rows,
            "nodes": nodes,
            "edges": edges,
            "spark_startup_seconds": round(startup_sec, 4),
            "load_seconds": round(load_sec, 4),
            "compute_seconds": round(comp_sec, 4),
            "seconds": round(load_sec + comp_sec, 4),
            "extra": extra,
        }
    finally:
        spark.stop()


def run_worker(args: argparse.Namespace) -> None:
    if args.tool == "pandas_networkx":
        res = _run_pandas_worker(args.sample_path, args.task)
    elif args.tool == "spark":
        res = _run_spark_worker(
            args.sample_path,
            args.task,
            driver_memory=args.driver_memory,
            shuffle_partitions=args.shuffle_partitions,
        )
    else:
        raise ValueError(f"Unknown tool: {args.tool}")

    res.update(
        {
            "status": "ok",
            "tool": args.tool,
            "task": args.task,
            "target_rows": int(args.rows),
            "repeat": int(args.repeat),
            "peak_rss_bytes": _get_tree_rss_bytes(os.getpid()),
            "min_available_ram_bytes": int(psutil.virtual_memory().available),
        }
    )
    Path(args.worker_output).write_text(json.dumps(res, indent=2), encoding="utf-8")


def _unit_key(rec: Dict[str, Any]) -> Tuple[str, str, int, int]:
    return (str(rec["tool"]), str(rec["task"]), int(rec["target_rows"]), int(rec["repeat"]))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _append_jsonl(path: Path, rec: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(rec) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def execute_unit(
    tool: str,
    task: str,
    rows: int,
    repeat: int,
    sample_path: Path,
    run_dir: Path,
    driver_memory: str,
    shuffle_partitions: int,
    timeout_seconds: int,
    low_mem_bytes: int,
    low_mem_duration: float,
    max_rss_bytes: int,
) -> Dict[str, Any]:
    tag = f"{tool}_{task}_{rows}_r{repeat}"
    worker_out = run_dir / f"_worker_{tag}.json"
    worker_log = run_dir / f"_worker_{tag}.log"
    if worker_out.exists():
        worker_out.unlink()

    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        "--tool",
        tool,
        "--task",
        task,
        "--rows",
        str(rows),
        "--repeat",
        str(repeat),
        "--sample-path",
        str(sample_path),
        "--driver-memory",
        driver_memory,
        "--shuffle-partitions",
        str(shuffle_partitions),
        "--worker-output",
        str(worker_out),
    ]

    started_iso = datetime.datetime.now().isoformat()
    t0 = time.perf_counter()
    min_avail_ram = psutil.virtual_memory().available
    peak_rss = 0
    low_mem_started: Optional[float] = None
    abort_status: Optional[str] = None
    abort_reason: Optional[str] = None

    LOGGER.info("Starting %s (timeout=%ds, max_rss=%dB, low_mem<%dB for %.0fs)", tag, timeout_seconds, max_rss_bytes, low_mem_bytes, low_mem_duration)

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

            if rss > max_rss_bytes:
                abort_status = "oom_guard"
                abort_reason = f"Process tree RSS {rss} bytes exceeded 9 GiB cap ({max_rss_bytes} bytes)."
                LOGGER.error("%s %s: %s", tag, abort_status, abort_reason)
                _kill_process_tree(proc.pid)
                break

            if avail < low_mem_bytes:
                if low_mem_started is None:
                    low_mem_started = now
                    LOGGER.warning("%s low available RAM: %.1f MiB; monitoring 60s", tag, avail / (1024 ** 2))
                elif (now - low_mem_started) >= low_mem_duration:
                    abort_status = "oom_guard"
                    abort_reason = f"Available RAM remained below {low_mem_bytes} bytes for {now - low_mem_started:.1f}s."
                    LOGGER.error("%s %s: %s", tag, abort_status, abort_reason)
                    _kill_process_tree(proc.pid)
                    break
            else:
                low_mem_started = None

            if elapsed >= timeout_seconds:
                abort_status = "timeout"
                abort_reason = f"Exceeded {timeout_seconds}s timeout (elapsed={elapsed:.1f}s)."
                LOGGER.error("%s %s: %s", tag, abort_status, abort_reason)
                _kill_process_tree(proc.pid)
                break

            time.sleep(0.5)

    wall_seconds = round(time.perf_counter() - t0, 4)
    finished_iso = datetime.datetime.now().isoformat()

    if abort_status is not None:
        rec = {
            "status": abort_status,
            "reason": abort_reason,
            "tool": tool,
            "task": task,
            "target_rows": rows,
            "rows": rows,
            "nodes": None,
            "edges": None,
            "repeat": repeat,
            "spark_startup_seconds": None,
            "load_seconds": None,
            "compute_seconds": None,
            "seconds": wall_seconds,
            "wall_seconds": wall_seconds,
            "peak_rss_bytes": int(peak_rss),
            "min_available_ram_bytes": int(min_avail_ram),
            "started_at": started_iso,
            "finished_at": finished_iso,
            "worker_log": worker_log.name,
        }
    elif proc.returncode == 0 and worker_out.exists():
        rec = json.loads(worker_out.read_text(encoding="utf-8"))
        rec["peak_rss_bytes"] = int(max(peak_rss, int(rec.get("peak_rss_bytes") or 0)))
        rec["min_available_ram_bytes"] = int(min(min_avail_ram, int(rec.get("min_available_ram_bytes") or min_avail_ram)))
        rec["wall_seconds"] = wall_seconds
        rec["started_at"] = started_iso
        rec["finished_at"] = finished_iso
        rec["worker_log"] = worker_log.name
        try:
            worker_out.unlink()
        except OSError:
            pass
    else:
        tail = ""
        try:
            tail = "\n".join(worker_log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:])
        except Exception:
            pass
        is_oom = ("MemoryError" in tail) or ("OutOfMemoryError" in tail)
        rec = {
            "status": "oom_guard" if is_oom else "error",
            "returncode": proc.returncode,
            "error_tail": tail,
            "tool": tool,
            "task": task,
            "target_rows": rows,
            "rows": rows,
            "nodes": None,
            "edges": None,
            "repeat": repeat,
            "spark_startup_seconds": None,
            "load_seconds": None,
            "compute_seconds": None,
            "seconds": wall_seconds,
            "wall_seconds": wall_seconds,
            "peak_rss_bytes": int(peak_rss),
            "min_available_ram_bytes": int(min_avail_ram),
            "started_at": started_iso,
            "finished_at": finished_iso,
            "worker_log": worker_log.name,
        }

    LOGGER.info(
        "Finished %s: status=%s load=%s compute=%s wall=%.2fs peak_rss=%.1fMiB min_ram=%.1fMiB",
        tag,
        rec["status"],
        rec.get("load_seconds"),
        rec.get("compute_seconds"),
        wall_seconds,
        rec["peak_rss_bytes"] / (1024 ** 2),
        rec["min_available_ram_bytes"] / (1024 ** 2),
    )
    return rec


def aggregate_pandas_vs_spark(
    run_dir: Path,
    copy_to_outputs: bool = True,
    finalize_readonly: bool = False,
) -> Dict[str, Any]:
    jsonl_path = run_dir / "pandas_vs_spark.jsonl"
    records = _load_jsonl(jsonl_path)

    grouped: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    non_ok: List[Dict[str, Any]] = []
    total_wall = 0.0

    for rec in records:
        total_wall += float(rec.get("wall_seconds") or 0.0)
        if rec.get("status") != "ok":
            non_ok.append(rec)
        grouped[(str(rec["tool"]), str(rec["task"]), int(rec["target_rows"]))].append(rec)

    summary_rows: List[Dict[str, Any]] = []
    for (tool, task, rows) in sorted(grouped.keys(), key=lambda x: (x[1], x[2], x[0])):
        grp = sorted(grouped[(tool, task, rows)], key=lambda r: int(r["repeat"]))
        ok_grp = [r for r in grp if r.get("status") == "ok"]
        load_vals = [float(r["load_seconds"]) for r in ok_grp if r.get("load_seconds") is not None]
        comp_vals = [float(r["compute_seconds"]) for r in ok_grp if r.get("compute_seconds") is not None]
        tot_vals = [float(r["seconds"]) for r in ok_grp if r.get("seconds") is not None]
        startup_vals = [float(r["spark_startup_seconds"]) for r in ok_grp if r.get("spark_startup_seconds") is not None]
        rss_vals = [int(r["peak_rss_bytes"]) for r in grp if r.get("peak_rss_bytes") is not None]
        ram_vals = [int(r["min_available_ram_bytes"]) for r in grp if r.get("min_available_ram_bytes") is not None]

        summary_rows.append(
            {
                "tool": tool,
                "task": task,
                "rows": rows,
                "ok_count": len(ok_grp),
                "total_count": len(grp),
                "statuses": [r.get("status") for r in grp],
                "mean_load_seconds": round(statistics.mean(load_vals), 4) if load_vals else None,
                "individual_load_seconds": [round(v, 4) for v in load_vals],
                "mean_compute_seconds": round(statistics.mean(comp_vals), 4) if comp_vals else None,
                "individual_compute_seconds": [round(v, 4) for v in comp_vals],
                "mean_total_seconds": round(statistics.mean(tot_vals), 4) if tot_vals else None,
                "individual_total_seconds": [round(v, 4) for v in tot_vals],
                "mean_spark_startup_seconds": round(statistics.mean(startup_vals), 4) if startup_vals else None,
                "mean_peak_rss_bytes": int(round(statistics.mean(rss_vals))) if rss_vals else None,
                "max_peak_rss_bytes": max(rss_vals) if rss_vals else None,
                "min_available_ram_bytes": min(ram_vals) if ram_vals else None,
                "nodes": ok_grp[0].get("nodes") if ok_grp else None,
                "edges": ok_grp[0].get("edges") if ok_grp else None,
            }
        )

    payload = {
        "benchmark": "pandas_networkx_vs_spark",
        "generated_at": datetime.datetime.now().isoformat(),
        "total_runs_recorded": len(records),
        "ok_runs_count": len(records) - len(non_ok),
        "non_ok_runs_count": len(non_ok),
        "total_benchmark_wall_seconds": round(total_wall, 4),
        "summary": summary_rows,
        "non_ok_runs": non_ok,
        "runs": records,
        "caveats": CAVEATS,
    }

    out_json = run_dir / "pandas_vs_spark.json"
    out_csv = run_dir / "pandas_vs_spark.csv"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fieldnames = [
        "tool",
        "task",
        "target_rows",
        "repeat",
        "status",
        "nodes",
        "edges",
        "spark_startup_seconds",
        "load_seconds",
        "compute_seconds",
        "seconds",
        "wall_seconds",
        "peak_rss_bytes",
        "min_available_ram_bytes",
        "reason",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k) for k in fieldnames})

    key_nums = [
        ("Total Runs Recorded", "pandas_vs_spark.json", "total_runs_recorded"),
        ("OK Runs Count", "pandas_vs_spark.json", "ok_runs_count"),
        ("Non-OK Runs Count", "pandas_vs_spark.json", "non_ok_runs_count"),
        ("Total Benchmark Wall Seconds", "pandas_vs_spark.json", "total_benchmark_wall_seconds"),
    ]
    for idx, row in enumerate(summary_rows):
        key_nums.append(
            (
                f"{row['tool']} {row['task']} {row['rows']} compute mean (s)",
                "pandas_vs_spark.json",
                f"summary.{idx}.mean_compute_seconds",
            )
        )

    write_card(
        run_dir=run_dir,
        run_id=run_dir.name,
        purpose="Compare pandas/NetworkX against Spark-native degree and connected-components computations from 100,000 to 4,000,000 rows (2 repeats, 10-minute timeout, 9 GiB RSS guard, 60s < 0.5 GiB available RAM guard).",
        command=f"python pandas_vs_spark.py --run-dir {run_dir}",
        inputs=[("Uniform Parquet samples (seed=42)", str(run_dir / "samples"))],
        outputs=[
            ("pandas_vs_spark.jsonl", "Append-only per-run records"),
            ("pandas_vs_spark.json", "Aggregated summary and individual runs"),
            ("pandas_vs_spark.csv", "Tabular CSV export"),
        ],
        key_numbers=key_nums,
        caveats=CAVEATS,
        not_shown=[
            "Does not compare multi-node cluster Spark scaling against single-node pandas.",
            "Does not test PageRank, triangles, or LPA in NetworkX.",
        ],
    )
    write_manifest(run_dir)
    if finalize_readonly:
        make_readonly(run_dir)

    if copy_to_outputs:
        import stat
        def _copy_to(src: Path, dst: Path) -> None:
            if dst.exists():
                dst.chmod(stat.S_IWRITE | stat.S_IREAD)
            shutil.copyfile(src, dst)
            dst.chmod(stat.S_IWRITE | stat.S_IREAD)

        outputs_dir = Path("outputs")
        outputs_dir.mkdir(parents=True, exist_ok=True)
        _copy_to(out_json, outputs_dir / "pandas_vs_spark.json")
        _copy_to(out_csv, outputs_dir / "pandas_vs_spark.csv")
        snap_dir = Path("results_snapshot")
        if snap_dir.exists():
            _copy_to(out_json, snap_dir / "pandas_vs_spark.json")
            _copy_to(out_csv, snap_dir / "pandas_vs_spark.csv")

    return payload


def run_orchestrator(args: argparse.Namespace) -> None:
    run_dir: Path = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(run_dir / "orchestrator.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)

    source_parquet = args.source_parquet or (DATA_DIR / "parquet" / "paysim_unpartitioned")
    if args.extra_full_pandas:
        full_rows = 6_362_620
        sizes = [full_rows]
        tools = ["pandas_networkx"]
        tasks = ["degree", "components"]
        sample_paths = {full_rows: source_parquet}
    else:
        sizes = [100_000] if args.test_tiny else DEFAULT_SIZES
        tools = ["pandas_networkx", "spark"]
        tasks = ["degree", "components"]
        sample_paths = prepare_samples(run_dir, sizes, source_parquet, seed=42)

    repeats = 1 if args.test_tiny else args.repeats

    run_info_path = run_dir / "run_info.json"
    if not run_info_path.exists():
        write_run_info(
            run_info_path,
            config={
                "sizes": sizes,
                "repeats": repeats,
                "source_parquet": str(source_parquet),
                "driver_memory": args.driver_memory,
                "shuffle_partitions": args.shuffle_partitions,
                "timeout_seconds": args.timeout_seconds,
                "low_mem_bytes": args.low_mem_bytes,
                "low_mem_duration_seconds": args.low_mem_duration,
                "max_rss_bytes": args.max_rss_bytes,
                "test_tiny": bool(args.test_tiny),
                "extra_full_pandas": bool(args.extra_full_pandas),
            },
        )

    jsonl_path = run_dir / "pandas_vs_spark.jsonl"
    existing = _load_jsonl(jsonl_path)
    completed: Dict[Tuple[str, str, int, int], Dict[str, Any]] = {_unit_key(r): r for r in existing}

    for tool in tools:
        for task in tasks:
            stopped_limit = False
            for rows in sizes:
                if stopped_limit:
                    LOGGER.info("Skipping %s %s rows=%d because smaller size hit timeout/oom_guard", tool, task, rows)
                    break

                size_hit_limit = False
                for rep in range(1, repeats + 1):
                    key = (tool, task, rows, rep)
                    if key in completed:
                        rec = completed[key]
                        LOGGER.info("Skipping already recorded unit %s (status=%s)", key, rec.get("status"))
                        if rec.get("status") in ("timeout", "oom_guard"):
                            size_hit_limit = True
                            break
                        continue

                    rec = execute_unit(
                        tool=tool,
                        task=task,
                        rows=rows,
                        repeat=rep,
                        sample_path=sample_paths[rows],
                        run_dir=run_dir,
                        driver_memory=args.driver_memory,
                        shuffle_partitions=args.shuffle_partitions,
                        timeout_seconds=args.timeout_seconds,
                        low_mem_bytes=args.low_mem_bytes,
                        low_mem_duration=args.low_mem_duration,
                        max_rss_bytes=args.max_rss_bytes,
                    )
                    _append_jsonl(jsonl_path, rec)
                    completed[key] = rec
                    time.sleep(2.0)

                    if rec.get("status") in ("timeout", "oom_guard"):
                        size_hit_limit = True
                        break

                if size_hit_limit:
                    stopped_limit = True

    summary = aggregate_pandas_vs_spark(
        run_dir,
        copy_to_outputs=not args.test_tiny,
        finalize_readonly=not args.test_tiny,
    )
    LOGGER.info(
        "pandas_vs_spark finished: total=%d ok=%d non_ok=%d wall=%.2fs",
        summary["total_runs_recorded"],
        summary["ok_runs_count"],
        summary["non_ok_runs_count"],
        summary["total_benchmark_wall_seconds"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare pandas/NetworkX against Spark.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--source-parquet", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--driver-memory", type=str, default=DEFAULT_DRIVER_MEMORY)
    parser.add_argument("--shuffle-partitions", type=int, default=DEFAULT_SHUFFLE_PARTITIONS)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--low-mem-bytes", type=int, default=DEFAULT_LOW_MEM_BYTES)
    parser.add_argument("--low-mem-duration", type=float, default=DEFAULT_LOW_MEM_DURATION_SECONDS)
    parser.add_argument("--max-rss-bytes", type=int, default=DEFAULT_MAX_RSS_BYTES)
    parser.add_argument("--test-tiny", action="store_true", help="Smoke test at 100k rows, 1 repeat.")
    parser.add_argument(
        "--extra-full-pandas",
        action="store_true",
        help="Run only pandas_networkx degree and components on the full 6,362,620-row Parquet dataset and append to jsonl.",
    )

    # Internal worker arguments
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--tool", type=str, default="pandas_networkx", help=argparse.SUPPRESS)
    parser.add_argument("--task", type=str, default="degree", help=argparse.SUPPRESS)
    parser.add_argument("--rows", type=int, default=100_000, help=argparse.SUPPRESS)
    parser.add_argument("--repeat", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--sample-path", type=Path, default=None, help=argparse.SUPPRESS)
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
        suffix = "c3-pvs-smoke" if args.test_tiny else "c3-pandas-vs-spark"
        args.run_dir = Path("runs") / f"{ts}_{suffix}"
    run_orchestrator(args)


if __name__ == "__main__":
    main()
