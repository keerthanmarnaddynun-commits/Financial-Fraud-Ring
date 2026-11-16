"""Execution stage timer with process-tree RSS memory sampling and Spark UI metric tracking."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Set
from urllib.error import URLError
from urllib.request import urlopen

import psutil

LOGGER = logging.getLogger("fraud_analytics.timer")


def get_process_tree_rss() -> int:
    """Calculate the total resident memory (RSS) of the current process and all child processes."""
    try:
        main_proc = psutil.Process(os.getpid())
        total_rss = main_proc.memory_info().rss
        for child in main_proc.children(recursive=True):
            try:
                total_rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total_rss
    except Exception:
        return 0


class StageTimer:
    """Context manager to measure wall-clock duration, memory footprint, and Spark shuffle metrics."""

    def __init__(
        self,
        name: str,
        results_dict: Optional[Dict[str, Any]] = None,
        spark: Optional[Any] = None,
        sampling_interval: float = 0.2,
    ) -> None:
        self.name = name
        self.results_dict = results_dict
        self.spark = spark
        self.sampling_interval = sampling_interval

        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.wall_seconds: float = 0.0

        self.baseline_rss_bytes: int = 0
        self.peak_rss_bytes: int = 0
        self.min_available_ram_bytes: int = 0

        self.shuffle_read_bytes_delta: int = 0
        self.shuffle_write_bytes_delta: int = 0
        self.stages_count_delta: int = 0

        self._running = False
        self._sampler_thread: Optional[threading.Thread] = None
        self._baseline_stage_ids: Set[int] = set()

    def _sample_memory_loop(self) -> None:
        while self._running:
            try:
                rss = get_process_tree_rss()
                if rss > self.peak_rss_bytes:
                    self.peak_rss_bytes = rss
                avail = psutil.virtual_memory().available
                if avail < self.min_available_ram_bytes:
                    self.min_available_ram_bytes = avail
            except Exception:
                pass
            time.sleep(self.sampling_interval)

    def _get_spark_stages(self) -> List[Dict[str, Any]]:
        if not self.spark:
            return []
        sc = getattr(self.spark, "sparkContext", None)
        if not sc or not sc.uiWebUrl or not sc.applicationId:
            return []
        url = f"{sc.uiWebUrl.rstrip('/')}/api/v1/applications/{sc.applicationId}/stages"
        try:
            with urlopen(url, timeout=3) as resp:
                return json.load(resp)
        except Exception:
            return []

    def __enter__(self) -> StageTimer:
        self.start_time = time.perf_counter()
        self.baseline_rss_bytes = get_process_tree_rss()
        self.peak_rss_bytes = self.baseline_rss_bytes
        self.min_available_ram_bytes = psutil.virtual_memory().available

        # Snapshot Spark completed stage IDs
        initial_stages = self._get_spark_stages()
        self._baseline_stage_ids = {
            s["stageId"]
            for s in initial_stages
            if isinstance(s, dict) and s.get("status") in ("COMPLETE", "FAILED")
        }

        # Start sampling thread
        self._running = True
        self._sampler_thread = threading.Thread(
            target=self._sample_memory_loop, daemon=True, name=f"StageTimer-{self.name}"
        )
        self._sampler_thread.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._running = False
        if self._sampler_thread:
            self._sampler_thread.join(timeout=1.0)

        # Final RSS sample
        final_rss = get_process_tree_rss()
        if final_rss > self.peak_rss_bytes:
            self.peak_rss_bytes = final_rss

        self.end_time = time.perf_counter()
        self.wall_seconds = self.end_time - self.start_time

        # Compute Spark deltas
        final_stages = self._get_spark_stages()
        new_stages = [
            s
            for s in final_stages
            if isinstance(s, dict)
            and s.get("stageId") not in self._baseline_stage_ids
            and s.get("status") in ("COMPLETE", "FAILED")
        ]
        self.stages_count_delta = len(new_stages)
        self.shuffle_read_bytes_delta = sum(
            int(s.get("shuffleReadBytes", 0)) for s in new_stages
        )
        self.shuffle_write_bytes_delta = sum(
            int(s.get("shuffleWriteBytes", 0)) for s in new_stages
        )

        record = {
            "wall_seconds": self.wall_seconds,
            "peak_rss_bytes": self.peak_rss_bytes,
            "baseline_rss_bytes": self.baseline_rss_bytes,
            "min_available_ram_bytes": self.min_available_ram_bytes,
            "stages_count_delta": self.stages_count_delta,
            "shuffle_read_bytes_delta": self.shuffle_read_bytes_delta,
            "shuffle_write_bytes_delta": self.shuffle_write_bytes_delta,
        }

        if self.results_dict is not None:
            if "timings_seconds" not in self.results_dict:
                self.results_dict["timings_seconds"] = {}
            self.results_dict["timings_seconds"][self.name] = self.wall_seconds

            if "stages" not in self.results_dict:
                self.results_dict["stages"] = {}
            self.results_dict["stages"][self.name] = record
