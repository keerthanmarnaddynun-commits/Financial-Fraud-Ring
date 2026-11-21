"""Unit tests for StageTimer and process-tree RSS memory sampling."""

import time
import numpy as np
import pytest
from stage_timer import StageTimer, get_process_tree_rss


def test_stage_timer_measures_time_and_memory():
    results = {}
    with StageTimer("test_stage", results, sampling_interval=0.05) as timer:
        time.sleep(0.15)

    assert timer.wall_seconds >= 0.10
    assert "timings_seconds" in results
    assert "test_stage" in results["timings_seconds"]
    assert results["timings_seconds"]["test_stage"] == timer.wall_seconds
    assert timer.peak_rss_bytes > 0
    assert timer.min_available_ram_bytes > 0


def test_stage_timer_detects_200mb_allocation():
    results = {}
    size_mb = 200
    num_elements = (size_mb * 1024 * 1024) // 8  # 200 MB of int64

    baseline = get_process_tree_rss()
    with StageTimer("memory_stage", results, sampling_interval=0.05) as timer:
        # Allocate 200MB array and touch every element so OS commits physical pages
        arr = np.ones((num_elements,), dtype=np.int64)
        arr += 1
        time.sleep(0.15)
        # Keep reference active
        total_sum = int(arr.sum())
        assert total_sum > 0

    del arr
    # Verify peak RSS detected the allocation above baseline
    rss_increase = timer.peak_rss_bytes - timer.baseline_rss_bytes
    assert timer.peak_rss_bytes > timer.baseline_rss_bytes
    # At least 150 MB of the 200 MB must be detected
    assert rss_increase >= 150 * 1024 * 1024, (
        f"Expected RSS increase >= 150MB, observed {rss_increase / (1024*1024):.2f}MB "
        f"(baseline: {timer.baseline_rss_bytes / 1e6:.1f}MB, peak: {timer.peak_rss_bytes / 1e6:.1f}MB)"
    )
