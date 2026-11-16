"""Smoke tests for Streamlit fraud analytics application using AppTest."""

import os
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

LOG_PATH = Path("logs") / "raw" / "20261116_1.10_app_smoke.txt"


def log_assertion(msg: str) -> None:
    """Log an assertion result to logs/raw/."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def test_app_loads_and_tabs_switch():
    """Verify app loads cleanly with 0 exceptions and all 6 tabs are present."""
    os.environ["FRAUD_OUTPUT_DIR"] = "outputs_test"
    at = AppTest.from_file("app.py", default_timeout=35)
    at.run()

    exc_count = len(at.exception)
    log_assertion(f"[TEST] test_app_loads_and_tabs_switch: exception_count={exc_count}")
    if at.exception:
        for ex in at.exception:
            log_assertion(f"  Exception: {ex.message}")
    assert exc_count == 0, f"App threw {exc_count} exception(s) on initial load"

    tab_count = len(at.tabs)
    log_assertion(f"[TEST] test_app_loads_and_tabs_switch: tabs_count={tab_count}")
    assert tab_count == 6, f"Expected 6 tabs, found {tab_count}"


def test_scorer_presets_and_distinct_probabilities():
    """Verify scorer presets, submit button interaction, and distinct output probabilities."""
    os.environ["FRAUD_OUTPUT_DIR"] = "outputs_test"
    at = AppTest.from_file("app.py", default_timeout=35)
    at.run()
    assert len(at.exception) == 0

    probs = []

    # Preset 0: Normal
    at.button[0].click().run()
    assert len(at.exception) == 0
    at.button[3].click().run()
    assert len(at.exception) == 0

    text_normal = ""
    if at.success:
        text_normal += at.success[-1].value
    if at.error:
        text_normal += at.error[-1].value
    m0 = re.search(r"Probability:\s*([0-9.]+)%", text_normal)
    assert m0 is not None, f"Could not find probability in output: {text_normal}"
    p0 = float(m0.group(1))
    probs.append(p0)
    log_assertion(f"[TEST] Scorer Normal Preset: probability={p0}%")

    # Preset 1: Suspicious
    at.button[1].click().run()
    assert len(at.exception) == 0
    at.button[3].click().run()
    assert len(at.exception) == 0

    text_susp = ""
    if at.error:
        text_susp += at.error[-1].value
    if at.success:
        text_susp += at.success[-1].value
    m1 = re.search(r"Probability:\s*([0-9.]+)%", text_susp)
    assert m1 is not None, f"Could not find probability in output: {text_susp}"
    p1 = float(m1.group(1))
    probs.append(p1)
    log_assertion(f"[TEST] Scorer Suspicious Preset: probability={p1}%")

    # Preset 2: Borderline
    at.button[2].click().run()
    assert len(at.exception) == 0
    at.button[3].click().run()
    assert len(at.exception) == 0

    text_bord = ""
    if at.success:
        text_bord += at.success[-1].value
    if at.error:
        text_bord += at.error[-1].value
    m2 = re.search(r"Probability:\s*([0-9.]+)%", text_bord)
    assert m2 is not None, f"Could not find probability in output: {text_bord}"
    p2 = float(m2.group(1))
    probs.append(p2)
    log_assertion(f"[TEST] Scorer Borderline Preset: probability={p2}%")

    distinct_count = len(set(probs))
    log_assertion(f"[TEST] Scorer Distinct Probabilities Count: {distinct_count}/3 ({probs})")
    assert distinct_count == 3, f"Expected 3 distinct probabilities, got {probs}"


def test_missing_data_graceful_handling():
    """Verify app handles completely missing outputs folder with informative banners and 0 exceptions."""
    os.environ["FRAUD_OUTPUT_DIR"] = "outputs_nonexistent_test_dir"
    at = AppTest.from_file("app.py", default_timeout=35)
    at.run()

    exc_count = len(at.exception)
    log_assertion(f"[TEST] test_missing_data_graceful_handling: exception_count={exc_count}")
    assert exc_count == 0, f"Missing data caused exception: {at.exception}"

    # Verify info banners appear
    info_count = len(at.info)
    log_assertion(f"[TEST] test_missing_data_graceful_handling: info_banners_count={info_count}")
    assert info_count > 0, "Expected informational guidance messages when output directory is missing"
