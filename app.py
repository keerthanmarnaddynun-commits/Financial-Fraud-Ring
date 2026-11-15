"""Truthful dashboard for the PaySim fraud analytics project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

PAGE_TITLE = "Financial Fraud Analytics"
OUTPUT_DIR = Path("outputs")
MODEL_NAMES_PATH = Path("model_names.json")


@st.cache_data
def read_json(path: str) -> Optional[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return None
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None


@st.cache_data
def load_model_names() -> Dict[str, str]:
    defaults = {
        "m0_rule": "M0 Rule baseline",
        "m1_tabular": "M1 Tabular",
        "m2_graph": "M2 Graph + structural",
        "m3_community_mule": "M3 Community + mule",
    }
    if MODEL_NAMES_PATH.exists():
        try:
            with MODEL_NAMES_PATH.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                defaults.update(loaded)
        except (json.JSONDecodeError, OSError):
            pass
    return defaults


st.set_page_config(page_title=PAGE_TITLE, layout="wide")

st.markdown(
    """
    <style>
        .main {
            background: #071a17;
            color: #edf7f4;
        }
        .stApp {
            background: linear-gradient(180deg, #071a17 0%, #0d1d1b 100%);
        }
        .spark-card {
            background: rgba(17, 26, 24, 0.9);
            border: 1px solid rgba(121, 155, 133, 0.6);
            border-radius: 12px;
            padding: 18px 18px 14px 18px;
            min-height: 170px;
            box-shadow: inset 0 0 0 1px rgba(186, 255, 211, 0.06);
        }
        .spark-label {
            font-size: 12px;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            color: #87a39b;
            margin-bottom: 12px;
            font-weight: 600;
        }
        .spark-value {
            font-size: clamp(1.8rem, 2.3vw, 3rem);
            font-weight: 700;
            color: #edf7f4;
            line-height: 1.1;
        }
        .spark-subtext {
            color: #a6b8b3;
            font-size: 0.92rem;
            margin-top: 10px;
            line-height: 1.5;
        }
        .spark-bar {
            height: 3px;
            width: 100%;
            background: rgba(255,255,255,0.08);
            border-radius: 999px;
            margin-top: 12px;
            overflow: hidden;
        }
        .spark-bar > span {
            display: block;
            height: 100%;
            background: #d9f86f;
            border-radius: 999px;
        }
        .spark-title {
            color: #edf7f4;
            font-size: 2.0rem;
            font-weight: 700;
            margin: 0 0 0.2rem 0;
        }
        .spark-appid {
            color: #8dcab0;
            font-size: 0.82rem;
            font-weight: 500;
            font-family: "SFMono-Regular", Consolas, monospace;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_seconds(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0f}s" if value >= 1 else f"{value:.2f}s"


def format_bytes(value: Optional[int]) -> str:
    if value is None:
        return "n/a"
    units = ["B", "KB", "MB", "GB", "TB"]
    num = float(value)
    for unit in units:
        if num < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(num)} {unit}"
            return f"{num:.1f} {unit}"
        num /= 1024
    return "n/a"


@st.cache_data
def read_metrics() -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "run_info": read_json(str(OUTPUT_DIR / "run_info.json")),
        "evaluation": read_json(str(OUTPUT_DIR / "evaluation.json")),
        "scalability": read_json(str(OUTPUT_DIR / "scalability.json")),
    }
    return data


metrics = read_metrics()
run_info = metrics.get("run_info")
scalability = metrics.get("scalability") or {}

spark_application_id = scalability.get("spark_application_id") or "local[*]"
if spark_application_id and spark_application_id != "local[*]":
    app_label = f"local[*] | {spark_application_id}"
else:
    app_label = "local[*]"

st.markdown(f'<div style="display:flex; justify-content:space-between; align-items:end; margin-bottom: 15px;">'
            f'<div class="spark-title">Scalability & runtime</div>'
            f'<div class="spark-appid">{app_label}</div>'
            f'</div>', unsafe_allow_html=True)

rows = []
if scalability:
    timings = scalability.get("timings_seconds", {}) or {}
    read_bytes = scalability.get("shuffle_read_bytes")
    write_bytes = scalability.get("shuffle_write_bytes")
    total_shuffle = (read_bytes or 0) + (write_bytes or 0)
    read_pct = (read_bytes / total_shuffle * 100) if total_shuffle else 0
    write_pct = (write_bytes / total_shuffle * 100) if total_shuffle else 0
    rows = [
        {
            "label": "Graph feature engineering",
            "value": format_seconds(timings.get("graph_feature_engineering")),
            "subtitle": "Most expensive measured pipeline stage",
            "width": min(100, max(10, (timings.get("graph_feature_engineering") or 0) / max(timings.get("pipeline_total") or 1, 1) * 100)),
        },
        {
            "label": "Graph construction",
            "value": format_seconds(timings.get("graph_construction")),
            "subtitle": "Vertex and edge materialization",
            "width": min(100, max(10, (timings.get("graph_construction") or 0) / max(timings.get("pipeline_total") or 1, 1) * 100)),
        },
        {
            "label": "Executor memory",
            "value": format_bytes(scalability.get("executor_storage_memory_used_bytes") or scalability.get("executor_memory_bytes")),
            "subtitle": "Observed Spark UI memory usage",
            "width": 70,
        },
        {
            "label": "Shuffle read",
            "value": f"{read_pct:.1f}%",
            "subtitle": f"{format_bytes(read_bytes)} across Spark stages",
            "width": read_pct,
        },
        {
            "label": "Shuffle write",
            "value": f"{write_pct:.1f}%",
            "subtitle": f"{format_bytes(write_bytes)} written across Spark stages",
            "width": write_pct,
        },
        {
            "label": "Stages & training sample",
            "value": f"{scalability.get('stage_count', 'n/a')} stages / {scalability.get('sampled_training_rows', 'n/a')} rows",
            "subtitle": "Spark stage count and bounded model sample",
            "width": 75,
        },
    ]

cards = st.columns(3)
for idx, row in enumerate(rows):
    with cards[idx % 3]:
        st.markdown(
            f"""
            <div class="spark-card">
                <div class="spark-label">{row['label']}</div>
                <div class="spark-value">{row['value']}</div>
                <div class="spark-bar"><span style="width:{min(100, max(12, row['width']))}%;"></span></div>
                <div class="spark-subtext">{row['subtitle']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("\n")

model_names = load_model_names()
model_tab, pipeline_tab, sample_tab = st.tabs(["Model results", "Pipeline evidence", "Sample data"])

with model_tab:
    metrics_data = metrics.get("evaluation")
    st.header("Model results")
    if not metrics_data:
        st.info("Model metrics are not available. Run python main.py to generate outputs/evaluation.json.")
    else:
        split_map = metrics_data.get("splits", {})
        for split_name, split_results in split_map.items():
            st.subheader(f"Split: {split_name}")
            rows_list = []
            for model_key, values in split_results.items():
                display_name = model_names.get(model_key, model_key)
                conf = values.get("confusion", {})
                rows_list.append(
                    {
                        "model": display_name,
                        "precision": values.get("precision"),
                        "recall": values.get("recall"),
                        "f1": values.get("f1"),
                        "roc_auc": values.get("roc_auc"),
                        "average_precision": values.get("average_precision"),
                        "rows": values.get("rows"),
                        "tp": conf.get("tp"),
                        "fp": conf.get("fp"),
                        "fn": conf.get("fn"),
                        "tn": conf.get("tn"),
                    }
                )
            if rows_list:
                st.dataframe(pd.DataFrame(rows_list), use_container_width=True)

with pipeline_tab:
    st.header("Pipeline evidence")
    if not scalability:
        st.info("No scalability data is available.")
    else:
        timings = scalability.get("timings_seconds", {})
        if timings:
            st.subheader("Stage timings (seconds)")
            st.dataframe(pd.DataFrame([timings]).T.rename(columns={0: "seconds"}), use_container_width=True)
        if scalability.get("engines"):
            st.subheader("Graph engine by algorithm")
            st.dataframe(
                pd.DataFrame.from_dict(
                    scalability["engines"], orient="index", columns=["engine"]
                ).reset_index().rename(columns={"index": "algorithm"}),
                use_container_width=True,
            )
        st.json(scalability)

with sample_tab:
    st.header("Sample data")
    csv_path = next(iter(sorted(Path("data").glob("*.csv"))), None)
    if csv_path is None:
        st.info("No sample CSV is available in data/.")
    else:
        try:
            sample_df = pd.read_csv(csv_path, nrows=1000)
            st.dataframe(sample_df.head(20), use_container_width=True)
        except Exception as exc:  # pragma: no cover
            st.warning(f"Unable to read sample data: {exc}")
