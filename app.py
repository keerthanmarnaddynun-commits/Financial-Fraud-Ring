"""Financial Fraud Ring Analytics - PaySim Big Data Pipeline.

Truthful Streamlit multi-tab dashboard reading strictly from pipeline output artifacts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

PAGE_TITLE = "Financial Fraud Ring Analytics - PaySim Big Data Pipeline"
OUTPUT_DIR = Path(os.getenv("FRAUD_OUTPUT_DIR", "outputs"))
MODEL_NAMES_PATH = Path("model_names.json")

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark styling without external web fonts or remote images
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #071a17 0%, #0b1716 100%);
            color: #edf7f4;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        .metric-card {
            background: rgba(17, 26, 24, 0.92);
            border: 1px solid rgba(121, 155, 133, 0.4);
            border-radius: 10px;
            padding: 16px 18px;
            color: #ffffff;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
            min-height: 140px;
            margin-bottom: 12px;
        }
        .metric-label {
            font-size: 11px;
            color: #87a39b;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .metric-val {
            font-size: clamp(1.6rem, 2.1vw, 2.4rem);
            font-weight: 700;
            color: #edf7f4;
            line-height: 1.15;
        }
        .metric-subtext {
            color: #a6b8b3;
            font-size: 0.85rem;
            margin-top: 8px;
            line-height: 1.4;
        }
        .metric-bar {
            height: 3px;
            width: 100%;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 999px;
            margin-top: 10px;
            overflow: hidden;
        }
        .metric-bar > span {
            display: block;
            height: 100%;
            background: #38bdf8;
            border-radius: 999px;
        }
        .badge-fraud {
            background-color: #ef4444;
            color: white;
            padding: 2px 7px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 11px;
        }
        .badge-normal {
            background-color: #10b981;
            color: white;
            padding: 2px 7px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 11px;
        }
        .header-title {
            color: #edf7f4;
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 4px;
        }
        .header-meta {
            color: #8dcab0;
            font-size: 0.82rem;
            font-family: Consolas, "Liberation Mono", Courier, monospace;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_file_mtime(path: Path) -> float:
    """Return file modification time to drive reactive caching."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data
def read_json_file(file_path_str: str, mtime: float) -> Optional[Dict[str, Any]]:
    path = Path(file_path_str)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None


def load_output_json(filename: str) -> Optional[Dict[str, Any]]:
    path = OUTPUT_DIR / filename
    return read_json_file(str(path), get_file_mtime(path))


@st.cache_data
def load_model_names_map(mtime: float) -> Dict[str, str]:
    defaults = {
        "m0_rule": "M0 Rule Baseline",
        "m1_tabular": "M1 Tabular Baseline",
        "m2_graph": "M2 Graph + Structural",
        "m3_community_mule": "M3 Community + Mule",
        "tabular_baseline": "Baseline (Tabular)",
        "graph_features": "Model 1 (Structural Graph)",
        "community_ring": "Model 2 (Community / Rings)",
        "Baseline": "Baseline",
        "Model 1 (Node2Vec)": "Model 1 (Node2Vec)",
        "Model 2 (Structural GraphX)": "Model 2 (Structural GraphX)",
        "Model 3 (Community / Rings)": "Model 3 (Community / Rings)",
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


def format_seconds(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}s" if value >= 1 else f"{value:.3f}s"


def format_bytes(value: Optional[int | float]) -> str:
    if value is None:
        return "n/a"
    num = float(value)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num) < 1024.0 or unit == "TB":
            return f"{num:,.2f} {unit}" if unit != "B" else f"{int(num)} B"
        num /= 1024.0
    return "n/a"


def render_metric_card(
    label: str,
    value: str,
    subtitle: str,
    ratio: Optional[float] = None,
) -> None:
    progress_html = ""
    if ratio is not None:
        clamped_pct = max(0.0, min(100.0, float(ratio) * 100.0))
        progress_html = f'<div class="metric-bar"><span style="width:{clamped_pct:.1f}%;"></span></div>'
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-val">{value}</div>
            {progress_html}
            <div class="metric-subtext">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_trained_model(mtime: float) -> Tuple[Optional[Any], str]:
    search_paths = [
        OUTPUT_DIR / "models" / "baseline.joblib",
        OUTPUT_DIR / "baseline.joblib",
        Path("outputs") / "models" / "baseline.joblib",
        Path("output") / "models" / "baseline.joblib",
    ]
    for p in search_paths:
        if p.exists():
            try:
                loaded = joblib.load(str(p))
                return loaded, str(p)
            except Exception:
                continue
    return None, ""


@st.cache_data
def load_fraud_sample_transactions(
    csv_path_str: str,
    mtime: float,
    max_fraud: int = 150,
    max_normal: int = 100,
) -> pd.DataFrame:
    path = Path(csv_path_str)
    if not path.exists():
        return pd.DataFrame()
    fraud_rows: List[pd.DataFrame] = []
    normal_rows: List[pd.DataFrame] = []
    tot_fraud = 0
    tot_normal = 0
    try:
        for chunk in pd.read_csv(path, chunksize=50000):
            if "isFraud" in chunk.columns:
                frauds = chunk[chunk["isFraud"] == 1]
                if not frauds.empty and tot_fraud < max_fraud:
                    take = frauds.head(max_fraud - tot_fraud)
                    fraud_rows.append(take)
                    tot_fraud += len(take)
                normals = chunk[chunk["isFraud"] == 0]
                if not normals.empty and tot_normal < max_normal:
                    take = normals.head(max_normal - tot_normal)
                    normal_rows.append(take)
                    tot_normal += len(take)
            if tot_fraud >= max_fraud and tot_normal >= max_normal:
                break
        collected: List[pd.DataFrame] = []
        if fraud_rows:
            collected.extend(fraud_rows)
        if normal_rows:
            collected.extend(normal_rows)
        if collected:
            return pd.concat(collected, ignore_index=True)
    except Exception:
        pass
    return pd.DataFrame()


# Load primary files
model_names = load_model_names_map(get_file_mtime(MODEL_NAMES_PATH))
scalability_data = load_output_json("scalability.json")
evaluation_data = load_output_json("evaluation.json")
data_quality_data = load_output_json("data_quality.json")
run_info_data = load_output_json("run_info.json")
scaling_runs_data = load_output_json("scaling_runs.json")

# Header
app_id = (
    scalability_data.get("spark_application_id")
    if scalability_data
    else "N/A"
)
st.markdown(
    f"""
    <div style="display:flex; justify-content:space-between; align-items:flex-end; margin-bottom: 18px; border-bottom: 1px solid rgba(121, 155, 133, 0.3); padding-bottom: 10px;">
        <div>
            <div class="header-title">{PAGE_TITLE}</div>
            <div style="color: #94a3b8; font-size: 0.9rem;">Empirical results loaded strictly from <code>{OUTPUT_DIR}</code></div>
        </div>
        <div class="header-meta">Spark Application: {app_id or 'local[*]'}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_overview, tab_models, tab_pipeline, tab_scaling, tab_scorer, tab_inspector = st.tabs(
    [
        "📈 Overview",
        "📊 Model Results",
        "⚡ Pipeline Evidence",
        "🚀 Scaling",
        "🛡️ Live Scorer",
        "🕸️ Fraud Ring Inspector",
    ]
)


# =========================================================================
# TAB 1: OVERVIEW
# =========================================================================
with tab_overview:
    st.subheader("System Performance & Overview")
    if not scalability_data and not data_quality_data:
        st.info(f"Overview metrics not available. Run `python main.py --output-dir {OUTPUT_DIR}` to generate pipeline output.")
    else:
        # Extract KPI values honestly
        timings = scalability_data.get("timings_seconds", {}) if scalability_data else {}
        pipeline_total_sec = timings.get("pipeline_total")
        txns_count = (
            scalability_data.get("transaction_count")
            if scalability_data
            else (data_quality_data.get("rows_kept") if data_quality_data else None)
        )
        accounts_count = (
            scalability_data.get("account_count")
            if scalability_data
            else (data_quality_data.get("distinct_accounts") if data_quality_data else None)
        )
        sh_read = scalability_data.get("shuffle_read_bytes") if scalability_data else None
        sh_write = scalability_data.get("shuffle_write_bytes") if scalability_data else None
        stages_count = scalability_data.get("stage_count") if scalability_data else None

        c1, c2, c3 = st.columns(3)
        with c1:
            val_str = f"{txns_count:,}" if txns_count is not None else "n/a"
            render_metric_card("Transactions Processed", val_str, "Validated payment transactions")
        with c2:
            val_str = f"{accounts_count:,}" if accounts_count is not None else "n/a"
            render_metric_card("Distinct Accounts", val_str, "Unique origin and destination entities")
        with c3:
            val_str = format_seconds(pipeline_total_sec)
            render_metric_card("Total Pipeline Runtime", val_str, "End-to-end execution duration")

        c4, c5, c6 = st.columns(3)
        with c4:
            read_str = format_bytes(sh_read)
            write_str = format_bytes(sh_write)
            sub_str = f"Read: {read_str} | Write: {write_str}"
            tot_bytes = (sh_read or 0) + (sh_write or 0) if (sh_read or sh_write) else None
            tot_str = format_bytes(tot_bytes) if tot_bytes is not None else "n/a"
            render_metric_card("Total Shuffle Volume", tot_str, sub_str)
        with c5:
            val_str = f"{stages_count:,}" if stages_count is not None else "n/a"
            render_metric_card("Spark Stages", val_str, "Distributed execution stages completed")
        with c6:
            exec_mem = scalability_data.get("executor_storage_memory_used_bytes") or scalability_data.get("executor_memory_bytes") if scalability_data else None
            render_metric_card("Executor Storage Memory", format_bytes(exec_mem), "Spark memory reported by REST listener")

        st.divider()

        # Data quality summary if present
        if data_quality_data:
            st.subheader("Data Quality & Ingestion Summary")
            dq1, dq2, dq3, dq4 = st.columns(4)
            with dq1:
                dq1.metric("Raw Rows Read", f"{data_quality_data.get('rows_read', 0):,}")
            with dq2:
                dq2.metric("Cleaned Rows Kept", f"{data_quality_data.get('rows_kept', 0):,}")
            with dq3:
                dq3.metric("Labeled Fraud Transactions", f"{data_quality_data.get('fraud_rows', 0):,}")
            with dq4:
                dq4.metric("Distinct Accounts", f"{data_quality_data.get('distinct_accounts', 0):,}")

            drops = data_quality_data.get("dropped_by_rule", {})
            if drops:
                st.write("**Dropped Rows Breakdown:**")
                drop_df = pd.DataFrame([
                    {"Validation Rule": k.replace("_", " ").title(), "Dropped Count": f"{v:,}"}
                    for k, v in drops.items()
                ])
                st.dataframe(drop_df, use_container_width=True)

        if scalability_data:
            with st.expander("Raw file: scalability.json"):
                st.json(scalability_data)


# =========================================================================
# TAB 2: MODEL RESULTS
# =========================================================================
with tab_models:
    st.subheader("Model Evaluation & Comparative Performance")
    if not evaluation_data:
        st.info(f"Model evaluation metrics not available. Run `python main.py --output-dir {OUTPUT_DIR}` to generate outputs/evaluation.json.")
    else:
        # Check layout: new 'splits' vs legacy
        is_splits_layout = "splits" in evaluation_data and isinstance(evaluation_data["splits"], dict)

        active_split_data: Dict[str, Any] = {}
        if is_splits_layout:
            splits_dict = evaluation_data["splits"]
            split_keys = list(splits_dict.keys())
            selected_split = st.selectbox("Evaluation Split Strategy", split_keys)
            active_split_data = splits_dict.get(selected_split, {})
        else:
            # Legacy layout: top-level keys like tabular_baseline, graph_features, community_ring
            st.caption("Displaying results from legacy evaluation format.")
            active_split_data = evaluation_data

        # Parse transaction metrics
        txn_records = []
        ring_records = []
        has_confusion_data = False

        for m_key, m_val in active_split_data.items():
            if m_key in ("splits", "models") or not isinstance(m_val, dict):
                continue
            display_name = model_names.get(m_key, m_key.replace("_", " ").title())

            # Check if using the new 'at_threshold' contract or legacy nested 'transaction'
            if "at_threshold" in m_val:
                t_data = m_val["at_threshold"]
                r_data = {}
            elif "transaction" in m_val:
                t_data = m_val.get("transaction", {})
                r_data = m_val.get("ring", {})
            else:
                t_data = m_val
                r_data = {}

            prec = t_data.get("precision", m_val.get("precision"))
            rec = t_data.get("recall", m_val.get("recall"))
            f1 = t_data.get("f1", m_val.get("f1"))
            auc = m_val.get("roc_auc", t_data.get("roc_auc"))
            ap = m_val.get("average_precision", t_data.get("average_precision"))
            rows = m_val.get("rows", t_data.get("rows"))

            record: Dict[str, Any] = {
                "Model Key": m_key,
                "Model": display_name,
                "Precision": float(prec) if prec is not None else None,
                "Recall": float(rec) if rec is not None else None,
                "F1-Score": float(f1) if f1 is not None else None,
                "ROC-AUC": float(auc) if auc is not None else None,
                "Test Rows": int(rows) if rows is not None else None,
            }
            if ap is not None:
                record["PR-AUC"] = float(ap)
            if "threshold" in m_val:
                record["Threshold"] = float(m_val["threshold"])
            if "false_alarms_per_10k" in m_val:
                record["False Alarms / 10k"] = float(m_val["false_alarms_per_10k"])

            # Check for honest confusion matrix
            conf = t_data.get("confusion")
            if isinstance(conf, dict) and all(k in conf for k in ("tp", "fp", "fn", "tn")):
                record["TP"] = conf["tp"]
                record["FP"] = conf["fp"]
                record["FN"] = conf["fn"]
                record["TN"] = conf["tn"]
                has_confusion_data = True

            txn_records.append(record)

            if r_data:
                ring_records.append({
                    "Model": display_name,
                    "Rings Evaluated": r_data.get("rings_evaluated"),
                    "Fraud Rings": r_data.get("fraud_rings"),
                    "Detected Rings": r_data.get("detected_rings"),
                    "Precision": r_data.get("precision"),
                    "Recall": r_data.get("recall"),
                    "F1-Score": r_data.get("f1"),
                    "ROC-AUC": r_data.get("roc_auc"),
                })

        df_txn = pd.DataFrame(txn_records)
        if not df_txn.empty:
            # Grouped bar chart
            metric_cols = [c for c in ["Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"] if c in df_txn.columns and df_txn[c].notna().any()]
            melted = df_txn.melt(id_vars=["Model"], value_vars=metric_cols, var_name="Metric", value_name="Score").dropna()

            fig_bar = px.bar(
                melted,
                x="Model",
                y="Score",
                color="Metric",
                barmode="group",
                text_auto=".3f",
                height=380,
                color_discrete_sequence=["#38bdf8", "#34d399", "#fbbf24", "#f87171", "#a78bfa"],
            )
            fig_bar.update_layout(
                yaxis_range=[0, 1.15],
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17, 26, 24, 0.6)",
                font=dict(color="#edf7f4"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            st.subheader("Transaction Classification Metrics Table")
            display_cols = ["Model"] + metric_cols + ["Test Rows"]
            st.dataframe(df_txn[display_cols].dropna(how="all", axis=1), use_container_width=True)

            st.divider()

            # Confusion Matrix: ONLY if real confusion counts exist in the file
            st.subheader("Confusion Matrix")
            if has_confusion_data:
                conf_models = df_txn[df_txn["TP"].notna()]["Model"].tolist()
                selected_model = st.selectbox("Select Model for Confusion Matrix", conf_models)
                sel_row = df_txn[df_txn["Model"] == selected_model].iloc[0]
                cm = [
                    [int(sel_row["TN"]), int(sel_row["FP"])],
                    [int(sel_row["FN"]), int(sel_row["TP"])],
                ]
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    color_continuous_scale="Blues",
                    labels=dict(x="Predicted Label", y="Actual Label", color="Count"),
                    x=["Legitimate (0)", "Fraud (1)"],
                    y=["Legitimate (0)", "Fraud (1)"],
                    height=320,
                )
                fig_cm.update_layout(
                    margin=dict(l=20, r=20, t=25, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#edf7f4"),
                )
                st.plotly_chart(fig_cm, use_container_width=True)
            else:
                st.info("Confusion matrix counts (TP/FP/FN/TN) are not present in this evaluation file. (Only exact measured confusion counts are shown; no estimates are synthesized).")

            # Legacy Ring Metrics Table
            if ring_records:
                st.divider()
                st.subheader("Community-level alert metrics (not ground-truth rings)")
                st.caption(
                    "Note: In PaySim, fraudulent activity primarily comprises isolated transaction pairs. "
                    "These metrics reflect community-level alert coverage rather than ground-truth syndicated fraud rings."
                )
                df_ring = pd.DataFrame(ring_records)
                st.dataframe(df_ring, use_container_width=True)

        with st.expander("Raw file: evaluation.json"):
            st.json(evaluation_data)


# =========================================================================
# TAB 3: PIPELINE EVIDENCE
# =========================================================================
with tab_pipeline:
    st.subheader("Spark Pipeline Evidence & Diagnostics")
    if not scalability_data:
        st.info(f"Pipeline scalability evidence not available. Run `python main.py --output-dir {OUTPUT_DIR}` to generate outputs/scalability.json.")
    else:
        timings = scalability_data.get("timings_seconds", {})
        if timings:
            st.subheader("Measured Stage Timings")
            timing_items = [
                {"Stage": k.replace("_", " ").title(), "Seconds": float(v)}
                for k, v in timings.items()
                if k != "pipeline_total"
            ]
            if timing_items:
                df_timings = pd.DataFrame(timing_items)
                fig_timings = px.bar(
                    df_timings,
                    x="Stage",
                    y="Seconds",
                    text_auto=".2f",
                    color="Seconds",
                    color_continuous_scale="Viridis",
                    height=350,
                )
                fig_timings.update_layout(
                    margin=dict(l=20, r=20, t=25, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(17, 26, 24, 0.6)",
                    font=dict(color="#edf7f4"),
                )
                st.plotly_chart(fig_timings, use_container_width=True)
                st.dataframe(pd.DataFrame([timings]).T.rename(columns={0: "Seconds"}), use_container_width=True)

        engines = scalability_data.get("engines")
        if engines:
            st.subheader("Graph Algorithm Execution Engines")
            eng_df = pd.DataFrame(
                [{"Algorithm": k, "Engine": v} for k, v in engines.items()]
            )
            st.dataframe(eng_df, use_container_width=True)

        st.subheader("Shuffle & Storage Resource Consumption")
        sh_read_b = scalability_data.get("shuffle_read_bytes")
        sh_write_b = scalability_data.get("shuffle_write_bytes")
        sh_df = pd.DataFrame([
            {"Metric": "Shuffle Read Bytes", "Value": format_bytes(sh_read_b), "Raw Bytes": sh_read_b},
            {"Metric": "Shuffle Write Bytes", "Value": format_bytes(sh_write_b), "Raw Bytes": sh_write_b},
            {"Metric": "Stage Count", "Value": str(scalability_data.get("stage_count")), "Raw Bytes": None},
            {"Metric": "Executor Storage Memory", "Value": format_bytes(scalability_data.get("executor_storage_memory_used_bytes") or scalability_data.get("executor_memory_bytes")), "Raw Bytes": None},
        ])
        st.dataframe(sh_df, use_container_width=True)

        if data_quality_data:
            st.subheader("Data Validation Report (`data_quality.json`)")
            st.dataframe(pd.DataFrame([data_quality_data]), use_container_width=True)

        if run_info_data:
            st.subheader("Execution Environment (`run_info.json`)")
            st.dataframe(pd.DataFrame([run_info_data]), use_container_width=True)

        with st.expander("Raw file: scalability.json"):
            st.json(scalability_data)


# =========================================================================
# TAB 4: SCALING
# =========================================================================
with tab_scaling:
    st.subheader("Distributed Systems Scaling Benchmarks")
    if not scaling_runs_data:
        st.info("Run the scaling benchmark (`python benchmark_scaling.py`) to generate scaling results.")
    else:
        st.write("Scaling benchmark data loaded from `scaling_runs.json`.")
        try:
            if isinstance(scaling_runs_data, list):
                df_scale = pd.DataFrame(scaling_runs_data)
                st.dataframe(df_scale, use_container_width=True)
                if "fraction" in df_scale.columns and "seconds" in df_scale.columns:
                    st.line_chart(df_scale.set_index("fraction")["seconds"])
            elif isinstance(scaling_runs_data, dict):
                st.json(scaling_runs_data)
        except Exception as e:
            st.error(f"Error rendering scaling data: {e}")

        with st.expander("Raw file: scaling_runs.json"):
            st.json(scaling_runs_data)


# =========================================================================
# TAB 5: LIVE SCORER
# =========================================================================
with tab_scorer:
    st.subheader("Interactive Live Transaction Scorer")
    st.markdown("Test trained model weights with arbitrary transaction features.")

    model_obj, model_source_path = load_trained_model(
        get_file_mtime(OUTPUT_DIR / "models" / "baseline.joblib")
    )

    if model_obj is None:
        st.info("Baseline model weights not found in `outputs/models/baseline.joblib` or `outputs/baseline.joblib`.")
    else:
        st.caption(f"Loaded model weights from: `{model_source_path}`")

        # 3 Presets
        st.markdown("**Load Preset Scenario:**")
        p_col1, p_col2, p_col3 = st.columns(3)

        preset_choice = None
        if p_col1.button("🟢 Normal Payment", use_container_width=True):
            preset_choice = "normal"
        if p_col2.button("🔴 Suspicious Cash-Out", use_container_width=True):
            preset_choice = "suspicious"
        if p_col3.button("🟡 Borderline Transfer", use_container_width=True):
            preset_choice = "borderline"

        if preset_choice == "normal":
            st.session_state["sc_type"] = "PAYMENT"
            st.session_state["sc_step"] = 14
            st.session_state["sc_amt"] = 85.0
            st.session_state["sc_old_orig"] = 1200.0
            st.session_state["sc_new_orig"] = 1115.0
            st.session_state["sc_old_dest"] = 350.0
            st.session_state["sc_new_dest"] = 435.0
        elif preset_choice == "suspicious":
            st.session_state["sc_type"] = "CASH_OUT"
            st.session_state["sc_step"] = 48
            st.session_state["sc_amt"] = 320000.0
            st.session_state["sc_old_orig"] = 320000.0
            st.session_state["sc_new_orig"] = 0.0
            st.session_state["sc_old_dest"] = 0.0
            st.session_state["sc_new_dest"] = 320000.0
        elif preset_choice == "borderline":
            st.session_state["sc_type"] = "TRANSFER"
            st.session_state["sc_step"] = 92
            st.session_state["sc_amt"] = 18500.0
            st.session_state["sc_old_orig"] = 25000.0
            st.session_state["sc_new_orig"] = 6500.0
            st.session_state["sc_old_dest"] = 5000.0
            st.session_state["sc_new_dest"] = 23500.0

        with st.form("live_scoring_form"):
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                cur_type = st.session_state.get("sc_type", "CASH_OUT")
                type_idx = ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"].index(cur_type) if cur_type in ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"] else 0
                in_type = st.selectbox("Transaction Type", ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"], index=type_idx)
                in_step = st.number_input("Time Step (Hour 1-744)", min_value=1, max_value=744, value=int(st.session_state.get("sc_step", 45)))
                in_amt = st.number_input("Transaction Amount ($)", min_value=0.01, value=float(st.session_state.get("sc_amt", 250000.0)), step=1000.0)
            with col_f2:
                in_old_orig = st.number_input("Sender Old Balance ($)", min_value=0.0, value=float(st.session_state.get("sc_old_orig", 250000.0)), step=1000.0)
                in_new_orig = st.number_input("Sender New Balance ($)", min_value=0.0, value=float(st.session_state.get("sc_new_orig", 0.0)), step=1000.0)
                in_old_dest = st.number_input("Receiver Old Balance ($)", min_value=0.0, value=float(st.session_state.get("sc_old_dest", 0.0)), step=1000.0)
                in_new_dest = st.number_input("Receiver New Balance ($)", min_value=0.0, value=float(st.session_state.get("sc_new_dest", 250000.0)), step=1000.0)

            score_submitted = st.form_submit_button("⚡ Score Transaction Risk")

        if score_submitted:
            err_orig = in_new_orig + in_amt - in_old_orig
            err_dest = in_old_dest + in_amt - in_new_dest

            type_cols = ["type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]
            type_vec = [1 if in_type == col.replace("type_", "") else 0 for col in type_cols]

            expected_features = getattr(model_obj, "feature_names_in_", None)

            # Construct input features dictionary
            feat_dict: Dict[str, Any] = {
                "step": in_step,
                "amount": in_amt,
                "oldbalanceOrg": in_old_orig,
                "newbalanceOrig": in_new_orig,
                "oldbalanceDest": in_old_dest,
                "newbalanceDest": in_new_dest,
                "errorBalanceOrig": err_orig,
                "errorBalanceDest": err_dest,
            }
            for col_name, val in zip(type_cols, type_vec):
                feat_dict[col_name] = val

            # Check if model expects isFlaggedFraud
            if expected_features is not None and "isFlaggedFraud" in expected_features:
                feat_dict["isFlaggedFraud"] = 0
                st.caption("ℹ️ Model weights expect `isFlaggedFraud`; passed 0 to avoid target leakage. The model should be retrained without this feature.")

            # Align columns
            if expected_features is not None:
                aligned_df = pd.DataFrame([[feat_dict.get(c, 0.0) for c in expected_features]], columns=expected_features)
            else:
                aligned_df = pd.DataFrame([feat_dict])

            # Predict probability
            try:
                prob = float(model_obj.predict_proba(aligned_df)[0, 1])
                pred = int(prob >= 0.5)

                r_col1, r_col2 = st.columns([1, 2])
                with r_col1:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100.0,
                        title={'text': "Fraud Risk (%)"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#ef4444" if prob >= 0.5 else "#10b981"},
                            'steps': [
                                {'range': [0, 50], 'color': "rgba(16, 185, 129, 0.2)"},
                                {'range': [50, 100], 'color': "rgba(239, 68, 68, 0.2)"},
                            ],
                            'threshold': {
                                'line': {'color': "#ef4444", 'width': 3},
                                'thickness': 0.75,
                                'value': 50,
                            },
                        },
                    ))
                    fig_gauge.update_layout(
                        height=260,
                        margin=dict(l=20, r=20, t=30, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#edf7f4"),
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with r_col2:
                    st.write("### Model Decision Outcome")
                    if pred == 1:
                        st.error(f"🚨 **FLAGGED HIGH RISK / FRAUD** (Probability: {prob * 100.0:.2f}%)")
                    else:
                        st.success(f"✅ **NORMAL TRANSACTION** (Probability: {prob * 100.0:.2f}%)")

                    # Feature contribution analysis (for XGBoost)
                    if hasattr(model_obj, "get_booster"):
                        try:
                            booster = model_obj.get_booster()
                            dmat = xgb.DMatrix(aligned_df)
                            contribs = booster.predict(dmat, pred_contribs=True)
                            # contribs shape: (1, num_features + 1), last element is base margin / bias
                            col_names_used = list(aligned_df.columns)
                            shap_vals = contribs[0][:-1]
                            contrib_records = [
                                {
                                    "Feature": col,
                                    "Contribution (log-odds)": round(float(v), 4),
                                    "Impact": "Increases Fraud Risk" if v > 0 else "Decreases Fraud Risk",
                                    "Magnitude": abs(float(v)),
                                }
                                for col, v in zip(col_names_used, shap_vals)
                            ]
                            df_contrib = pd.DataFrame(contrib_records).sort_values(by="Magnitude", ascending=False).drop(columns=["Magnitude"])
                            st.write("**Top Feature Contributions (Measured SHAP Margins):**")
                            st.dataframe(df_contrib.head(6), use_container_width=True)
                        except Exception as ex:
                            st.caption(f"Feature contributions not computed: {ex}")

            except Exception as e:
                st.error(f"Scoring error: {e}")


# =========================================================================
# TAB 6: FRAUD RING INSPECTOR
# =========================================================================
with tab_inspector:
    st.subheader("Fraud Ring Inspector")
    st.markdown("Topological network visualization: **Sample of labelled fraud transactions, not detected rings**.")

    # Locate dataset
    data_dir = Path("data")
    csv_candidates = sorted(data_dir.glob("*.csv")) + sorted(data_dir.glob("*/*.csv"))
    csv_path = csv_candidates[0] if csv_candidates else None

    if csv_path is None:
        st.info("No PaySim CSV dataset found in `data/`. Place the CSV file in `data/` to view the graph inspector.")
    else:
        sample_df = load_fraud_sample_transactions(str(csv_path), get_file_mtime(csv_path))
        if sample_df.empty:
            st.warning("Could not read sample transactions from the CSV dataset.")
        else:
            ctl1, ctl2, ctl3 = st.columns(3)
            with ctl1:
                filter_choice = st.radio("Transaction Subset", ["Fraud Transactions Only", "All Sampled Transactions"], horizontal=True)
            with ctl2:
                max_render = st.slider("Max Transactions to Render", min_value=10, max_value=150, value=40, step=10)
            with ctl3:
                account_search = st.text_input("Filter by Account ID", "")

            filtered_df = sample_df.copy()
            if filter_choice == "Fraud Transactions Only" and "isFraud" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["isFraud"] == 1]
            if account_search.strip():
                filtered_df = filtered_df[
                    filtered_df["nameOrig"].str.contains(account_search, case=False, na=False) |
                    filtered_df["nameDest"].str.contains(account_search, case=False, na=False)
                ]
            filtered_df = filtered_df.head(max_render)

            if filtered_df.empty:
                st.warning("No transactions match the selected filter criteria.")
            else:
                G = nx.DiGraph()
                for _, row in filtered_df.iterrows():
                    src = str(row["nameOrig"])
                    dst = str(row["nameDest"])
                    amt = float(row.get("amount", 0.0))
                    is_f = int(row.get("isFraud", 0))
                    t_type = str(row.get("type", "TRANSFER"))
                    G.add_edge(src, dst, amount=amt, is_fraud=is_f, type=t_type)

                # Assign roles based on degree
                for n in G.nodes():
                    in_d = G.in_degree(n)
                    out_d = G.out_degree(n)
                    if in_d > 0 and out_d > 0:
                        G.nodes[n]["role"] = "Intermediary / Mule Account"
                    elif in_d > 0:
                        G.nodes[n]["role"] = "Destination Account"
                    else:
                        G.nodes[n]["role"] = "Origin Account"

                pos = nx.spring_layout(G, seed=42, k=0.35)

                # Build edge traces
                edge_norm_x, edge_norm_y = [], []
                edge_fraud_x, edge_fraud_y = [], []

                for u, v, data in G.edges(data=True):
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    if data.get("is_fraud") == 1:
                        edge_fraud_x.extend([x0, x1, None])
                        edge_fraud_y.extend([y0, y1, None])
                    else:
                        edge_norm_x.extend([x0, x1, None])
                        edge_norm_y.extend([y0, y1, None])

                trace_norm_edges = go.Scatter(
                    x=edge_norm_x, y=edge_norm_y,
                    line=dict(width=1, color="rgba(148, 163, 184, 0.35)"),
                    hoverinfo="none",
                    mode="lines",
                    name="Normal Transaction Link",
                )
                trace_fraud_edges = go.Scatter(
                    x=edge_fraud_x, y=edge_fraud_y,
                    line=dict(width=2.5, color="#ef4444"),
                    hoverinfo="none",
                    mode="lines",
                    name="Fraud Transaction Link",
                )

                # Build node traces colored by role
                node_x, node_y, node_colors, node_texts, node_sizes = [], [], [], [], []
                for n in G.nodes():
                    x, y = pos[n]
                    node_x.append(x)
                    node_y.append(y)
                    role = G.nodes[n]["role"]
                    in_d = G.in_degree(n)
                    out_d = G.out_degree(n)

                    if role == "Origin Account":
                        node_colors.append("#38bdf8")  # Sky blue
                        node_sizes.append(11)
                    elif role == "Destination Account":
                        node_colors.append("#10b981")  # Emerald green
                        node_sizes.append(11)
                    else:
                        node_colors.append("#f59e0b")  # Amber / Intermediary
                        node_sizes.append(15)

                    node_texts.append(
                        f"Account: <b>{n}</b><br>Role: <b>{role}</b><br>In-Degree: {in_d} | Out-Degree: {out_d}"
                    )

                trace_nodes = go.Scatter(
                    x=node_x, y=node_y,
                    mode="markers",
                    hoverinfo="text",
                    text=node_texts,
                    marker=dict(
                        color=node_colors,
                        size=node_sizes,
                        line=dict(width=1.5, color="#ffffff"),
                    ),
                    name="Accounts (Blue=Origin, Green=Dest, Amber=Intermediary)",
                )

                fig_net = go.Figure(
                    data=[trace_norm_edges, trace_fraud_edges, trace_nodes],
                    layout=go.Layout(
                        title=dict(text="Network Graph (Sample of Labelled Fraud Transactions, Not Detected Rings)", font=dict(size=14)),
                        showlegend=True,
                        hovermode="closest",
                        margin=dict(b=20, l=20, r=20, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=500,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="#071a17",
                        font=dict(color="#edf7f4"),
                    ),
                )
                st.plotly_chart(fig_net, use_container_width=True)

                st.subheader("Sub-Graph Transaction Records")
                cols_to_show = [c for c in ["step", "type", "nameOrig", "nameDest", "amount", "oldbalanceOrg", "newbalanceOrig", "isFraud"] if c in filtered_df.columns]
                st.dataframe(filtered_df[cols_to_show], use_container_width=True)

st.divider()
st.caption("Financial Fraud Ring Analytics - PaySim Big Data Pipeline • Strictly honest data layer")
