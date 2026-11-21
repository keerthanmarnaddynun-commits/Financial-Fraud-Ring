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
    card_html = (
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-val">{value}</div>'
        f'{progress_html}'
        f'<div class="metric-subtext">{subtitle}</div>'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


@st.cache_resource
def load_trained_model(mtime: float) -> Tuple[Optional[Any], str]:
    search_paths = [
        OUTPUT_DIR / "models" / "m1_tabular.joblib",
        OUTPUT_DIR / "models" / "baseline.joblib",
        OUTPUT_DIR / "baseline.joblib",
        Path("outputs") / "models" / "m1_tabular.joblib",
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


@st.cache_data
def list_available_rings(rings_dir_str: str, mtime: float) -> List[str]:
    """List available ring JSON files in outputs/rings/."""
    rings_dir = Path(rings_dir_str)
    if not rings_dir.exists():
        return []
    return sorted([f.name for f in rings_dir.glob("ring_*.json")])


@st.cache_data
def load_ring_file(ring_path_str: str, mtime: float) -> Optional[Dict[str, Any]]:
    """Load a single ring JSON export."""
    path = Path(ring_path_str)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data
def load_rings_summary_csv(csv_path_str: str, mtime: float) -> Optional[pd.DataFrame]:
    """Load the full rings summary CSV."""
    path = Path(csv_path_str)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# Load primary files
model_names = load_model_names_map(get_file_mtime(MODEL_NAMES_PATH))
scalability_data = load_output_json("scalability.json")
evaluation_data = load_output_json("evaluation.json")
data_quality_data = load_output_json("data_quality.json")
run_info_data = load_output_json("run_info.json")
scaling_runs_data = load_output_json("scaling_runs.json")
pandas_vs_spark_data = load_output_json("pandas_vs_spark.json")
storage_benchmark_data = load_output_json("storage_benchmark.json")
business_impact_data = load_output_json("business_impact.json")
ring_analysis_data = load_output_json("ring_analysis.json")
planted_rings_data = load_output_json("planted_ring_validation.json")

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

tab_overview, tab_models, tab_pipeline, tab_impact, tab_rings, tab_scaling, tab_scorer = st.tabs(
    [
        "📈 Overview",
        "📊 Model Results",
        "⚡ Pipeline Evidence",
        "💼 Business Impact",
        "🕸️ Fraud Rings",
        "🚀 Scaling",
        "🛡️ Live Scorer",
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
        phases = scalability_data.get("phases", {}) if scalability_data else {}
        timings = scalability_data.get("timings_seconds", {}) if scalability_data else {}
        pipeline_total_sec = phases.get("pipeline_total") or timings.get("pipeline_total")
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
    st.info(
        "⚠️ **Methodological & Evaluation Limitations:**\n"
        "- **Simulated Data**: PaySim is a synthetic financial dataset generated via multi-agent simulation; real production fraud patterns may exhibit differing topologies.\n"
        "- **Transductive Graph Features**: Graph metrics (PageRank, Triangle Count, Degree, LPA Community IDs) were computed on the static snapshot of the transaction graph rather than rolling past-only subgraphs.\n"
        "- **Split Evaluation Integrity**: Models were trained on time-split rows. Results for `group_dest` and `random` are displayed for reference only under pending-retraining notice."
    )

    if not evaluation_data:
        st.info(f"Model evaluation metrics not available. Run `python main.py --output-dir {OUTPUT_DIR}` to generate outputs/evaluation.json.")
    else:
        # Check layout: new 'splits' vs legacy
        is_splits_layout = "splits" in evaluation_data and isinstance(evaluation_data["splits"], dict)

        active_split_data: Dict[str, Any] = {}
        if is_splits_layout:
            splits_dict = evaluation_data["splits"]
            split_keys = list(splits_dict.keys())
            time_idx = split_keys.index("time") if "time" in split_keys else 0
            selected_split = st.selectbox("Evaluation Split Strategy", split_keys, index=time_idx)
            if selected_split in ("group_dest", "random"):
                st.warning("⚠️ not a valid test: models were trained on time-split rows, retraining pending")
            active_split_data = splits_dict.get(selected_split, {})
        else:
            # Legacy layout: top-level keys like tabular_baseline, graph_features, community_ring
            st.caption("Displaying results from legacy evaluation format.")
            active_split_data = evaluation_data

        # Population and prevalence extraction for honest captions
        first_model_data = next((v for k, v in active_split_data.items() if isinstance(v, dict) and "rows" in v), {})
        test_rows_pop = first_model_data.get("rows")
        prev_rate = first_model_data.get("prevalence")

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
            if test_rows_pop is not None and prev_rate is not None:
                st.caption(
                    f"📊 **Evaluation Population:** {test_rows_pop:,} test transactions | "
                    f"**True Fraud Prevalence:** {prev_rate * 100:.3f}% ({int(round(test_rows_pop * prev_rate)):,} fraud cases) | "
                    f"**Threshold Selection:** Validation set maximum F1"
                )
            display_cols = [c for c in ["Model", "Threshold", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC", "False Alarms / 10k", "Test Rows"] if c in df_txn.columns]
            st.dataframe(df_txn[display_cols].dropna(how="all", axis=1), use_container_width=True)

            st.divider()

            # Precision-Recall Curves (if pr_curves.csv present)
            pr_path = OUTPUT_DIR / "pr_curves.csv"
            if pr_path.exists():
                st.subheader("Precision-Recall Curves")
                if test_rows_pop is not None and prev_rate is not None:
                    st.caption(
                        f"📊 Evaluated on {test_rows_pop:,} test rows (prevalence {prev_rate * 100:.3f}%). "
                        f"Curves computed from score deciles/quantiles in `{pr_path.name}`."
                    )
                try:
                    pr_df = pd.read_csv(pr_path)
                    if is_splits_layout and "split" in pr_df.columns:
                        pr_df = pr_df[pr_df["split"] == selected_split]
                    if not pr_df.empty:
                        pr_df["Model Name"] = pr_df["model"].map(lambda m: model_names.get(m, m.replace("_", " ").title()))
                        fig_pr = px.line(
                            pr_df,
                            x="recall",
                            y="precision",
                            color="Model Name",
                            title=f"Precision-Recall Curves ({selected_split if is_splits_layout else 'Test Set'})",
                            height=380,
                        )
                        fig_pr.update_layout(
                            xaxis_title="Recall",
                            yaxis_title="Precision",
                            yaxis_range=[0, 1.05],
                            margin=dict(l=20, r=20, t=35, b=20),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(17, 26, 24, 0.6)",
                            font=dict(color="#edf7f4"),
                        )
                        st.plotly_chart(fig_pr, use_container_width=True)
                except Exception as ex:
                    st.caption(f"Could not render PR curves: {ex}")

                st.divider()

            # Feature Importance (if feature_importance.json present)
            feat_imp_data = load_output_json("feature_importance.json")
            if feat_imp_data and isinstance(feat_imp_data, dict):
                st.subheader("Feature Importance (XGBoost Gain)")
                avail_models = [m for m in feat_imp_data.keys() if feat_imp_data[m]]
                if avail_models:
                    model_choice = st.selectbox(
                        "Select Model for Feature Importance",
                        avail_models,
                        format_func=lambda m: model_names.get(m, m.replace("_", " ").title()),
                    )
                    top_feats = feat_imp_data[model_choice][:15]
                    if top_feats:
                        df_imp = pd.DataFrame(top_feats).sort_values(by="importance", ascending=True)
                        fig_imp = px.bar(
                            df_imp,
                            x="importance",
                            y="feature",
                            orientation="h",
                            title=f"Top Features by Gain: {model_names.get(model_choice, model_choice)}",
                            height=400,
                            color_discrete_sequence=["#38bdf8"],
                        )
                        fig_imp.update_layout(
                            xaxis_title="Feature Importance (Gain)",
                            yaxis_title="Feature",
                            margin=dict(l=20, r=20, t=35, b=20),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(17, 26, 24, 0.6)",
                            font=dict(color="#edf7f4"),
                        )
                        st.plotly_chart(fig_imp, use_container_width=True)

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
                if test_rows_pop is not None and prev_rate is not None:
                    st.caption(
                        f"📊 Measured confusion matrix on {test_rows_pop:,} test transactions "
                        f"(prevalence: {prev_rate * 100:.3f}%). Operating threshold: `{float(sel_row.get('Threshold', 0.5)):.4f}`."
                    )
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
                if test_rows_pop is not None:
                    st.caption(f"Evaluated across community graph partitions extracted from {test_rows_pop:,} test transactions.")

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
        phases = scalability_data.get("phases", {})
        graph_stages = scalability_data.get("graph_stages", {})
        stages_detail = scalability_data.get("stages", {})
        timings = scalability_data.get("timings_seconds", {})

        # New layout: phases block present
        if phases:
            st.subheader("Pipeline Execution Phases")
            phase_items = []
            for k in ["ingestion_and_validation", "graph_construction", "graph_feature_engineering", "model_training", "distributed_scoring_and_eval"]:
                if k in phases:
                    phase_items.append({"Phase": k.replace("_", " ").title(), "Seconds": float(phases[k])})
            if phase_items:
                df_phases = pd.DataFrame(phase_items)
                fig_phases = px.bar(
                    df_phases,
                    x="Phase",
                    y="Seconds",
                    text_auto=".2f",
                    color="Seconds",
                    color_continuous_scale="Viridis",
                    height=320,
                )
                fig_phases.update_layout(
                    margin=dict(l=20, r=20, t=25, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(17, 26, 24, 0.6)",
                    font=dict(color="#edf7f4"),
                )
                st.plotly_chart(fig_phases, use_container_width=True)
                st.dataframe(df_phases, use_container_width=True)

            if "pipeline_total" in phases:
                st.caption(f"End-to-end pipeline duration: {format_seconds(phases['pipeline_total'])}")

            if graph_stages:
                st.subheader("Graph Algorithm Stages")
                graph_items = [{"Stage": k.replace("_", " ").title(), "Seconds": float(v)} for k, v in graph_stages.items()]
                df_graph = pd.DataFrame(graph_items)
                st.dataframe(df_graph, use_container_width=True)

            if stages_detail:
                st.subheader("Stage Resource Diagnostics")
                stage_rows = []
                for stg_name, stg_metrics in stages_detail.items():
                    if isinstance(stg_metrics, dict):
                        stage_rows.append({
                            "Stage": stg_name.replace("_", " ").title(),
                            "Wall Seconds": f"{stg_metrics.get('wall_seconds', 0.0):.2f}",
                            "Peak RSS": format_bytes(stg_metrics.get("peak_rss_bytes")),
                            "Min Avail RAM": format_bytes(stg_metrics.get("min_available_ram_bytes")),
                            "Stages Delta": stg_metrics.get("stages_count_delta"),
                            "Shuffle Read Delta": format_bytes(stg_metrics.get("shuffle_read_bytes_delta")),
                            "Shuffle Write Delta": format_bytes(stg_metrics.get("shuffle_write_bytes_delta")),
                        })
                if stage_rows:
                    st.dataframe(pd.DataFrame(stage_rows), use_container_width=True)

        elif timings:
            # Old layout fallback
            st.subheader("Measured Stage Timings")
            timing_items = []
            for k, v in timings.items():
                if k == "pipeline_total":
                    continue
                if isinstance(v, (int, float)):
                    timing_items.append({"Stage": k.replace("_", " ").title(), "Seconds": float(v)})
                elif isinstance(v, dict) and k == "timings_seconds":
                    for sub_k, sub_v in v.items():
                        if isinstance(sub_v, (int, float)):
                            timing_items.append({"Stage": f"Graph: {sub_k.replace('_', ' ').title()}", "Seconds": float(sub_v)})
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
                st.dataframe(df_timings, use_container_width=True)

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
# TAB 4: BUSINESS IMPACT
# =========================================================================
with tab_impact:
    st.subheader("Business Impact & Threshold Analysis (Time Split)")
    st.markdown(
        "Financial cost-benefit analysis of fraud detection across score operating thresholds. "
        "Evaluated on the 193,939 time-split test transactions (2,062 ground-truth fraud cases)."
    )

    if not business_impact_data:
        st.info(
            f"Business impact analysis not available. Run `python business_impact.py --output-dir {OUTPUT_DIR}` to generate."
        )
    else:
        st.caption(
            f"All financial values are denominated in **{business_impact_data.get('currency', 'PaySim simulated currency units')}**."
        )

        impact_models = business_impact_data.get("models", {})
        rule_baseline = business_impact_data.get("rule_baseline", {})
        total_fraud_val = float(business_impact_data.get("fraud_value_total", 0.0))
        total_fraud_txns = int(business_impact_data.get("fraud_rows", 0))

        # Headline Operating Table at Validation Thresholds
        st.markdown("#### Headline Operating Performance (Validation Thresholds)")
        def_cost_val = int(business_impact_data.get("default_review_cost", 10000))
        st.caption(
            f"Fixed at the validation threshold and review cost {def_cost_val:,} and does not follow the controls."
        )
        headline_rows = business_impact_data.get("headline_operating_table", [])
        if headline_rows:
            df_headline = pd.DataFrame([
                {
                    "Model": model_names.get(r.get("model_key", ""), r.get("model_key", "")),
                    "Threshold": f"{r.get('operating_threshold', 0.5):.4f}" if isinstance(r.get("operating_threshold"), float) else str(r.get("operating_threshold")),
                    "Source": r.get("threshold_source", "validation"),
                    "Total Alerts": f"{int(r.get('alerts', 0)):,}",
                    "True Alerts": f"{int(r.get('true_alerts', 0)):,}",
                    "False Alerts": f"{int(r.get('false_alerts', 0)):,}",
                    "Alerts/10k": f"{float(r.get('alerts_per_10k', 0.0)):.2f}",
                    "Fraud Caught": f"{float(r.get('fraud_value_caught', 0.0)):,.2f}",
                    "Value Caught %": f"{float(r.get('share_fraud_value_caught', 0.0)) * 100:.2f}%",
                    "Net Saving (units)": f"{float(r.get('net_financial_saving', 0.0)):,.2f}",
                }
                for r in headline_rows
            ])
            st.dataframe(df_headline, use_container_width=True, hide_index=True)

        if not impact_models:
            st.warning("No model threshold grids found in business_impact.json.")
        else:
            model_keys = list(impact_models.keys())
            c_sel1, c_sel2 = st.columns([2, 2])
            with c_sel1:
                selected_impact_model = st.selectbox(
                    "Select Model for Analysis",
                    options=model_keys,
                    format_func=lambda k: model_names.get(k, k),
                    key="impact_model_select",
                )
            with c_sel2:
                review_cost = st.number_input(
                    "Investigation Cost per Alert (simulated currency units, assumption)",
                    min_value=0.0,
                    value=float(business_impact_data.get("default_review_cost", 10000.0)),
                    step=1000.0,
                    help="Manual investigator review cost incurred per flagged transaction alert (default assumption: 10,000 units).",
                    key="impact_review_cost",
                )

            m_data = impact_models.get(selected_impact_model, {})
            grid = m_data.get("grid", [])

            if not grid:
                st.warning("Threshold grid is empty for the selected model.")
            else:
                # Find best threshold maximizing net saving = fraud_value_caught - review_cost * alerts (hindsight view)
                best_point = max(
                    grid,
                    key=lambda p: float(p.get("fraud_value_caught", 0.0)) - review_cost * float(p.get("alerts", 0)),
                )
                best_th = float(best_point.get("threshold", 0.5))
                best_caught = float(best_point.get("fraud_value_caught", 0.0))
                best_alerts = int(best_point.get("alerts", 0))
                best_net = best_caught - (review_cost * best_alerts)

                # Lookup validation threshold from evaluation.json (falling back to business_impact.json)
                val_th = None
                if evaluation_data and isinstance(evaluation_data, dict):
                    time_split_eval = evaluation_data.get("splits", {}).get("time", {})
                    if selected_impact_model in time_split_eval:
                        val_th = time_split_eval[selected_impact_model].get("threshold")
                    elif selected_impact_model in evaluation_data:
                        val_th = evaluation_data[selected_impact_model].get("threshold")
                if val_th is None:
                    val_th = m_data.get("validation_threshold", 0.5)
                val_th = float(val_th)

                # Ensure slider resets to the selected model's validation threshold on model change
                if st.session_state.get("_last_impact_model") != selected_impact_model:
                    st.session_state["_last_impact_model"] = selected_impact_model
                    st.session_state["impact_threshold_slider"] = float(round(val_th, 4))

                st.slider(
                    "Operating Score Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(round(val_th, 4)),
                    step=0.01,
                    key="impact_threshold_slider",
                )
                user_th = float(st.session_state.get("impact_threshold_slider", val_th))

                # Candidate points include the grid and exact headline validation point
                candidate_points = list(grid)
                headline_point = m_data.get("headline_operating_point")
                if headline_point and headline_point not in candidate_points:
                    candidate_points.append(headline_point)

                # Find closest point to chosen slider value
                chosen_point = min(candidate_points, key=lambda p: abs(float(p.get("threshold", 0.0)) - user_th))
                th_val = float(chosen_point.get("threshold", 0.0))
                alerts_val = int(chosen_point.get("alerts", 0))
                true_alerts_val = int(chosen_point.get("true_alerts", 0))
                false_alerts_val = int(chosen_point.get("false_alerts", 0))
                caught_val = float(chosen_point.get("fraud_value_caught", 0.0))
                missed_val = float(chosen_point.get("fraud_value_missed", 0.0))
                share_caught = float(chosen_point.get("share_fraud_value_caught", 0.0))
                alerts_10k = float(chosen_point.get("alerts_per_10k", 0.0))
                net_saving_val = caught_val - (review_cost * alerts_val)

                # Best operating threshold callout
                st.info(
                    f"🔍 **Hindsight on test rows, not a recommended operating point:** Operating at `threshold = {best_th:.4f}` maximizes net saving on test rows "
                    f"at review cost {review_cost:,.0f} (**{best_net:,.2f}** simulated currency units net saving, {best_alerts:,} alerts)."
                )

                # KPI Cards for chosen threshold
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                with kpi1:
                    render_metric_card(
                        "Total Alerts",
                        f"{alerts_val:,}",
                        f"{alerts_10k:.1f} per 10k rows",
                        ratio=alerts_val / max(1, int(business_impact_data.get("test_rows", 193939))),
                    )
                with kpi2:
                    render_metric_card(
                        "Fraud Value Caught",
                        f"{caught_val:,.0f}",
                        f"{share_caught * 100:.1f}% of total ({true_alerts_val:,} true alerts)",
                        ratio=share_caught,
                    )
                with kpi3:
                    render_metric_card(
                        "Fraud Value Missed",
                        f"{missed_val:,.0f}",
                        f"{(1.0 - share_caught) * 100:.1f}% missed",
                        ratio=1.0 - share_caught,
                    )
                with kpi4:
                    render_metric_card(
                        "Net Financial Saving",
                        f"{net_saving_val:,.0f}",
                        f"Caught minus ({review_cost:,.0f} × alerts)",
                    )

                # Side-by-side comparison table: Selected vs Best vs Rule Baseline
                st.markdown("#### Performance Comparison")
                rule_alerts = int(rule_baseline.get("alerts", 0))
                rule_true_alerts = int(rule_baseline.get("true_alerts", 0))
                rule_false_alerts = int(rule_baseline.get("false_alerts", 0))
                rule_caught = float(rule_baseline.get("fraud_value_caught", 0.0))
                rule_missed = float(rule_baseline.get("fraud_value_missed", 0.0))
                rule_share = float(rule_baseline.get("share_fraud_value_caught", 0.0))
                rule_net = rule_caught - (review_cost * rule_alerts)

                comp_data = {
                    "Metric": [
                        "Threshold",
                        "Alerts Triggered",
                        "True Alerts (Fraud)",
                        "False Alerts (Non-Fraud)",
                        "Alerts per 10k Rows",
                        "Fraud Value Caught (units)",
                        "Fraud Value Missed (units)",
                        "Share of Fraud Caught (%)",
                        "Net Saving (units)",
                    ],
                    f"Selected Point ({th_val:.4f})": [
                        f"{th_val:.4f}",
                        f"{alerts_val:,}",
                        f"{true_alerts_val:,}",
                        f"{false_alerts_val:,}",
                        f"{alerts_10k:.2f}",
                        f"{caught_val:,.2f}",
                        f"{missed_val:,.2f}",
                        f"{share_caught * 100:.2f}%",
                        f"{net_saving_val:,.2f}",
                    ],
                    f"Best Point ({best_th:.4f})": [
                        f"{best_th:.4f}",
                        f"{best_alerts:,}",
                        f"{int(best_point.get('true_alerts', 0)):,}",
                        f"{int(best_point.get('false_alerts', 0)):,}",
                        f"{float(best_point.get('alerts_per_10k', 0.0)):.2f}",
                        f"{best_caught:,.2f}",
                        f"{float(best_point.get('fraud_value_missed', 0.0)):,.2f}",
                        f"{float(best_point.get('share_fraud_value_caught', 0.0)) * 100:.2f}%",
                        f"{best_net:,.2f}",
                    ],
                    "Rule Baseline (isFlaggedFraud)": [
                        "Amount > 200k",
                        f"{rule_alerts:,}",
                        f"{rule_true_alerts:,}",
                        f"{rule_false_alerts:,}",
                        f"{float(rule_baseline.get('alerts_per_10k', 0.0)):.2f}",
                        f"{rule_caught:,.2f}",
                        f"{rule_missed:,.2f}",
                        f"{rule_share * 100:.2f}%",
                        f"{rule_net:,.2f}",
                    ],
                }
                st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

                # Plotly Net Saving Curve across Threshold Grid
                st.markdown("#### Net Saving vs Operating Threshold")
                th_x = [float(p.get("threshold", 0.0)) for p in grid]
                net_y = [
                    float(p.get("fraud_value_caught", 0.0)) - review_cost * float(p.get("alerts", 0))
                    for p in grid
                ]

                fig_impact = go.Figure()
                fig_impact.add_trace(
                    go.Scatter(
                        x=th_x,
                        y=net_y,
                        mode="lines",
                        name=f"Net Saving (Cost={review_cost:,.0f})",
                        line=dict(color="#38bdf8", width=3),
                        hoverinfo="text",
                        hovertext=[
                            f"Threshold: {x:.4f}<br>Net Saving: {y:,.2f}<br>Alerts: {int(p.get('alerts', 0)):,}"
                            for x, y, p in zip(th_x, net_y, grid)
                        ],
                    )
                )
                fig_impact.add_trace(
                    go.Scatter(
                        x=[best_th],
                        y=[best_net],
                        mode="markers+text",
                        name=f"Hindsight Test Best ({best_th:.4f}) [Not Recommended Operating Point]",
                        marker=dict(color="#10b981", size=12, symbol="star"),
                        text=[f"Best: {best_net:,.0f}"],
                        textposition="top center",
                    )
                )
                fig_impact.add_trace(
                    go.Scatter(
                        x=[th_val],
                        y=[net_saving_val],
                        mode="markers",
                        name=f"Current Slider ({th_val:.4f})",
                        marker=dict(color="#f59e0b", size=10, symbol="diamond"),
                    )
                )
                fig_impact.add_hline(
                    y=rule_net,
                    line_dash="dash",
                    line_color="#ef4444",
                    annotation_text=f"Rule Baseline Net ({rule_net:,.0f})",
                    annotation_position="bottom right",
                )

                fig_impact.update_layout(
                    title=dict(
                        text=f"Net Financial Saving Curve across Thresholds ({model_names.get(selected_impact_model, selected_impact_model)})",
                        font=dict(size=14),
                    ),
                    xaxis=dict(title="Score Threshold", showgrid=True, gridcolor="rgba(148,163,184,0.15)"),
                    yaxis=dict(
                        title="Simulated Currency Units",
                        showgrid=True,
                        gridcolor="rgba(148,163,184,0.15)",
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#071a17",
                    font=dict(color="#edf7f4"),
                    margin=dict(l=40, r=40, t=50, b=40),
                    height=450,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_impact, use_container_width=True)

        # Sensitivity Table across review costs 1,000 / 10,000 / 100,000
        sens_rows = business_impact_data.get("sensitivity_table", [])
        if sens_rows:
            sens_costs = business_impact_data.get("sensitivity_review_costs", [1000.0, 10000.0, 100000.0])
            sens_costs_str = " / ".join(f"{int(c):,}" for c in sens_costs)
            st.markdown(f"#### Review Cost Sensitivity Analysis ({sens_costs_str} units)")
            st.caption(
                f"Fixed at review costs {sens_costs_str} and does not follow the controls."
            )
            df_sens = pd.DataFrame([
                {
                    "Review Cost": f"{float(s.get('review_cost', 0.0)):,.0f}",
                    "Model": model_names.get(s.get("model_key", ""), s.get("model_key", "")),
                    "Mode": s.get("mode", ""),
                    "Threshold": f"{float(s.get('threshold', 0.0)):.4f}" if isinstance(s.get("threshold"), (int, float)) else str(s.get("threshold")),
                    "Alerts": f"{int(s.get('alerts', 0)):,}",
                    "Fraud Caught": f"{float(s.get('fraud_value_caught', 0.0)):,.2f}",
                    "Net Saving (units)": f"{float(s.get('net_financial_saving', 0.0)):,.2f}",
                }
                for s in sens_rows
            ])
            st.dataframe(df_sens, use_container_width=True, hide_index=True)

        with st.expander("Raw file: business_impact.json", expanded=False):
            st.json(business_impact_data)


# =========================================================================
# TAB 5: FRAUD RINGS
# =========================================================================
with tab_rings:
    st.subheader("Fraud Ring Analysis & Topology")
    st.markdown(
        "Topological graph analysis of fraud components and planted synthetic ring detection. "
        "Evaluated on ground-truth fraud transactions from the PaySim dataset."
    )

    if not ring_analysis_data:
        st.info(
            f"Fraud ring analysis not available. Run `python rings.py --output-dir {OUTPUT_DIR}` to generate."
        )
    else:
        finding_text = ring_analysis_data.get("finding", "")
        if finding_text:
            st.info(f"📋 **Empirical Finding:** {finding_text}")

        rkpi1, rkpi2, rkpi3, rkpi4 = st.columns(4)
        with rkpi1:
            render_metric_card(
                "Fraud Accounts",
                f"{ring_analysis_data.get('number_of_fraud_accounts', 0):,}",
                "Distinct nodes in fraud subgraph",
            )
        with rkpi2:
            render_metric_card(
                "Fraud Components",
                f"{ring_analysis_data.get('number_of_components', 0):,}",
                "Connected components",
            )
        with rkpi3:
            render_metric_card(
                "Components (Size 2)",
                f"{ring_analysis_data.get('number_of_components_by_size', {}).get('2', 0):,}",
                "Isolated 1-to-1 account pairs",
            )
        with rkpi4:
            render_metric_card(
                "Components (Size ≥ 3)",
                f"{ring_analysis_data.get('count_size_ge_3', 0):,}",
                "All size 3 (2-to-1 funnels)",
            )

        st.markdown("#### Real Exported Fraud Rings (Size ≥ 3)")
        rings_dir = OUTPUT_DIR / "rings"
        available_ring_files = list_available_rings(str(rings_dir), get_file_mtime(rings_dir))

        if not available_ring_files:
            st.caption(f"No individual ring JSON files found in `{rings_dir}`.")
        else:
            ring_options = {}
            for r_file in available_ring_files:
                r_path = rings_dir / r_file
                r_data = load_ring_file(str(r_path), get_file_mtime(r_path))
                if r_data:
                    label = (
                        f"Ring {r_data.get('ring_id', r_file)} — "
                        f"Amount: {float(r_data.get('total_amount', 0.0)):,.2f} | "
                        f"Nodes: {r_data.get('size', 3)} | "
                        f"Txns: {r_data.get('transaction_count', 2)}"
                    )
                    ring_options[label] = (r_file, r_data)

            chosen_ring_label = st.selectbox(
                "Select Real Fraud Ring to Inspect",
                options=list(ring_options.keys()),
                key="real_ring_select",
            )

            if chosen_ring_label:
                _, chosen_ring = ring_options[chosen_ring_label]

                G_ring = nx.DiGraph()
                nodes_list = chosen_ring.get("nodes", [])
                for nd in nodes_list:
                    G_ring.add_node(nd["id"], role=nd.get("role", "unknown"))

                edges_list = chosen_ring.get("edges", [])
                for ed in edges_list:
                    G_ring.add_edge(
                        ed["src"],
                        ed["dst"],
                        amount=float(ed.get("amount", 0.0)),
                        step=int(ed.get("step", 0)),
                        type=str(ed.get("type", "CASH_OUT")),
                    )

                pos = nx.circular_layout(G_ring) if len(G_ring.nodes) <= 4 else nx.spring_layout(G_ring, seed=42)

                edge_x, edge_y = [], []
                for u, v, d in G_ring.edges(data=True):
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

                trace_ring_edges = go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    line=dict(width=2.5, color="#ef4444"),
                    hoverinfo="none",
                    mode="lines",
                    name="Fraud Transfer Edge",
                )

                node_x, node_y, node_colors, node_texts, node_sizes = [], [], [], [], []
                for nid in G_ring.nodes():
                    x, y = pos[nid]
                    node_x.append(x)
                    node_y.append(y)
                    role = G_ring.nodes[nid].get("role", "unknown")

                    if role == "origin":
                        node_colors.append("#38bdf8")  # Sky blue
                        node_sizes.append(22)
                    elif role == "destination":
                        node_colors.append("#ef4444")  # Coral / Red
                        node_sizes.append(24)
                    else:
                        node_colors.append("#a855f7")  # Purple
                        node_sizes.append(26)

                    in_d = G_ring.in_degree(nid)
                    out_d = G_ring.out_degree(nid)
                    node_texts.append(
                        f"Account: <b>{nid}</b><br>Role: <b>{role.upper()}</b><br>In-degree: {in_d} | Out-degree: {out_d}"
                    )

                trace_ring_nodes = go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode="markers+text",
                    hoverinfo="text",
                    text=[nid[:8] + ".." for nid in G_ring.nodes()],
                    textposition="bottom center",
                    textfont=dict(color="#ffffff", size=10),
                    hovertext=node_texts,
                    marker=dict(
                        color=node_colors,
                        size=node_sizes,
                        line=dict(width=2, color="#ffffff"),
                    ),
                    name="Accounts (Blue=Origin, Red=Destination, Purple=Both)",
                )

                fig_ring = go.Figure(
                    data=[trace_ring_edges, trace_ring_nodes],
                    layout=go.Layout(
                        title=dict(
                            text=f"Directed Topology: Ring {chosen_ring.get('ring_id')} (Total Amount: {float(chosen_ring.get('total_amount', 0)):,.2f} simulated currency units)",
                            font=dict(size=14),
                        ),
                        showlegend=True,
                        hovermode="closest",
                        margin=dict(b=20, l=20, r=20, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=420,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="#071a17",
                        font=dict(color="#edf7f4"),
                    ),
                )
                st.plotly_chart(fig_ring, use_container_width=True)

                st.markdown("**Ring Transactions:**")
                df_edges = pd.DataFrame(edges_list)
                if not df_edges.empty:
                    df_edges["amount"] = df_edges["amount"].apply(lambda v: f"{v:,.2f}")
                    st.dataframe(df_edges, use_container_width=True, hide_index=True)

        rings_summary_path = OUTPUT_DIR / "rings_summary.csv"
        df_rings_summary = load_rings_summary_csv(str(rings_summary_path), get_file_mtime(rings_summary_path))
        if df_rings_summary is not None and not df_rings_summary.empty:
            with st.expander("Full Real Fraud Rings Table (Components with Size ≥ 3)", expanded=False):
                st.dataframe(df_rings_summary[df_rings_summary["size"] >= 3], use_container_width=True)

    # Planted Ring Results Table (Marked as synthetic injection for validation)
    st.divider()
    st.markdown("#### Planted-Ring Benchmark (Synthetic Injection for Validation)")
    st.caption(
        "Evaluation of Spark-native Connected Components and LPA graph clustering on 60 synthetic rings "
        "(chains, funnels, cycles) injected into a 200,000-row PaySim sample at 3 camouflage levels."
    )

    if not planted_rings_data:
        st.info(
            f"Planted-ring validation not available. Run `python planted_rings.py --output-dir {OUTPUT_DIR}` to generate."
        )
    else:
        algo_choice = st.radio(
            "Select Community Detection Algorithm",
            options=["Connected Components", "Label Propagation (LPA)"],
            horizontal=True,
            key="planted_algo_radio",
        )

        algo_key = "connected_components" if "Connected" in algo_choice else "label_propagation"
        algo_data = planted_rings_data.get("algorithms", {}).get(algo_key, {})
        results_list = algo_data.get("results_by_pattern_and_camouflage", [])
        if not results_list and algo_key == "connected_components":
            results_list = planted_rings_data.get("results_by_pattern_and_camouflage", [])
        elif not results_list and algo_key == "label_propagation":
            results_list = planted_rings_data.get("results_by_pattern_and_camouflage_lpa", [])

        if results_list:
            df_planted = pd.DataFrame(results_list)
            df_display = pd.DataFrame({
                "Pattern": df_planted["pattern"].str.capitalize(),
                "Camouflage": df_planted["camouflage"],
                "Planted": df_planted["planted"],
                "Strict Recovered": df_planted["recovered"],
                "Strict Rec Rate": (df_planted["recovery_rate"] * 100).apply(lambda v: f"{v:.1f}%"),
                "Ring Contained Rate": (df_planted.get("ring_contained_rate", df_planted["recovery_rate"]) * 100).apply(lambda v: f"{v:.1f}%"),
                "Mean Ring Purity": (df_planted.get("mean_ring_purity", 1.0 - df_planted["contamination"]) * 100).apply(lambda v: f"{v:.2f}%"),
                "Largest Comp with Planted": df_planted.get("largest_component_size_with_planted", 0).apply(lambda v: f"{int(v):,}"),
                "Homogeneity": df_planted["homogeneity"].apply(lambda v: f"{v:.4f}"),
                "Completeness": df_planted["completeness"].apply(lambda v: f"{v:.4f}"),
                "V-Measure": df_planted["v_measure"].apply(lambda v: f"{v:.4f}"),
                "Contamination": (df_planted["contamination"] * 100).apply(lambda v: f"{v:.1f}%"),
            })
            st.dataframe(df_display, use_container_width=True, hide_index=True)

            if algo_key == "connected_components":
                st.caption(
                    "📌 **Connected Components Observations:** 100.0% of planted ring accounts fall into a single detected component across all patterns and camouflage levels (`ring_contained_rate = 1.0000`). "
                    "At 0% camouflage, strict recovery is 100% with 0.0% contamination. With camouflage transactions (20% and 50%), bridging edges add background accounts into the component, "
                    "reducing ring purity to 16.7%–38.1% and increasing contamination to 61.9%–83.3%. "
                    "The largest detected component containing planted accounts has 116 accounts (funnel 50%), 47 accounts (chain 50%), and 44 accounts (cycle 50%)."
                )
            else:
                st.caption(
                    "📌 **Label Propagation (LPA) Observations:** At 0% camouflage, LPA recovers 11/21 rings (cycle 4/7, funnel 7/7). "
                    "At 20% and 50% camouflage, LPA limits contamination to 0.0%–55.0% and bounds component sizes tightly "
                    "(largest component containing planted accounts is 7–18 accounts), though chains tend to fragment along directed hops."
                )

    with st.expander("Explore Raw Dataset Sample Subgraph (from CSV)", expanded=False):
        data_dir = Path("data")
        csv_candidates = sorted(data_dir.glob("*.csv")) + sorted(data_dir.glob("*/*.csv"))
        csv_path = csv_candidates[0] if csv_candidates else None

        if csv_path is None:
            st.info("No PaySim CSV dataset found in `data/`.")
        else:
            sample_df = load_fraud_sample_transactions(str(csv_path), get_file_mtime(csv_path))
            if not sample_df.empty:
                max_render = st.slider("Max Transactions to Render", min_value=10, max_value=100, value=30, step=10, key="raw_sample_render_slider")
                sub_df = sample_df[sample_df["isFraud"] == 1].head(max_render)
                st.dataframe(sub_df[["step", "type", "nameOrig", "nameDest", "amount", "isFraud"]], use_container_width=True)

    with st.expander("Raw file: ring_analysis.json & planted_ring_validation.json", expanded=False):
        if ring_analysis_data:
            st.markdown("**ring_analysis.json:**")
            st.json(ring_analysis_data)
        if planted_rings_data:
            st.markdown("**planted_ring_validation.json:**")
            st.json(planted_rings_data)


# =========================================================================
# TAB 6: SCALING
# =========================================================================
with tab_scaling:
    st.subheader("Distributed Systems Scaling & Storage Benchmarks")
    if not any([scaling_runs_data, pandas_vs_spark_data, storage_benchmark_data]):
        st.info(
            "No benchmark outputs found. Run `python storage_benchmark.py`, "
            "`python benchmark_scaling.py`, and `python pandas_vs_spark.py` to generate results."
        )
    else:
        st.markdown(
            """
            <div class="caveat-box">
                <b>Measurement Caveats:</b> All benchmarks were executed on a single Windows 11 laptop
                (15.37 GiB shared RAM, Intel Core Ultra 7 155H with mixed performance and efficiency cores,
                local NTFS disk) using default Spark settings (<code>spark.driver.memory=6g</code>,
                <code>spark.sql.shuffle.partitions=48</code>). Uniform transaction sampling (seed=42)
                changes graph density across sample fractions; node/edge counts, individual repeat values,
                and any timeout or low-memory guard aborts are shown alongside means.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---- 1. Data-Size & Core Scaling ----
        if scaling_runs_data and isinstance(scaling_runs_data, dict):
            ds_summary = scaling_runs_data.get("data_size_summary", [])
            cs_summary = scaling_runs_data.get("core_scaling_summary", [])
            all_runs = scaling_runs_data.get("runs", [])
            non_ok_scale = scaling_runs_data.get("non_ok_runs", [])
            superseded_scale = scaling_runs_data.get("superseded_runs", [])

            st.markdown("### 1. Spark Data-Size Scaling (Log-Log)")
            if ds_summary:
                ds_rows = []
                for row in ds_summary:
                    indiv = ", ".join(f"{v:.2f}s" for v in row.get("individual_total_seconds", [])) or "none"
                    ds_rows.append(
                        {
                            "Fraction": row.get("fraction"),
                            "OK / Total": f"{row.get('ok_count', 0)}/{row.get('total_count', 0)}",
                            "Statuses": ", ".join(row.get("statuses", [])),
                            "Rows": int(row["mean_rows"]) if row.get("mean_rows") is not None else "n/a",
                            "Nodes": int(row["mean_nodes"]) if row.get("mean_nodes") is not None else "n/a",
                            "Edges": int(row["mean_edges"]) if row.get("mean_edges") is not None else "n/a",
                            "Mean Total (s)": f"{row['mean_total_seconds']:.2f}" if row.get("mean_total_seconds") is not None else "n/a",
                            "Individual Repeats": indiv,
                            "Mean Peak RSS": format_bytes(row.get("mean_peak_rss_bytes")),
                            "Min Avail RAM": format_bytes(row.get("min_available_ram_bytes")),
                            "Mean Shuffle": format_bytes(row.get("mean_shuffle_bytes")),
                        }
                    )
                st.dataframe(pd.DataFrame(ds_rows), use_container_width=True)

                ok_ds_pts = [
                    {
                        "Fraction": float(r["fraction"]),
                        "Total Seconds": float((r.get("seconds") or {}).get("total", r.get("wall_seconds", 0))),
                        "Repeat": f"Repeat {r.get('repeat')}",
                    }
                    for r in all_runs
                    if r.get("suite") == "data_scaling"
                    and r.get("status") == "ok"
                    and not r.get("superseded")
                ]
                if ok_ds_pts:
                    df_ds_plot = pd.DataFrame(ok_ds_pts)
                    fig_ds = px.scatter(
                        df_ds_plot,
                        x="Fraction",
                        y="Total Seconds",
                        color="Repeat",
                        log_x=True,
                        log_y=True,
                        title="Graph-Stage Runtime vs Sample Fraction (Log-Log, Individual Repeats)",
                        height=360,
                    )
                    mean_pts = [r for r in ds_summary if r.get("mean_total_seconds") is not None]
                    if mean_pts:
                        fig_ds.add_trace(
                            go.Scatter(
                                x=[float(r["fraction"]) for r in mean_pts],
                                y=[float(r["mean_total_seconds"]) for r in mean_pts],
                                mode="lines+markers",
                                name="Mean (ok)",
                                line=dict(color="#38bdf8", width=2.5),
                            )
                        )
                    fig_ds.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(17, 26, 24, 0.6)",
                        font=dict(color="#edf7f4"),
                    )
                    st.plotly_chart(fig_ds, use_container_width=True)

            st.markdown("### 2. Spark Core-Scaling Speedup")
            if cs_summary:
                core_frac = scaling_runs_data.get("core_scaling_fraction", 0.10)
                st.caption(
                    f"Measured at sample fraction `{core_frac}` across `local[1..16]` (2 repeats each); "
                    "speedup = mean_time(local[1]) / mean_time(local[n])."
                )
                cs_rows = []
                for row in cs_summary:
                    indiv_t = ", ".join(f"{v:.2f}s" for v in row.get("individual_total_seconds", [])) or "none"
                    indiv_sp = ", ".join(f"{v:.2f}x" for v in row.get("individual_speedups", [])) or "none"
                    cs_rows.append(
                        {
                            "Master": row.get("master"),
                            "Cores": row.get("cores"),
                            "Statuses": ", ".join(row.get("statuses", [])),
                            "Mean Total (s)": f"{row['mean_total_seconds']:.2f}" if row.get("mean_total_seconds") is not None else "n/a",
                            "Individual Repeats (s)": indiv_t,
                            "Speedup": f"{row['speedup']:.3f}x" if row.get("speedup") is not None else "n/a",
                            "Individual Speedups": indiv_sp,
                            "Efficiency": f"{row['efficiency']:.3f}" if row.get("efficiency") is not None else "n/a",
                            "Mean Peak RSS": format_bytes(row.get("mean_peak_rss_bytes")),
                            "Min Avail RAM": format_bytes(row.get("min_available_ram_bytes")),
                        }
                    )
                st.dataframe(pd.DataFrame(cs_rows), use_container_width=True)

                df_cs_plot = pd.DataFrame(
                    [
                        {
                            "Cores": int(r["cores"]),
                            "Speedup": float(r["speedup"]),
                            "Mean Seconds": float(r["mean_total_seconds"]),
                        }
                        for r in cs_summary
                        if r.get("speedup") is not None
                    ]
                )
                if not df_cs_plot.empty:
                    fig_cs = px.line(
                        df_cs_plot,
                        x="Cores",
                        y="Speedup",
                        markers=True,
                        title=f"Speedup vs Spark Local Threads (Fraction = {core_frac:.2f})",
                        height=340,
                    )
                    fig_cs.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(17, 26, 24, 0.6)",
                        font=dict(color="#edf7f4"),
                    )
                    st.plotly_chart(fig_cs, use_container_width=True)

            if non_ok_scale:
                with st.expander(f"Active Non-OK Scaling Runs Recorded ({len(non_ok_scale)})", expanded=True):
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "Suite": r.get("suite"),
                                    "Fraction": r.get("fraction"),
                                    "Master": r.get("master"),
                                    "Repeat": r.get("repeat"),
                                    "Status": r.get("status"),
                                    "Wall (s)": r.get("wall_seconds"),
                                    "Peak RSS": format_bytes(r.get("peak_rss_bytes")),
                                    "Min Avail RAM": format_bytes(r.get("min_available_ram_bytes")),
                                    "Reason": r.get("reason", ""),
                                }
                                for r in non_ok_scale
                            ]
                        ),
                        use_container_width=True,
                    )
            if superseded_scale:
                with st.expander(f"Superseded Prior Scaling Attempts Retained ({len(superseded_scale)})", expanded=False):
                    st.caption(
                        "Original aborted_low_memory attempts (< 0.5 GiB for 60s guard) retained in scaling_runs.jsonl and scaling_runs.json; "
                        "re-run with < 0.2 GiB for 120s guard."
                    )
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "Suite": r.get("suite"),
                                    "Fraction": r.get("fraction"),
                                    "Master": r.get("master"),
                                    "Repeat": r.get("repeat"),
                                    "Status": r.get("status"),
                                    "Wall (s)": r.get("wall_seconds"),
                                    "Peak RSS": format_bytes(r.get("peak_rss_bytes")),
                                    "Min Avail RAM": format_bytes(r.get("min_available_ram_bytes")),
                                    "Note": r.get("superseded_note") or r.get("reason", ""),
                                }
                                for r in superseded_scale
                            ]
                        ),
                        use_container_width=True,
                    )
        elif scaling_runs_data and isinstance(scaling_runs_data, list):
            st.dataframe(pd.DataFrame(scaling_runs_data), use_container_width=True)

        # ---- 2. pandas/NetworkX vs Spark ----
        st.markdown("### 3. pandas/NetworkX vs Spark-Native Comparison")
        if not pandas_vs_spark_data:
            st.info("Run `python pandas_vs_spark.py` to populate the single-node vs Spark comparison.")
        else:
            pvs_summary = pandas_vs_spark_data.get("summary", [])
            if pvs_summary:
                pvs_rows = []
                for row in pvs_summary:
                    indiv_c = ", ".join(f"{v:.3f}s" for v in row.get("individual_compute_seconds", [])) or "none"
                    pvs_rows.append(
                        {
                            "Task": row.get("task"),
                            "Tool": row.get("tool"),
                            "Rows": int(row.get("rows", 0)),
                            "Statuses": ", ".join(row.get("statuses", [])),
                            "Load Mean (s)": f"{row['mean_load_seconds']:.3f}" if row.get("mean_load_seconds") is not None else "n/a",
                            "Compute Mean (s)": f"{row['mean_compute_seconds']:.3f}" if row.get("mean_compute_seconds") is not None else "n/a",
                            "Compute Repeats": indiv_c,
                            "Peak RSS": format_bytes(row.get("mean_peak_rss_bytes") or row.get("max_peak_rss_bytes")),
                            "Min Avail RAM": format_bytes(row.get("min_available_ram_bytes")),
                        }
                    )
                st.dataframe(pd.DataFrame(pvs_rows), use_container_width=True)

                df_pvs_ok = pd.DataFrame(
                    [
                        {
                            "Rows": int(r["rows"]),
                            "Series": f"{r['tool']} ({r['task']})",
                            "Compute Seconds": float(r["mean_compute_seconds"]),
                            "Peak RSS (GiB)": float(r["mean_peak_rss_bytes"]) / (1024 ** 3),
                        }
                        for r in pvs_summary
                        if r.get("mean_compute_seconds") is not None
                    ]
                )
                if not df_pvs_ok.empty:
                    c1, c2 = st.columns(2)
                    with c1:
                        fig_pvs_t = px.line(
                            df_pvs_ok,
                            x="Rows",
                            y="Compute Seconds",
                            color="Series",
                            markers=True,
                            log_x=True,
                            log_y=True,
                            title="Compute Time vs Rows (Log-Log)",
                            height=350,
                        )
                        fig_pvs_t.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(17, 26, 24, 0.6)",
                            font=dict(color="#edf7f4"),
                        )
                        st.plotly_chart(fig_pvs_t, use_container_width=True)
                    with c2:
                        fig_pvs_m = px.line(
                            df_pvs_ok,
                            x="Rows",
                            y="Peak RSS (GiB)",
                            color="Series",
                            markers=True,
                            log_x=True,
                            title="Peak Process-Tree RSS (GiB)",
                            height=350,
                        )
                        fig_pvs_m.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(17, 26, 24, 0.6)",
                            font=dict(color="#edf7f4"),
                        )
                        st.plotly_chart(fig_pvs_m, use_container_width=True)

        # ---- 3. Storage Benchmark: CSV vs Parquet ----
        st.markdown("### 4. Storage Benchmark: CSV vs Parquet")
        if not storage_benchmark_data:
            st.info("Run `python storage_benchmark.py` to populate CSV vs Parquet measurements.")
        else:
            fmts = storage_benchmark_data.get("formats", {})
            if fmts:
                st_rows = []
                chart_rows = []
                for fmt_key, fmt_val in fmts.items():
                    ops = fmt_val.get("operations", {})
                    st_rows.append(
                        {
                            "Format": fmt_key,
                            "Size (MiB)": fmt_val.get("size_mb"),
                            "Compression vs CSV": f"{fmt_val.get('compression_ratio_vs_csv', 1.0):.2f}x",
                            "Count Mean (s)": ops.get("count_rows", {}).get("mean_seconds"),
                            "Count Repeats": ", ".join(str(x) for x in ops.get("count_rows", {}).get("repeats_seconds", [])),
                            "2-Col Read Mean (s)": ops.get("read_two_columns", {}).get("mean_seconds"),
                            "2-Col Repeats": ", ".join(str(x) for x in ops.get("read_two_columns", {}).get("repeats_seconds", [])),
                            "Filter Mean (s)": ops.get("filter_transfer_fraud", {}).get("mean_seconds"),
                            "Filter Repeats": ", ".join(str(x) for x in ops.get("filter_transfer_fraud", {}).get("repeats_seconds", [])),
                        }
                    )
                    for op_k, op_lbl in [
                        ("count_rows", "Count Rows"),
                        ("read_two_columns", "Read 2 Columns"),
                        ("filter_transfer_fraud", "Filter TRANSFER + Fraud"),
                    ]:
                        chart_rows.append(
                            {
                                "Format": fmt_key,
                                "Operation": op_lbl,
                                "Mean Seconds": ops.get(op_k, {}).get("mean_seconds", 0.0),
                            }
                        )
                st.dataframe(pd.DataFrame(st_rows), use_container_width=True)
                fig_st = px.bar(
                    pd.DataFrame(chart_rows),
                    x="Operation",
                    y="Mean Seconds",
                    color="Format",
                    barmode="group",
                    text_auto=".2f",
                    title="CSV vs Parquet Mean Query Latency (3 Repeats)",
                    height=330,
                )
                fig_st.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(17, 26, 24, 0.6)",
                    font=dict(color="#edf7f4"),
                )
                st.plotly_chart(fig_st, use_container_width=True)

        with st.expander("Raw Benchmark JSON Files"):
            if scaling_runs_data:
                st.markdown("**scaling_runs.json**")
                st.json(scaling_runs_data)
            if pandas_vs_spark_data:
                st.markdown("**pandas_vs_spark.json**")
                st.json(pandas_vs_spark_data)
            if storage_benchmark_data:
                st.markdown("**storage_benchmark.json**")
                st.json(storage_benchmark_data)


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

        def apply_scorer_preset(name: str) -> None:
            st.session_state["sc_selected_preset"] = name
            if name == "Normal Payment":
                st.session_state["sc_type"] = "PAYMENT"
                st.session_state["sc_step"] = 14
                st.session_state["sc_amt"] = 85.0
                st.session_state["sc_old_orig"] = 1200.0
                st.session_state["sc_new_orig"] = 1115.0
                st.session_state["sc_old_dest"] = 350.0
                st.session_state["sc_new_dest"] = 435.0
            elif name == "Suspicious Cash-Out":
                st.session_state["sc_type"] = "CASH_OUT"
                st.session_state["sc_step"] = 48
                st.session_state["sc_amt"] = 320000.0
                st.session_state["sc_old_orig"] = 320000.0
                st.session_state["sc_new_orig"] = 0.0
                st.session_state["sc_old_dest"] = 0.0
                st.session_state["sc_new_dest"] = 320000.0
            elif name == "Borderline Transfer":
                st.session_state["sc_type"] = "TRANSFER"
                st.session_state["sc_step"] = 92
                st.session_state["sc_amt"] = 18500.0
                st.session_state["sc_old_orig"] = 25000.0
                st.session_state["sc_new_orig"] = 6500.0
                st.session_state["sc_old_dest"] = 5000.0
                st.session_state["sc_new_dest"] = 23500.0

        selected_preset = st.session_state.get("sc_selected_preset", None)

        p_col1, p_col2, p_col3 = st.columns(3)
        b_normal = p_col1.button(
            "🟢 Normal Payment",
            type="primary" if selected_preset == "Normal Payment" else "secondary",
            use_container_width=True,
            on_click=apply_scorer_preset,
            args=("Normal Payment",),
        )
        b_suspicious = p_col2.button(
            "🔴 Suspicious Cash-Out",
            type="primary" if selected_preset == "Suspicious Cash-Out" else "secondary",
            use_container_width=True,
            on_click=apply_scorer_preset,
            args=("Suspicious Cash-Out",),
        )
        b_borderline = p_col3.button(
            "🟡 Borderline Transfer",
            type="primary" if selected_preset == "Borderline Transfer" else "secondary",
            use_container_width=True,
            on_click=apply_scorer_preset,
            args=("Borderline Transfer",),
        )

        if b_normal:
            apply_scorer_preset("Normal Payment")
        elif b_suspicious:
            apply_scorer_preset("Suspicious Cash-Out")
        elif b_borderline:
            apply_scorer_preset("Borderline Transfer")

        selected_preset = st.session_state.get("sc_selected_preset", None)
        st.caption(f"Selected preset: {selected_preset or 'None'}")

        with st.form("live_scoring_form"):
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                cur_type = st.session_state.get("sc_type", "CASH_OUT")
                type_idx = ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"].index(cur_type) if cur_type in ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"] else 0
                in_type = st.selectbox("Transaction Type", ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"], index=type_idx)
                in_step = st.number_input("Time Step (Hour 1-744)", min_value=1, max_value=744, value=int(st.session_state.get("sc_step", 45)))
                in_amt = st.number_input("Transaction Amount (simulated currency units)", min_value=0.01, value=float(st.session_state.get("sc_amt", 250000.0)), step=1000.0)
            with col_f2:
                in_old_orig = st.number_input("Sender Old Balance (simulated currency units)", min_value=0.0, value=float(st.session_state.get("sc_old_orig", 250000.0)), step=1000.0)
                in_new_orig = st.number_input("Sender New Balance (simulated currency units)", min_value=0.0, value=float(st.session_state.get("sc_new_orig", 0.0)), step=1000.0)
                in_old_dest = st.number_input("Receiver Old Balance (simulated currency units)", min_value=0.0, value=float(st.session_state.get("sc_old_dest", 0.0)), step=1000.0)
                in_new_dest = st.number_input("Receiver New Balance (simulated currency units)", min_value=0.0, value=float(st.session_state.get("sc_new_dest", 250000.0)), step=1000.0)

            score_submitted = st.form_submit_button("⚡ Score Transaction Risk")

        if score_submitted:
            err_orig = in_new_orig + in_amt - in_old_orig
            err_dest = in_old_dest + in_amt - in_new_dest

            type_cols = ["type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]
            type_vec = [1 if in_type == col.replace("type_", "") else 0 for col in type_cols]

            expected_features = getattr(model_obj, "feature_names_in_", None)

            # Construct input features dictionary
            feat_dict: Dict[str, Any] = {
                "type": in_type,
                "amount": in_amt,
                "oldbalanceOrg": in_old_orig,
                "newbalanceOrig": in_new_orig,
                "oldbalanceDest": in_old_dest,
                "newbalanceDest": in_new_dest,
                "errorBalanceOrig": err_orig,
                "errorBalanceDest": err_dest,
                "is_merchant": 1.0 if in_type == "PAYMENT" else 0.0,
                "step": in_step,
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
                    if hasattr(model_obj, "named_steps"):
                        try:
                            prep = model_obj.named_steps.get("preprocessing")
                            clf = model_obj.named_steps.get("classifier")
                            if clf and hasattr(clf, "get_booster") and prep:
                                booster = clf.get_booster()
                                trans_arr = prep.transform(aligned_df)
                                feature_names_out = [
                                    f.replace("numeric__", "").replace("type__", "")
                                    for f in prep.get_feature_names_out()
                                ]
                                dmat = xgb.DMatrix(trans_arr, feature_names=feature_names_out)
                                contribs = booster.predict(dmat, pred_contribs=True)
                                shap_vals = contribs[0][:-1]
                                contrib_records = [
                                    {
                                        "Feature": col,
                                        "Contribution (log-odds)": round(float(v), 4),
                                        "Impact": "Increases Fraud Risk" if v > 0 else "Decreases Fraud Risk",
                                        "Magnitude": abs(float(v)),
                                    }
                                    for col, v in zip(feature_names_out, shap_vals)
                                ]
                                df_contrib = (
                                    pd.DataFrame(contrib_records)
                                    .sort_values(by="Magnitude", ascending=False)
                                    .drop(columns=["Magnitude"])
                                )
                                st.write("**Top Feature Contributions (Measured SHAP Margins):**")
                                st.dataframe(df_contrib.head(6), use_container_width=True)
                        except Exception as ex:
                            st.caption(f"Feature contributions not computed: {ex}")
                    elif hasattr(model_obj, "get_booster"):
                        try:
                            booster = model_obj.get_booster()
                            dmat = xgb.DMatrix(aligned_df)
                            contribs = booster.predict(dmat, pred_contribs=True)
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
                            df_contrib = (
                                pd.DataFrame(contrib_records)
                                .sort_values(by="Magnitude", ascending=False)
                                .drop(columns=["Magnitude"])
                            )
                            st.write("**Top Feature Contributions (Measured SHAP Margins):**")
                            st.dataframe(df_contrib.head(6), use_container_width=True)
                        except Exception as ex:
                            st.caption(f"Feature contributions not computed: {ex}")

            except Exception as e:
                st.error(f"Scoring error: {e}")


st.divider()
st.caption("Financial Fraud Ring Analytics - PaySim Big Data Pipeline • Strictly honest data layer")
