"""
generate_visuals.py
Generates presentation-grade visual artifacts directly from accepted pipeline outputs in outputs/.
Saves figures to outputs/visuals/:
  - pr_curves.png
  - confusion_matrices.png (time split)
  - feature_importance.png
  - business_impact.png
  - ring_size_histogram.png
  - scaling_benchmark.png (when scaling_runs.json exists)

Strictly honest data layer: reads only accepted outputs from the designated output directory.
If required output files are missing, reports: "Required output files not found. Run python main.py."
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# Set writable MPLCONFIGDIR to prevent sandbox / permission warnings
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_cache"))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Clean publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 1.0,
    "grid.color": "#e5e5e5",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "figure.autolayout": True,
})

MODEL_DISPLAY_NAMES = {
    "m0_rule": "M0 Rule Baseline",
    "m1_tabular": "M1 Tabular Baseline",
    "m1_tabular_noerr": "M1 Tabular (No Error Bal)",
    "m2_graph": "M2 Graph + Structural",
    "m3_community_mule": "M3 Community + Mule",
}


def plot_pr_curves(pr_csv_path: Path, output_path: Path) -> bool:
    """Plot Precision-Recall curves for all models evaluated on the time split."""
    if not pr_csv_path.exists():
        print(f"[-] Missing {pr_csv_path}. Run python main.py.")
        return False

    try:
        df = pd.read_csv(pr_csv_path)
    except Exception as e:
        print(f"[-] Error reading {pr_csv_path}: {e}. Run python main.py.")
        return False

    if "split" in df.columns:
        df = df[df["split"] == "time"]

    if df.empty or "recall" not in df.columns or "precision" not in df.columns:
        print(f"[-] No time-split PR curve data found in {pr_csv_path}. Run python main.py.")
        return False

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    colors = {
        "m0_rule": "#94a3b8",
        "m1_tabular": "#38bdf8",
        "m1_tabular_noerr": "#fbbf24",
        "m2_graph": "#34d399",
        "m3_community_mule": "#a855f7",
    }

    models_found = df["model"].unique()
    for m in models_found:
        m_df = df[df["model"] == m].sort_values(by="recall")
        label = MODEL_DISPLAY_NAMES.get(m, m.replace("_", " ").title())
        color = colors.get(m, "#4f46e5")
        ax.plot(m_df["recall"], m_df["precision"], label=label, color=color, linewidth=2.2)

    ax.set_xlabel("Recall (Share of Ground-Truth Fraud Flagged)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision (True Fraud / Flagged Alerts)", fontsize=11, fontweight="bold")
    ax.set_title("Precision-Recall Curves (Time Split Test Set)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved PR curves: {output_path}")
    return True


def plot_confusion_matrices(eval_json_path: Path, output_path: Path) -> bool:
    """Plot measured confusion matrices for all models on the time split."""
    if not eval_json_path.exists():
        print(f"[-] Missing {eval_json_path}. Run python main.py.")
        return False

    try:
        with eval_json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[-] Error reading {eval_json_path}: {e}. Run python main.py.")
        return False

    splits_data = data.get("splits", {})
    time_data = splits_data.get("time", {})
    if not time_data:
        print(f"[-] No time split evaluation found in {eval_json_path}. Run python main.py.")
        return False

    models_with_cm = []
    for m_key, m_val in time_data.items():
        if isinstance(m_val, dict):
            conf = m_val.get("at_threshold", {}).get("confusion")
            if conf and all(k in conf for k in ("tp", "fp", "fn", "tn")):
                models_with_cm.append((m_key, m_val, conf))

    if not models_with_cm:
        print(f"[-] No confusion matrix counts found in {eval_json_path}. Run python main.py.")
        return False

    n_models = len(models_with_cm)
    cols = min(n_models, 3)
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), dpi=300)
    axes = np.array(axes).reshape(-1)

    for i, (m_key, m_val, conf) in enumerate(models_with_cm):
        ax = axes[i]
        cm = np.array([
            [conf["tn"], conf["fp"]],
            [conf["fn"], conf["tp"]]
        ])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            annot_kws={"fontsize": 11, "fontweight": "bold"},
        )
        display_name = MODEL_DISPLAY_NAMES.get(m_key, m_key.replace("_", " ").title())
        th = m_val.get("threshold", 0.5)
        ax.set_title(f"{display_name}\n(Threshold = {th:.4f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=9)
        ax.set_ylabel("True Label", fontsize=9)
        ax.set_xticklabels(["Legitimate (0)", "Fraud (1)"], fontsize=9)
        ax.set_yticklabels(["Legitimate (0)", "Fraud (1)"], fontsize=9)

    # Hide unused subplots
    for j in range(len(models_with_cm), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Measured Confusion Matrices (Time Split Test Set)", fontsize=14, fontweight="bold", y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved confusion matrices: {output_path}")
    return True


def plot_feature_importance(feat_json_path: Path, output_path: Path) -> bool:
    """Plot top 15 features by gain for trained XGBoost models."""
    if not feat_json_path.exists():
        print(f"[-] Missing {feat_json_path}. Run python main.py.")
        return False

    try:
        with feat_json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[-] Error reading {feat_json_path}: {e}. Run python main.py.")
        return False

    valid_models = {k: v for k, v in data.items() if isinstance(v, list) and len(v) > 0}
    if not valid_models:
        print(f"[-] No feature importance entries in {feat_json_path}. Run python main.py.")
        return False

    n_models = len(valid_models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 7), dpi=300, sharey=False)
    if n_models == 1:
        axes = [axes]

    for ax, (m_key, feats) in zip(axes, valid_models.items()):
        top_15 = feats[:15]
        df_feats = pd.DataFrame(top_15).sort_values(by="importance", ascending=True)
        ax.barh(df_feats["feature"], df_feats["importance"], color="#38bdf8", edgecolor="#0284c7")
        display_name = MODEL_DISPLAY_NAMES.get(m_key, m_key.replace("_", " ").title())
        ax.set_title(display_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Relative Importance (XGBoost Gain)", fontsize=9, fontweight="bold")
        max_imp = max(df_feats["importance"]) if not df_feats.empty else 1.0
        ax.set_xlim(0, max_imp * 1.15)

    fig.suptitle("Top 15 Feature Importances by Model Gain", fontsize=14, fontweight="bold", y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved feature importance: {output_path}")
    return True


def plot_business_impact(impact_json_path: Path, output_path: Path, review_cost: float = 10000.0) -> bool:
    """Plot net financial savings and fraud value caught curves across score thresholds."""
    if not impact_json_path.exists():
        print(f"[-] Missing {impact_json_path}. Run python main.py.")
        return False

    try:
        with impact_json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[-] Error reading {impact_json_path}: {e}. Run python main.py.")
        return False

    models_data = data.get("models", {})
    rule_base = data.get("rule_baseline", {})
    if not models_data:
        print(f"[-] No model grids in {impact_json_path}. Run python main.py.")
        return False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    colors = {
        "m1_tabular": "#38bdf8",
        "m1_tabular_noerr": "#fbbf24",
        "m2_graph": "#34d399",
        "m3_community_mule": "#a855f7",
    }

    rule_alerts = float(rule_base.get("alerts", 0))
    rule_caught = float(rule_base.get("fraud_value_caught", 0.0))
    rule_net = rule_caught - (review_cost * rule_alerts)

    for m_key, m_val in models_data.items():
        grid = m_val.get("grid", [])
        if not grid:
            continue
        th_vals = [float(p.get("threshold", 0.0)) for p in grid]
        net_vals = [float(p.get("fraud_value_caught", 0.0)) - review_cost * float(p.get("alerts", 0)) for p in grid]
        share_vals = [float(p.get("share_fraud_value_caught", 0.0)) * 100 for p in grid]
        color = colors.get(m_key, "#4f46e5")
        label = MODEL_DISPLAY_NAMES.get(m_key, m_key)

        ax1.plot(th_vals, net_vals, label=label, color=color, linewidth=2.0)
        ax2.plot(th_vals, share_vals, label=label, color=color, linewidth=2.0)

    # Reference line for rule baseline
    ax1.axhline(rule_net, color="#ef4444", linestyle="--", linewidth=1.5, label=f"Rule Baseline ({rule_net:,.0f})")
    rule_share = float(rule_base.get("share_fraud_value_caught", 0.0)) * 100
    ax2.axhline(rule_share, color="#ef4444", linestyle="--", linewidth=1.5, label=f"Rule Baseline ({rule_share:.1f}%)")

    ax1.set_xlabel("Score Operating Threshold", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Net Financial Saving (Simulated Currency Units)", fontsize=10, fontweight="bold")
    ax1.set_title(f"Net Financial Saving (Review Cost = {review_cost:,.0f} units)", fontsize=11, fontweight="bold")
    ax1.legend(loc="best", fontsize=9, frameon=True)

    ax2.set_xlabel("Score Operating Threshold", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Share of Fraud Value Caught (%)", fontsize=10, fontweight="bold")
    ax2.set_title("Fraud Value Capture Rate across Thresholds", fontsize=11, fontweight="bold")
    ax2.set_ylim(-2, 105)
    ax2.legend(loc="best", fontsize=9, frameon=True)

    fig.suptitle("Business Impact Analysis on Time Split Test Set", fontsize=13, fontweight="bold", y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved business impact: {output_path}")
    return True


def plot_ring_size_histogram(ring_json_path: Path, output_path: Path) -> bool:
    """Plot distribution of connected component sizes in ground-truth fraud data."""
    if not ring_json_path.exists():
        print(f"[-] Missing {ring_json_path}. Run python main.py.")
        return False

    try:
        with ring_json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[-] Error reading {ring_json_path}: {e}. Run python main.py.")
        return False

    size_dist = data.get("number_of_components_by_size", {})
    if not size_dist:
        print(f"[-] No component distribution in {ring_json_path}. Run python main.py.")
        return False

    # Extract sizes and counts
    categories = ["Size 2\n(1-to-1 Pairs)", "Size 3\n(2-to-1 Funnels)", "Size ≥ 4\n(Chains / Cycles)"]
    count_size_2 = int(size_dist.get("2", 0))
    count_size_3 = int(size_dist.get("3", 0))
    count_size_ge_4 = sum(int(v) for k, v in size_dist.items() if int(k) >= 4)
    counts = [count_size_2, count_size_3, count_size_ge_4]

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    colors = ["#38bdf8", "#fbbf24", "#ef4444"]
    bars = ax.bar(categories, counts, color=colors, edgecolor="#334155", linewidth=1.2, width=0.5)

    # Use log scale on y-axis due to extreme disparity (8,125 vs 44 vs 0)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 25000)
    ax.set_ylabel("Component Count (Log Scale)", fontsize=11, fontweight="bold")
    ax.set_title("Ground-Truth Fraud Component Size Distribution in PaySim", fontsize=13, fontweight="bold", pad=12)

    for bar, count in zip(bars, counts):
        h = bar.get_height()
        ax.annotate(
            f"{count:,}",
            xy=(bar.get_x() + bar.get_width() / 2, max(h, 0.8)),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    finding = data.get("finding", "")
    if finding:
        ax.text(
            0.5, -0.22,
            f"Empirical Finding: {finding}",
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=8.5,
            style="italic",
            wrap=True,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.9),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved ring size histogram: {output_path}")
    return True


def plot_scaling_data_size(scaling_json_path: Path, output_path: Path) -> bool:
    """Plot graph-stage wall time versus data size on a log-log scale with individual repeats."""
    if not scaling_json_path.exists():
        print(f"[*] Note: {scaling_json_path} not found. Skipping data-size scaling chart.")
        return False

    try:
        with scaling_json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[-] Error reading {scaling_json_path}: {e}")
        return False

    summary = data.get("data_size_summary", []) if isinstance(data, dict) else []
    runs = data.get("runs", []) if isinstance(data, dict) else []
    if not summary:
        print(f"[*] No data_size_summary in {scaling_json_path}. Skipping.")
        return False

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    # Individual OK repeats
    ok_x, ok_y = [], []
    non_ok_notes = []
    for r in runs:
        if r.get("suite") != "data_scaling":
            continue
        frac = float(r.get("fraction", 0))
        if r.get("status") == "ok":
            tot = float((r.get("seconds") or {}).get("total", r.get("wall_seconds", 0)))
            ok_x.append(frac)
            ok_y.append(tot)
        else:
            non_ok_notes.append(f"f={frac:.2f} r{r.get('repeat')}: {r.get('status')}")

    mean_x, mean_y = [], []
    for row in summary:
        if row.get("mean_total_seconds") is not None:
            mean_x.append(float(row["fraction"]))
            mean_y.append(float(row["mean_total_seconds"]))

    if mean_x:
        ax.plot(mean_x, mean_y, color="#0284c7", linewidth=2.4, marker="s", markersize=7, label="Mean (status=ok)")
    if ok_x:
        ax.scatter(ok_x, ok_y, color="#38bdf8", edgecolor="#0f172a", s=55, zorder=5, label="Individual repeats (ok)")

    # Annotate aborted fractions along x-axis reference
    aborted_fracs = sorted({float(row["fraction"]) for row in summary if row.get("ok_count", 0) < row.get("total_count", 0)})
    for af in aborted_fracs:
        ax.axvline(af, color="#ef4444", linestyle=":", alpha=0.65, linewidth=1.4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([0.10, 0.25, 0.50, 1.00])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("PaySim Transaction Sample Fraction (log scale)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Graph-Stage Execution Time, Seconds (log scale)", fontsize=11, fontweight="bold")
    ax.set_title("Spark Data-Size Scaling (local[*], 6g driver, 48 partitions)", fontsize=12.5, fontweight="bold", pad=12)

    note_parts = ["Single machine (15.37 GiB RAM), uniform sampling changes graph density, 2 repeats."]
    if non_ok_notes:
        note_parts.append("Non-ok runs (vertical dotted lines): " + "; ".join(non_ok_notes))
    ax.text(
        0.5, -0.22,
        "\n".join(note_parts),
        ha="center", va="top", transform=ax.transAxes,
        fontsize=8.2, style="italic",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f8fafc", edgecolor="#cbd5e1"),
    )
    ax.legend(loc="upper left", frameon=True, facecolor="white")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved data-size scaling chart: {output_path}")
    return True


def plot_scaling_cores_speedup(scaling_json_path: Path, output_path: Path) -> bool:
    """Plot speedup and wall time versus Spark local[n] core count."""
    if not scaling_json_path.exists():
        print(f"[*] Note: {scaling_json_path} not found. Skipping core-scaling chart.")
        return False

    try:
        with scaling_json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[-] Error reading {scaling_json_path}: {e}")
        return False

    summary = data.get("core_scaling_summary", []) if isinstance(data, dict) else []
    if not summary:
        print(f"[*] No core_scaling_summary in {scaling_json_path}. Skipping.")
        return False

    cores = [int(r["cores"]) for r in summary if r.get("speedup") is not None]
    speedups = [float(r["speedup"]) for r in summary if r.get("speedup") is not None]
    mean_secs = [float(r["mean_total_seconds"]) for r in summary if r.get("mean_total_seconds") is not None]
    frac = data.get("core_scaling_fraction", 0.10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)

    # Panel 1: Speedup vs Cores
    ax1.plot(cores, speedups, color="#10b981", marker="o", linewidth=2.4, markersize=7, label="Measured Speedup (mean)")
    for row in summary:
        c = int(row["cores"])
        for sp in row.get("individual_speedups", []):
            ax1.scatter([c], [float(sp)], color="#34d399", edgecolor="#064e3b", s=42, zorder=5)
    ax1.plot(cores, [1.0] * len(cores), color="#94a3b8", linestyle=":", linewidth=1.2, label="1.0x Baseline (local[1])")
    ax1.set_xticks(cores)
    ax1.set_xlabel("Spark Local Threads (local[n])", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Speedup vs local[1]", fontsize=10.5, fontweight="bold")
    ax1.set_title(f"Core-Scaling Speedup (Fraction = {frac:.2f})", fontsize=11.5, fontweight="bold")
    ax1.legend(loc="best", frameon=True)

    # Panel 2: Wall Time vs Cores
    ax2.plot(cores, mean_secs, color="#6366f1", marker="s", linewidth=2.4, markersize=7, label="Mean Time (s)")
    for row in summary:
        c = int(row["cores"])
        for val in row.get("individual_total_seconds", []):
            ax2.scatter([c], [float(val)], color="#a5b4fc", edgecolor="#312e81", s=42, zorder=5)
    ax2.set_xticks(cores)
    ax2.set_xlabel("Spark Local Threads (local[n])", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Graph-Stage Execution Time (Seconds)", fontsize=10.5, fontweight="bold")
    ax2.set_title(f"Runtime Across Thread Counts (Fraction = {frac:.2f})", fontsize=11.5, fontweight="bold")
    ax2.legend(loc="best", frameon=True)

    fig.suptitle("Spark Core Scaling on Hybrid Laptop CPU (Intel Core Ultra 7 155H, 2 Repeats)", fontsize=12.5, fontweight="bold", y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved core-scaling speedup chart: {output_path}")
    return True


def plot_pandas_vs_spark(pvs_json_path: Path, output_path: Path) -> bool:
    """Plot pandas/NetworkX vs Spark compute time and peak memory across dataset sizes."""
    if not pvs_json_path.exists():
        print(f"[*] Note: {pvs_json_path} not found. Skipping pandas vs Spark chart.")
        return False

    try:
        with pvs_json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[-] Error reading {pvs_json_path}: {e}")
        return False

    summary = data.get("summary", []) if isinstance(data, dict) else []
    if not summary:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=300)
    styles = {
        ("pandas_networkx", "degree"): ("#0284c7", "o", "-", "pandas (degree)"),
        ("spark", "degree"): ("#38bdf8", "s", "--", "Spark (degree)"),
        ("pandas_networkx", "components"): ("#10b981", "o", "-", "NetworkX (components)"),
        ("spark", "components"): ("#ef4444", "s", "--", "Spark (components)"),
    }

    # Panel 1: Compute time (log-log)
    ax_t = axes[0]
    ax_m = axes[1]
    non_ok_labels = []

    for (tool, task), (color, marker, ls, label) in styles.items():
        rows_ok = [r for r in summary if r["tool"] == tool and r["task"] == task and r.get("mean_compute_seconds") is not None]
        rows_ok.sort(key=lambda r: int(r["rows"]))
        if rows_ok:
            xs = [int(r["rows"]) for r in rows_ok]
            ys_t = [float(r["mean_compute_seconds"]) for r in rows_ok]
            ys_m = [float(r["mean_peak_rss_bytes"]) / (1024 ** 3) for r in rows_ok]
            ax_t.plot(xs, ys_t, color=color, marker=marker, linestyle=ls, linewidth=2.2, markersize=6, label=label)
            ax_m.plot(xs, ys_m, color=color, marker=marker, linestyle=ls, linewidth=2.2, markersize=6, label=label)
            for r in rows_ok:
                for iv in r.get("individual_compute_seconds", []):
                    ax_t.scatter([int(r["rows"])], [float(iv)], color=color, alpha=0.45, s=28)

        rows_fail = [r for r in summary if r["tool"] == tool and r["task"] == task and r.get("ok_count", 0) == 0]
        for rf in rows_fail:
            non_ok_labels.append(f"{label} @ {int(rf['rows']):,}: {','.join(rf.get('statuses', []))}")
            if rf.get("max_peak_rss_bytes"):
                ax_m.scatter(
                    [int(rf["rows"])],
                    [float(rf["max_peak_rss_bytes"]) / (1024 ** 3)],
                    color=color, marker="X", s=110, zorder=6,
                )

    ax_t.set_xscale("log")
    ax_t.set_yscale("log")
    ax_t.set_xlabel("Sample Size (Transactions, log scale)", fontsize=10.5, fontweight="bold")
    ax_t.set_ylabel("Compute Time, Seconds (log scale)", fontsize=10.5, fontweight="bold")
    ax_t.set_title("Compute Time (Load Excluded, 2 Repeats)", fontsize=11.5, fontweight="bold")
    ax_t.legend(loc="best", fontsize=8.5, frameon=True)

    ax_m.set_xscale("log")
    ax_m.set_xlabel("Sample Size (Transactions, log scale)", fontsize=10.5, fontweight="bold")
    ax_m.set_ylabel("Peak Process-Tree RSS (GiB)", fontsize=10.5, fontweight="bold")
    ax_m.set_title("Peak Memory Footprint (RSS)", fontsize=11.5, fontweight="bold")
    ax_m.legend(loc="best", fontsize=8.5, frameon=True)

    subtitle = "Single-machine comparison (15.37 GiB RAM, 10-min timeout, 9 GiB RSS cap)."
    if non_ok_labels:
        subtitle += " Limit hit: " + "; ".join(non_ok_labels)
    fig.suptitle(f"pandas/NetworkX vs Spark-Native Graph Benchmarks\n{subtitle}", fontsize=11.5, fontweight="bold", y=1.04)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved pandas vs Spark chart: {output_path}")
    return True


def plot_storage_benchmark(storage_json_path: Path, output_path: Path) -> bool:
    """Plot CSV vs Parquet file sizes and 3-repeat query read latencies."""
    if not storage_json_path.exists():
        print(f"[*] Note: {storage_json_path} not found. Skipping storage benchmark chart.")
        return False

    try:
        with storage_json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[-] Error reading {storage_json_path}: {e}")
        return False

    fmts = data.get("formats", {}) if isinstance(data, dict) else {}
    if not fmts:
        return False

    order = [k for k in ("csv", "parquet_partitioned", "parquet_unpartitioned") if k in fmts]
    labels = {
        "csv": "CSV\n(Explicit Schema)",
        "parquet_partitioned": "Parquet\n(Partitioned by type)",
        "parquet_unpartitioned": "Parquet\n(Unpartitioned)",
    }
    colors = {
        "csv": "#94a3b8",
        "parquet_partitioned": "#38bdf8",
        "parquet_unpartitioned": "#10b981",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=300)

    # Panel 1: Storage size (MB)
    sizes = [float(fmts[k]["size_mb"]) for k in order]
    bars = ax1.bar([labels[k] for k in order], sizes, color=[colors[k] for k in order], edgecolor="#334155", width=0.55)
    for bar, k in zip(bars, order):
        sz = float(fmts[k]["size_mb"])
        ratio = float(fmts[k].get("compression_ratio_vs_csv", 1.0))
        ax1.annotate(
            f"{sz:.1f} MB\n({ratio:.2f}x)",
            xy=(bar.get_x() + bar.get_width() / 2, sz),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold",
        )
    ax1.set_ylim(0, max(sizes) * 1.22)
    ax1.set_ylabel("On-Disk Size (MiB)", fontsize=10.5, fontweight="bold")
    ax1.set_title("PaySim Dataset Storage Size (6.36M Rows)", fontsize=11.5, fontweight="bold")

    # Panel 2: Query times (3-repeat means + individual points)
    ops = [
        ("count_rows", "Count Rows"),
        ("read_two_columns", "Read 2 Cols\n(sum agg)"),
        ("filter_transfer_fraud", "Filter TRANSFER\n& isFraud=1"),
    ]
    x = np.arange(len(ops))
    width = 0.25
    for idx, k in enumerate(order):
        means = [float(fmts[k]["operations"][op_k]["mean_seconds"]) for op_k, _ in ops]
        offset = (idx - (len(order) - 1) / 2.0) * width
        ax2.bar(x + offset, means, width=width, color=colors[k], edgecolor="#334155", label=labels[k].replace("\n", " "))
        for j, (op_k, _) in enumerate(ops):
            reps = fmts[k]["operations"][op_k].get("repeats_seconds", [])
            ax2.scatter([x[j] + offset] * len(reps), reps, color="#0f172a", s=18, zorder=5, alpha=0.75)

    ax2.set_xticks(x)
    ax2.set_xticklabels([lbl for _, lbl in ops], fontsize=9.5)
    ax2.set_ylabel("Query Latency, Seconds (Mean + 3 Repeats)", fontsize=10.5, fontweight="bold")
    ax2.set_title("Spark Query Read Latency (3 Repeats; Dots = Individual Runs)", fontsize=11.5, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8.5, frameon=True)

    fig.suptitle("Storage Format Benchmark: CSV vs Parquet (6,362,620 Transactions)", fontsize=12.5, fontweight="bold", y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved storage benchmark chart: {output_path}")
    return True


def plot_scaling_benchmark(scaling_json_path: Path, output_path: Path) -> bool:
    """Backward-compatible wrapper writing the data-size scaling chart to scaling_benchmark.png."""
    return plot_scaling_data_size(scaling_json_path, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate presentation visuals from accepted pipeline outputs.")
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("FRAUD_OUTPUT_DIR", "outputs"),
        help="Directory containing accepted pipeline outputs (default: outputs)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    visuals_dir = out_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Reading pipeline outputs from: {out_dir}")
    print(f"[*] Target visuals directory: {visuals_dir}")

    # Verify primary files exist
    required_files = [
        out_dir / "pr_curves.csv",
        out_dir / "evaluation.json",
        out_dir / "feature_importance.json",
        out_dir / "business_impact.json",
        out_dir / "ring_analysis.json",
    ]
    missing = [f.name for f in required_files if not f.exists()]
    if missing:
        print(f"[-] Warning: The following output files are missing in {out_dir}: {', '.join(missing)}")
        print("[-] Please run python main.py to generate pipeline outputs.")

    success_count = 0
    if plot_pr_curves(out_dir / "pr_curves.csv", visuals_dir / "pr_curves.png"):
        success_count += 1
    if plot_confusion_matrices(out_dir / "evaluation.json", visuals_dir / "confusion_matrices.png"):
        success_count += 1
    if plot_feature_importance(out_dir / "feature_importance.json", visuals_dir / "feature_importance.png"):
        success_count += 1
    if plot_business_impact(out_dir / "business_impact.json", visuals_dir / "business_impact.png"):
        success_count += 1
    if plot_ring_size_histogram(out_dir / "ring_analysis.json", visuals_dir / "ring_size_histogram.png"):
        success_count += 1

    # Chunk 3 benchmark visuals (generated when benchmark JSON outputs exist)
    bench_count = 0
    if plot_scaling_data_size(out_dir / "scaling_runs.json", visuals_dir / "scaling_data_size.png"):
        bench_count += 1
        plot_scaling_benchmark(out_dir / "scaling_runs.json", visuals_dir / "scaling_benchmark.png")
    if plot_scaling_cores_speedup(out_dir / "scaling_runs.json", visuals_dir / "scaling_cores_speedup.png"):
        bench_count += 1
    if plot_pandas_vs_spark(out_dir / "pandas_vs_spark.json", visuals_dir / "pandas_vs_spark.png"):
        bench_count += 1
    if plot_storage_benchmark(out_dir / "storage_benchmark.json", visuals_dir / "storage_benchmark.png"):
        bench_count += 1

    print(
        f"[*] Generated {success_count}/5 core visual artifacts and {bench_count}/4 benchmark visual artifacts in {visuals_dir}"
    )


if __name__ == "__main__":
    main()
