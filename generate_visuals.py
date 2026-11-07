"""
generate_visuals.py
Generates presentation-grade visualizations and charts from evaluation metrics CSVs
located in output/metrics/. Saves high-resolution figures to output/visuals/.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 1.0,
    "grid.color": "#e5e5e5",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "figure.autolayout": True
})

METRICS_DIR = os.path.join("output", "metrics")
VISUALS_DIR = os.path.join("output", "visuals")
os.makedirs(VISUALS_DIR, exist_ok=True)


def plot_spark_scaling(df_scaling: pd.DataFrame):
    """
    Plots Spark Systems Scaling benchmarks:
    1. Execution time breakdown per GraphX algorithm.
    2. JVM memory utilization vs. graph size (nodes/edges).
    """
    print("[*] Generating Spark Systems Scaling visualizations...")

    # Figure 1: Algorithm Execution Time Breakdown (Grouped Bar Chart)
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    
    x = np.arange(len(df_scaling["Graph Scale"]))
    width = 0.18

    algos = [
        ("PageRank Time (s)", "#2b5c8f", "PageRank"),
        ("Connected Comp Time (s)", "#e06666", "Connected Components"),
        ("Triangle Count Time (s)", "#6aa84f", "Triangle Count"),
        ("LPA Time (s)", "#f6b26b", "Label Propagation (LPA)")
    ]

    for i, (col, color, label) in enumerate(algos):
        offset = (i - 1.5) * width
        rects = ax1.bar(x + offset, df_scaling[col], width, label=label, color=color, edgecolor="black", linewidth=0.6)
        # Add value labels on top of bars
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(f"{height:.2f}s",
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax1.set_xlabel("Graph Scale (% of Sampled Transactions)", fontsize=11, fontweight="bold", labelpad=10)
    ax1.set_ylabel("Execution Time (Seconds)", fontsize=11, fontweight="bold")
    ax1.set_title("Spark GraphFrames Algorithm Runtime Across Graph Scales", fontsize=13, fontweight="bold", pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{scale}\n({nodes:,} nodes, {edges:,} edges)" for scale, nodes, edges in zip(df_scaling["Graph Scale"], df_scaling["Nodes"], df_scaling["Edges"])], fontsize=10)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    ax1.set_ylim(0, max(df_scaling["Connected Comp Time (s)"]) * 1.25)

    runtime_fig_path = os.path.join(VISUALS_DIR, "spark_algorithm_runtime_scaling.png")
    fig.savefig(runtime_fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {runtime_fig_path}")

    # Figure 2: Total Runtime and JVM Heap Memory Scaling
    fig, ax_time = plt.subplots(figsize=(9, 5.5), dpi=300)
    ax_mem = ax_time.twinx()

    scales = df_scaling["Graph Scale"].tolist()
    total_times = df_scaling["Total Time (s)"].tolist()
    jvm_mems = df_scaling["JVM Used Mem (MB)"].tolist()

    l1 = ax_time.plot(scales, total_times, marker="o", markersize=8, color="#d9534f", linewidth=2.5, label="Total Time (s)")
    l2 = ax_mem.plot(scales, jvm_mems, marker="s", markersize=8, color="#0275d8", linewidth=2.5, linestyle="--", label="JVM Used Memory (MB)")

    for i, (t, m) in enumerate(zip(total_times, jvm_mems)):
        ax_time.annotate(f"{t:.1f}s", xy=(i, t), xytext=(-15, 8), textcoords="offset points", fontweight="bold", color="#d9534f")
        ax_mem.annotate(f"{m:.1f} MB", xy=(i, m), xytext=(8, -12), textcoords="offset points", fontweight="bold", color="#0275d8")

    ax_time.set_xlabel("Graph Scale", fontsize=11, fontweight="bold", labelpad=10)
    ax_time.set_ylabel("Total Pipeline Stage Duration (Seconds)", fontsize=11, fontweight="bold", color="#d9534f")
    ax_mem.set_ylabel("Spark JVM Heap Allocated (MB)", fontsize=11, fontweight="bold", color="#0275d8")
    ax_time.set_title("Spark System Scaling: Total Execution Time & JVM Heap Footprint", fontsize=13, fontweight="bold", pad=15)
    
    # Combined legend
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax_time.legend(lines, labels, loc="center left", frameon=True, facecolor="white", framealpha=0.9)
    ax_mem.set_ylim(0, 4200)

    mem_fig_path = os.path.join(VISUALS_DIR, "spark_systems_memory_scaling.png")
    fig.savefig(mem_fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {mem_fig_path}")


def plot_ring_detection_comparison(df_rings: pd.DataFrame):
    """
    Plots ring-level and community detection metrics across models.
    """
    print("[*] Generating Ring Detection & Community Metrics visual...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    metrics = [
        ("Ring Detection Rate", "#2e7d32"),
        ("Cluster Purity", "#1565c0"),
        ("Cluster Completeness", "#ef6c00"),
        ("V-Measure", "#6a1b9a")
    ]

    models = df_rings["Model"].tolist()
    x = np.arange(len(models))
    width = 0.20

    for i, (m_col, color) in enumerate(metrics):
        offset = (i - 1.5) * width
        vals = df_rings[m_col].tolist()
        rects = ax.bar(x + offset, vals, width, label=m_col, color=color, edgecolor="black", linewidth=0.6)
        for rect in rects:
            height = rect.get_height()
            label_text = f"{height:.2f}" if height >= 0.1 else f"{height:.3f}"
            ax.annotate(label_text,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlabel("Model Architecture", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Metric Value (0.0 - 1.0)", fontsize=11, fontweight="bold")
    ax.set_title("Fraud Ring Detection & Community Clustering Performance Comparison", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9.5, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95)
    ax.set_ylim(0, 1.25)

    ring_fig_path = os.path.join(VISUALS_DIR, "ring_detection_comparison.png")
    fig.savefig(ring_fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {ring_fig_path}")


def plot_transaction_performance(df_txns: pd.DataFrame):
    """
    Plots transaction-level classification metrics comparison.
    """
    print("[*] Generating Transaction-Level Performance visual...")
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    metrics = ["Precision", "Recall", "F1-Score", "AUC-ROC", "PR-AUC"]
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    models = df_txns["Model"].tolist()
    x = np.arange(len(models))
    width = 0.15

    for i, (m_col, color) in enumerate(zip(metrics, palette)):
        offset = (i - 2) * width
        vals = df_txns[m_col].tolist()
        rects = ax.bar(x + offset, vals, width, label=m_col, color=color, edgecolor="black", linewidth=0.6)
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlabel("Model Architecture", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=11, fontweight="bold")
    ax.set_title("Transaction-Level Classification Performance Across Comparative Models", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9.5, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.95)
    ax.set_ylim(0, 1.25)

    txn_fig_path = os.path.join(VISUALS_DIR, "transaction_performance_comparison.png")
    fig.savefig(txn_fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {txn_fig_path}")


def plot_master_dashboard(df_scaling: pd.DataFrame, df_rings: pd.DataFrame, df_txns: pd.DataFrame):
    """
    Creates a comprehensive 4-panel dashboard suitable for inclusion in research presentations/papers.
    """
    print("[*] Generating Comprehensive Multi-Panel Dashboard...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

    # Panel 1: Spark Scaling Runtime Breakdown
    ax1 = axes[0, 0]
    scales = df_scaling["Graph Scale"]
    ax1.plot(scales, df_scaling["PageRank Time (s)"], marker="o", label="PageRank", linewidth=2, color="#2b5c8f")
    ax1.plot(scales, df_scaling["Connected Comp Time (s)"], marker="s", label="Connected Components", linewidth=2, color="#e06666")
    ax1.plot(scales, df_scaling["Triangle Count Time (s)"], marker="^", label="Triangle Count", linewidth=2, color="#6aa84f")
    ax1.plot(scales, df_scaling["LPA Time (s)"], marker="d", label="Label Propagation (LPA)", linewidth=2, color="#f6b26b")
    ax1.set_title("(A) GraphX Algorithm Runtime vs Scale", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Graph Scale Fraction", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Execution Time (s)", fontsize=10, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True)

    # Panel 2: JVM Memory Scaling
    ax2 = axes[0, 1]
    sns.barplot(x="Graph Scale", y="JVM Used Mem (MB)", data=df_scaling, ax=ax2, hue="Graph Scale", palette="Blues_d", legend=False, edgecolor="black")
    for p in ax2.patches:
        ax2.annotate(f"{p.get_height():.1f} MB",
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha="center", va="bottom", fontsize=9, fontweight="bold",
                     xytext=(0, 3), textcoords="offset points")
    ax2.set_title("(B) Spark JVM Heap Allocation Across Graph Scales", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Graph Scale Fraction", fontsize=10, fontweight="bold")
    ax2.set_ylabel("JVM Memory Used (MB)", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 2000)

    # Panel 3: Ring Level Metric Breakdown
    ax3 = axes[1, 0]
    models = df_rings["Model"]
    x = np.arange(len(models))
    w = 0.35
    ax3.bar(x - w/2, df_rings["Ring Detection Rate"], width=w, label="Ring Detection Rate", color="#2e7d32", edgecolor="black")
    ax3.bar(x + w/2, df_rings["V-Measure"], width=w, label="V-Measure", color="#6a1b9a", edgecolor="black")
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, fontsize=8.5, fontweight="bold")
    ax3.set_title("(C) Fraud Ring Detection Rate vs Syndicate V-Measure", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Metric Score", fontsize=10, fontweight="bold")
    ax3.legend(loc="upper right", frameon=True)
    ax3.set_ylim(0, 1.2)

    # Panel 4: Confusion Matrix Analysis (Normalized)
    ax4 = axes[1, 1]
    tp = df_txns["True Positive"].iloc[0]
    fn = df_txns["False Negative"].iloc[0]
    fp = df_txns["False Positive"].iloc[0]
    tn = df_txns["True Negative"].iloc[0]
    cm = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax4,
                xticklabels=["Pred Legitimate (0)", "Pred Fraud (1)"],
                yticklabels=["Actual Legitimate (0)", "Actual Fraud (1)"])
    ax4.set_title("(D) Test Set Confusion Matrix (Baseline & Graph Models)", fontsize=12, fontweight="bold")
    ax4.set_ylabel("True Category", fontsize=10, fontweight="bold")
    ax4.set_xlabel("Model Classification", fontsize=10, fontweight="bold")

    plt.suptitle("End-to-End Financial Fraud Analytics Evaluation Dashboard", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()

    dashboard_path = os.path.join(VISUALS_DIR, "fraud_analytics_comprehensive_dashboard.png")
    fig.savefig(dashboard_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {dashboard_path}")


def main():
    txn_csv = os.path.join(METRICS_DIR, "transaction_level_metrics.csv")
    ring_csv = os.path.join(METRICS_DIR, "ring_level_metrics.csv")
    scaling_csv = os.path.join(METRICS_DIR, "spark_systems_scaling.csv")

    for path in [txn_csv, ring_csv, scaling_csv]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required metrics CSV not found: {path}. Run run_pipeline.py first.")

    df_txns = pd.read_csv(txn_csv)
    df_rings = pd.read_csv(ring_csv)
    df_scaling = pd.read_csv(scaling_csv)

    plot_spark_scaling(df_scaling)
    plot_ring_detection_comparison(df_rings)
    plot_transaction_performance(df_txns)
    plot_master_dashboard(df_scaling, df_rings, df_txns)

    print("\n[+] All visual charts successfully generated and saved to 'output/visuals/'.")


if __name__ == "__main__":
    main()
