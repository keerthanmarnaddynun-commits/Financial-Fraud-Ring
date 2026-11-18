"""Deterministic report and template renderer reading strictly from outputs/ and docs/.

Generates:
  - docs/RESULTS_TABLES.md
  - docs/FACTS_FOR_PAPER.md
  - Any template under docs/templates/*.tmpl (e.g., README.md.tmpl -> README.md)

Supports placeholders in templates:
  - {{table:<name>}}
  - {{num:<file>:<jsonpath>}} or {{num:<file>:<jsonpath>:<fmt>}}
    Supported formats: raw (default), int, 2f, 3f, 4f, pct2, pct4, bytes, mib, gib
  - {{text:<relative_path>}} (embeds UTF-8 text file content)

Optionally syncs small final result artifacts (JSON, CSV, PNG) to results_snapshot/
and writes results_snapshot/SHA256SUMS.txt.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import stat
from pathlib import Path
from typing import Any, Dict, List, Tuple

from provenance import write_manifest

TABLE_PLACEHOLDER_RE = re.compile(r"\{\{table:([a-zA-Z0-9_]+)\}\}")
NUM_PLACEHOLDER_RE = re.compile(r"\{\{num:([^:}]+):([^:}]+)(?::([a-zA-Z0-9_]+))?\}\}")
TEXT_PLACEHOLDER_RE = re.compile(r"\{\{text:([^}]+)\}\}")


def _fmt_bytes(val: Any) -> str:
    if val is None:
        return "not available"
    num = float(val)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if num < 1024.0:
            return f"{num:.2f} {unit}"
        num /= 1024.0
    return f"{num:.2f} PiB"


def _fmt_num(val: Any, decimals: int = 4) -> str:
    if val is None:
        return "not available"
    return f"{float(val):.{decimals}f}"


def _fmt_int(val: Any) -> str:
    if val is None:
        return "not available"
    return f"{int(round(float(val))):,}"


def _fmt_pct(val: Any, decimals: int = 2) -> str:
    if val is None:
        return "not available"
    return f"{float(val) * 100.0:.{decimals}f}%"


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _lookup_json_path(data: Any, expr: str) -> Any:
    current = data
    for key in expr.split("."):
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list):
            idx = int(key)
            current = current[idx]
        else:
            raise KeyError(f"JSON path '{expr}' failed at segment '{key}'")
    return current


def _format_value(val: Any, fmt: str | None) -> str:
    if val is None:
        return "not available"
    if not fmt or fmt == "raw":
        return str(val)
    if fmt == "int":
        return _fmt_int(val)
    if fmt == "2f":
        return _fmt_num(val, 2)
    if fmt == "3f":
        return _fmt_num(val, 3)
    if fmt == "4f":
        return _fmt_num(val, 4)
    if fmt == "pct2":
        return _fmt_pct(val, 2)
    if fmt == "pct4":
        return _fmt_pct(val, 4)
    if fmt == "bytes":
        return _fmt_bytes(val)
    if fmt == "mib":
        return f"{float(val) / (1024 ** 2):.2f} MiB"
    if fmt == "gib":
        return f"{float(val) / (1024 ** 3):.2f} GiB"
    raise ValueError(f"Unsupported format specifier: {fmt}")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class ReportContext:
    """Holds loaded result artifacts and builds named Markdown tables programmatically."""

    def __init__(self, outputs_dir: Path, project_root: Path) -> None:
        self.outputs_dir = outputs_dir
        self.project_root = project_root
        self._json_cache: Dict[str, Any] = {}
        self.data_quality = self.get_json("outputs/data_quality.json")
        self.raw_facts = self.get_json("docs/raw_data_facts.json")
        self.run_info = self.get_json("outputs/run_info.json")
        self.scalability = self.get_json("outputs/scalability.json")
        self.evaluation = self.get_json("outputs/evaluation.json")
        self.business_impact = self.get_json("outputs/business_impact.json")
        self.ring_analysis = self.get_json("outputs/ring_analysis.json")
        self.planted_rings = self.get_json("outputs/planted_ring_validation.json")
        self.scaling_runs = self.get_json("outputs/scaling_runs.json")
        self.pandas_vs_spark = self.get_json("outputs/pandas_vs_spark.json")
        self.storage_benchmark = self.get_json("outputs/storage_benchmark.json")
        self.model_config = self.get_json("configs/model_config.json")
        self.tables: Dict[str, str] = self._build_all_tables()

    def get_json(self, rel_or_name: str) -> Any:
        if rel_or_name in self._json_cache:
            return self._json_cache[rel_or_name]
        p = Path(rel_or_name)
        if not p.is_absolute():
            candidate = self.project_root / p
            if not candidate.exists():
                candidate = self.outputs_dir / p
            p = candidate
        data = _load_json(p)
        self._json_cache[rel_or_name] = data
        return data

    def _build_data_counts_table(self) -> str:
        dq = self.data_quality
        rf = self.raw_facts
        rows = [
            ["Dataset file", str(rf.get("file")), "docs/raw_data_facts.json", "file"],
            ["Rows read", _fmt_int(dq.get("rows_read")), "outputs/data_quality.json", "rows_read"],
            ["Rows kept after validation", _fmt_int(dq.get("rows_kept")), "outputs/data_quality.json", "rows_kept"],
            ["Dropped: missing/null", _fmt_int(dq["dropped_by_rule"]["missing_required_columns_or_nulls"]), "outputs/data_quality.json", "dropped_by_rule.missing_required_columns_or_nulls"],
            ["Dropped: non-numeric", _fmt_int(dq["dropped_by_rule"]["non_numeric_values"]), "outputs/data_quality.json", "dropped_by_rule.non_numeric_values"],
            ["Dropped: invalid type/account", _fmt_int(dq["dropped_by_rule"]["invalid_type_or_account_filter"]), "outputs/data_quality.json", "dropped_by_rule.invalid_type_or_account_filter"],
            ["Total fraud rows (isFraud=1)", _fmt_int(dq.get("fraud_rows")), "outputs/data_quality.json", "fraud_rows"],
            ["Whole-file fraud prevalence", _fmt_pct(dq["fraud_rows"] / dq["rows_kept"], 4), "outputs/data_quality.json", "fraud_rows / rows_kept"],
            ["Fraud rows: CASH_OUT", _fmt_int(rf["fraud_rows_by_type"]["CASH_OUT"]), "docs/raw_data_facts.json", "fraud_rows_by_type.CASH_OUT"],
            ["Fraud rows: TRANSFER", _fmt_int(rf["fraud_rows_by_type"]["TRANSFER"]), "docs/raw_data_facts.json", "fraud_rows_by_type.TRANSFER"],
            ["Time steps (min .. max)", f"{rf['step_min']} .. {rf['step_max']}", "docs/raw_data_facts.json", "step_min, step_max"],
            ["Distinct origin accounts (nameOrig)", _fmt_int(rf.get("distinct_nameOrig_count")), "docs/raw_data_facts.json", "distinct_nameOrig_count"],
            ["Distinct destination accounts (nameDest)", _fmt_int(rf.get("distinct_nameDest_count")), "docs/raw_data_facts.json", "distinct_nameDest_count"],
            ["Distinct accounts (union)", _fmt_int(dq.get("distinct_accounts")), "outputs/data_quality.json", "distinct_accounts"],
            ["Destination prefix C (customer)", _fmt_int(rf["rows_by_dest_prefix"]["C"]), "docs/raw_data_facts.json", "rows_by_dest_prefix.C"],
            ["Destination prefix M (merchant)", _fmt_int(rf["rows_by_dest_prefix"]["M"]), "docs/raw_data_facts.json", "rows_by_dest_prefix.M"],
        ]
        for t_name, cnt in sorted(dq.get("by_type", {}).items()):
            rows.append([f"Rows by type: {t_name}", _fmt_int(cnt), "outputs/data_quality.json", f"by_type.{t_name}"])
        return _md_table(["Metric", "Value", "Source File", "Key"], rows)

    def _build_splits_overview_table(self) -> str:
        splits = self.evaluation.get("splits", {})
        rows: List[List[str]] = []
        for sp_name in ("time", "group_dest", "random"):
            sp_data = splits.get(sp_name, {})
            ref_model = sp_data.get("m1_tabular") or next(iter(sp_data.values()), {})
            conf = (ref_model.get("at_threshold") or {}).get("confusion", {})
            fraud_cnt = int(conf.get("tp", 0)) + int(conf.get("fn", 0))
            rows.append(
                [
                    sp_name,
                    _fmt_int(ref_model.get("rows")),
                    _fmt_int(fraud_cnt),
                    _fmt_pct(ref_model.get("prevalence"), 4),
                    str(ref_model.get("threshold_source")),
                    f"splits.{sp_name}.m1_tabular.(rows, prevalence, at_threshold.confusion)",
                ]
            )
        return _md_table(
            ["Split", "Test Rows", "Test Fraud Rows (tp+fn)", "Test Prevalence", "Threshold Source", "Source Key (`outputs/evaluation.json`)"],
            rows,
        )

    def _build_model_split_table(self, split_name: str) -> str:
        sp_data = self.evaluation.get("splits", {}).get(split_name, {})
        model_order = ["m0_rule", "m1_tabular", "m1_tabular_noerr", "m2_graph", "m3_community_mule"]
        rows: List[List[str]] = []
        for m in model_order:
            if m not in sp_data:
                continue
            md = sp_data[m]
            at = md.get("at_threshold", {})
            conf = at.get("confusion", {})
            rows.append(
                [
                    m,
                    _fmt_num(md.get("threshold"), 6),
                    str(md.get("threshold_source")),
                    _fmt_num(at.get("precision"), 4),
                    _fmt_num(at.get("recall"), 4),
                    _fmt_num(at.get("f1"), 4),
                    _fmt_num(md.get("average_precision"), 4),
                    _fmt_num(md.get("roc_auc"), 6),
                    _fmt_num(md.get("false_alarms_per_10k"), 2),
                    _fmt_int(conf.get("tp")),
                    _fmt_int(conf.get("fp")),
                    _fmt_int(conf.get("fn")),
                    _fmt_int(conf.get("tn")),
                ]
            )
        return _md_table(
            [
                "Model",
                "Threshold",
                "Threshold Source",
                "Precision",
                "Recall",
                "F1",
                "PR-AUC (AP)",
                "ROC-AUC",
                "False Alarms / 10k",
                "TP",
                "FP",
                "FN",
                "TN",
            ],
            rows,
        )

    def _build_business_impact_headline_table(self) -> str:
        bi = self.business_impact
        rows: List[List[str]] = []
        for item in bi.get("headline_operating_table", []):
            rows.append(
                [
                    str(item.get("model_key")),
                    _fmt_num(item.get("operating_threshold"), 6),
                    str(item.get("threshold_source")),
                    _fmt_int(item.get("alerts")),
                    _fmt_int(item.get("true_alerts")),
                    _fmt_int(item.get("false_alerts")),
                    _fmt_num(item.get("alerts_per_10k"), 2),
                    f"{float(item.get('fraud_value_caught', 0.0)):,.2f}",
                    f"{float(item.get('fraud_value_missed', 0.0)):,.2f}",
                    _fmt_pct(item.get("share_fraud_value_caught"), 2),
                    f"{float(item.get('net_financial_saving', 0.0)):,.2f}",
                ]
            )
        return _md_table(
            [
                "Model",
                "Operating Threshold",
                "Threshold Source",
                "Alerts",
                "True Alerts",
                "False Alerts",
                "Alerts / 10k",
                "Fraud Value Caught",
                "Fraud Value Missed",
                "Share Caught",
                "Net Saving (Review Cost 10,000 assumption)",
            ],
            rows,
        )

    def _build_business_impact_sensitivity_table(self) -> str:
        bi = self.business_impact
        rows: List[List[str]] = []
        for item in bi.get("sensitivity_table", []):
            mode = str(item.get("mode"))
            mode_label = (
                "validation_threshold"
                if mode == "validation_threshold"
                else "hindsight_best_test (not a recommended operating point)"
            )
            rows.append(
                [
                    _fmt_int(item.get("review_cost")),
                    str(item.get("model_key")),
                    mode_label,
                    _fmt_num(item.get("threshold"), 6),
                    _fmt_int(item.get("alerts")),
                    f"{float(item.get('fraud_value_caught', 0.0)):,.2f}",
                    f"{float(item.get('net_financial_saving', 0.0)):,.2f}",
                ]
            )
        return _md_table(
            [
                "Review Cost (assumption)",
                "Model",
                "Threshold Mode",
                "Threshold",
                "Alerts",
                "Fraud Value Caught",
                "Net Financial Saving",
            ],
            rows,
        )

    def _build_real_ring_table(self) -> str:
        ra = self.ring_analysis
        by_sz = ra.get("number_of_components_by_size", {})
        rows = [
            ["Fraud accounts in fraud subgraph", _fmt_int(ra.get("number_of_fraud_accounts")), "outputs/ring_analysis.json", "number_of_fraud_accounts"],
            ["Total fraud-subgraph components", _fmt_int(ra.get("number_of_components")), "outputs/ring_analysis.json", "number_of_components"],
            ["Components of size 2 (1-to-1 pairs)", _fmt_int(by_sz.get("2")), "outputs/ring_analysis.json", "number_of_components_by_size.2"],
            ["Components of size 3 (2-to-1 funnels)", _fmt_int(by_sz.get("3")), "outputs/ring_analysis.json", "number_of_components_by_size.3"],
            ["Components of size >= 3", _fmt_int(ra.get("count_size_ge_3")), "outputs/ring_analysis.json", "count_size_ge_3"],
            ["Components of size >= 4", _fmt_int(sum(int(v) for k, v in by_sz.items() if int(k) >= 4)), "outputs/ring_analysis.json", "number_of_components_by_size (>=4)"],
            ["Exported ring summary rows", _fmt_int(ra.get("exported_rings_count")), "outputs/ring_analysis.json", "exported_rings_count"],
        ]
        return _md_table(["Metric", "Value", "Source File", "Key"], rows)

    def _build_planted_ring_table(self, algo_key: str) -> str:
        algo = self.planted_rings.get("algorithms", {}).get(algo_key, {})
        rows: List[List[str]] = []
        for item in algo.get("results_by_pattern_and_camouflage", []):
            rows.append(
                [
                    str(item.get("pattern")),
                    str(item.get("camouflage")),
                    _fmt_int(item.get("planted")),
                    _fmt_int(item.get("recovered")),
                    _fmt_pct(item.get("recovery_rate"), 1),
                    _fmt_pct(item.get("ring_contained_rate"), 1),
                    _fmt_num(item.get("mean_ring_purity"), 4),
                    _fmt_int(item.get("largest_component_size_with_planted")),
                    _fmt_num(item.get("homogeneity"), 4),
                    _fmt_num(item.get("completeness"), 4),
                    _fmt_num(item.get("v_measure"), 4),
                    _fmt_num(item.get("contamination"), 4),
                ]
            )
        return _md_table(
            [
                "Pattern",
                "Camouflage",
                "Planted",
                "Recovered",
                "Strict Recovery",
                "Ring Contained",
                "Mean Purity",
                "Largest Component w/ Planted",
                "Homogeneity",
                "Completeness",
                "V-Measure",
                "Contamination",
            ],
            rows,
        )

    def _build_data_scaling_table(self) -> str:
        rows: List[List[str]] = []
        for row in self.scaling_runs.get("data_size_summary", []):
            indiv = ", ".join(_fmt_num(v, 2) for v in row.get("individual_total_seconds", [])) or "none"
            rows.append(
                [
                    f"{float(row['fraction']):.2f}",
                    f"{row['ok_count']}/{row['total_count']}",
                    ", ".join(row.get("statuses", [])),
                    str(row.get("superseded_count", 0)),
                    _fmt_int(row.get("mean_rows")),
                    _fmt_int(row.get("mean_nodes")),
                    _fmt_int(row.get("mean_edges")),
                    _fmt_num(row.get("mean_total_seconds"), 2),
                    indiv,
                    _fmt_bytes(row.get("mean_peak_rss_bytes")),
                    _fmt_bytes(row.get("min_available_ram_bytes")),
                    _fmt_bytes(row.get("mean_shuffle_bytes")),
                ]
            )
        return _md_table(
            [
                "Fraction",
                "Active OK/Total",
                "Active Statuses",
                "Superseded Prior",
                "Rows",
                "Nodes",
                "Edges",
                "Mean Total (s)",
                "Individual Repeats (s)",
                "Mean Peak RSS",
                "Min Avail RAM",
                "Mean Shuffle",
            ],
            rows,
        )

    def _build_core_scaling_table(self) -> str:
        rows: List[List[str]] = []
        for row in self.scaling_runs.get("core_scaling_summary", []):
            indiv_t = ", ".join(_fmt_num(v, 2) for v in row.get("individual_total_seconds", [])) or "none"
            indiv_sp = ", ".join(f"{_fmt_num(v, 3)}x" for v in row.get("individual_speedups", [])) or "none"
            rows.append(
                [
                    str(row.get("master")),
                    f"{float(row['fraction']):.2f}",
                    ", ".join(row.get("statuses", [])),
                    _fmt_num(row.get("mean_total_seconds"), 2),
                    indiv_t,
                    f"{_fmt_num(row.get('speedup'), 3)}x" if row.get("speedup") is not None else "n/a",
                    indiv_sp,
                    _fmt_num(row.get("efficiency"), 3),
                    _fmt_bytes(row.get("mean_peak_rss_bytes")),
                    _fmt_bytes(row.get("min_available_ram_bytes")),
                ]
            )
        return _md_table(
            [
                "Master",
                "Fraction",
                "Statuses",
                "Mean Total (s)",
                "Individual Repeats (s)",
                "Speedup",
                "Individual Speedups",
                "Efficiency",
                "Mean Peak RSS",
                "Min Avail RAM",
            ],
            rows,
        )

    def _build_scaling_superseded_table(self) -> str:
        rows: List[List[str]] = []
        for r in self.scaling_runs.get("superseded_runs", []):
            rows.append(
                [
                    str(r.get("suite")),
                    f"{float(r.get('fraction', 0)):.2f}",
                    str(r.get("master")),
                    str(r.get("repeat")),
                    str(r.get("status")),
                    _fmt_num(r.get("wall_seconds"), 2),
                    _fmt_bytes(r.get("peak_rss_bytes")),
                    _fmt_bytes(r.get("min_available_ram_bytes")),
                    str(r.get("reason", "")),
                ]
            )
        return _md_table(
            ["Suite", "Fraction", "Master", "Repeat", "Status", "Wall (s)", "Peak RSS", "Min Avail RAM", "Reason"],
            rows,
        )

    def _build_pandas_vs_spark_table(self) -> str:
        rows: List[List[str]] = []
        for row in self.pandas_vs_spark.get("summary", []):
            indiv_c = ", ".join(_fmt_num(v, 3) for v in row.get("individual_compute_seconds", [])) or "none"
            rows.append(
                [
                    str(row.get("task")),
                    str(row.get("tool")),
                    _fmt_int(row.get("rows")),
                    ", ".join(row.get("statuses", [])),
                    _fmt_num(row.get("mean_load_seconds"), 3),
                    _fmt_num(row.get("mean_compute_seconds"), 3),
                    indiv_c,
                    _fmt_bytes(row.get("mean_peak_rss_bytes") or row.get("max_peak_rss_bytes")),
                    _fmt_bytes(row.get("min_available_ram_bytes")),
                ]
            )
        return _md_table(
            [
                "Task",
                "Tool",
                "Rows",
                "Statuses",
                "Load Mean (s)",
                "Compute Mean (s)",
                "Compute Repeats (s)",
                "Peak RSS",
                "Min Avail RAM",
            ],
            rows,
        )

    def _build_storage_table(self) -> str:
        fmts = self.storage_benchmark.get("formats", {})
        speedups = self.storage_benchmark.get("speedup_vs_csv", {})
        rows: List[List[str]] = []
        for fmt_key in ("csv", "parquet_partitioned", "parquet_unpartitioned"):
            if fmt_key not in fmts:
                continue
            f_obj = fmts[fmt_key]
            ops = f_obj.get("operations", {})
            sp = speedups.get(fmt_key, {})
            rows.append(
                [
                    fmt_key,
                    _fmt_num(f_obj.get("size_mb"), 2),
                    f"{_fmt_num(f_obj.get('compression_ratio_vs_csv'), 3)}x",
                    f"{_fmt_num(ops['count_rows']['mean_seconds'], 4)} ({', '.join(_fmt_num(x, 2) for x in ops['count_rows']['repeats_seconds'])})",
                    f"{_fmt_num(ops['read_two_columns']['mean_seconds'], 4)} ({', '.join(_fmt_num(x, 2) for x in ops['read_two_columns']['repeats_seconds'])})",
                    f"{_fmt_num(ops['filter_transfer_fraud']['mean_seconds'], 4)} ({', '.join(_fmt_num(x, 2) for x in ops['filter_transfer_fraud']['repeats_seconds'])})",
                    "1.00x" if fmt_key == "csv" else f"{_fmt_num(sp.get('count_rows'), 2)}x / {_fmt_num(sp.get('read_two_columns'), 2)}x / {_fmt_num(sp.get('filter_transfer_fraud'), 2)}x",
                ]
            )
        return _md_table(
            [
                "Format",
                "Size (MiB)",
                "Compression vs CSV",
                "Count Rows Mean s (3 repeats)",
                "Read 2 Cols Mean s (3 repeats)",
                "Filter TRANSFER+Fraud Mean s (3 repeats)",
                "Speedup vs CSV (Count / 2-Col / Filter)",
            ],
            rows,
        )

    def _build_spark_engines_table(self) -> str:
        info = self.scalability.get("algorithm_info", {})
        timings = self.scalability.get("timings_seconds", {})
        rows: List[List[str]] = []
        for algo_key in ("degrees", "pagerank", "triangles", "components", "lpa", "behaviour"):
            a = info.get(algo_key, {})
            t_key = "behaviour_features" if algo_key == "behaviour" else algo_key
            rows.append(
                [
                    algo_key,
                    str(a.get("engine", self.scalability.get("engines", {}).get(algo_key, "not available"))),
                    str(a.get("iterations", "not available")),
                    str(a.get("converged", "not available")),
                    _fmt_num(timings.get(t_key), 2) if t_key in timings else "not available",
                    f"outputs/scalability.json: algorithm_info.{algo_key}, timings_seconds.{t_key}",
                ]
            )
        return _md_table(
            ["Algorithm", "Engine", "Iterations", "Converged", "Full-Run Stage Time (s)", "Source Key"],
            rows,
        )

    def _build_pipeline_timings_table(self) -> str:
        sc = self.scalability
        t = sc.get("timings_seconds", {})
        rows = [
            ["ingestion_and_validation", _fmt_num(t.get("ingestion_and_validation"), 2), "outputs/scalability.json", "timings_seconds.ingestion_and_validation"],
            ["graph_construction", _fmt_num(t.get("graph_construction"), 2), "outputs/scalability.json", "timings_seconds.graph_construction"],
            ["graph_feature_engineering (phase)", _fmt_num(t.get("graph_feature_engineering"), 2), "outputs/scalability.json", "timings_seconds.graph_feature_engineering"],
            ["model_training (3 splits)", _fmt_num(t.get("model_training"), 2), "outputs/scalability.json", "timings_seconds.model_training"],
            ["distributed_scoring_and_eval (3 splits)", _fmt_num(t.get("distributed_scoring_and_eval"), 2), "outputs/scalability.json", "timings_seconds.distributed_scoring_and_eval"],
            ["Total Spark shuffle read", _fmt_bytes(sc.get("shuffle_read_bytes")), "outputs/scalability.json", "shuffle_read_bytes"],
            ["Total Spark shuffle write", _fmt_bytes(sc.get("shuffle_write_bytes")), "outputs/scalability.json", "shuffle_write_bytes"],
            ["Spark completed stage count", _fmt_int(sc.get("stage_count")), "outputs/scalability.json", "stage_count"],
            ["Sampled training rows (3 splits sum)", _fmt_int(sc.get("sampled_training_rows")), "outputs/scalability.json", "sampled_training_rows"],
            ["Memory warning recorded", str(sc.get("memory_warning")), "outputs/scalability.json", "memory_warning"],
        ]
        return _md_table(["Phase / Metric", "Value", "Source File", "Key"], rows)

    def _build_environment_table(self) -> str:
        ri = self.run_info
        req_lines = [
            ln.strip()
            for ln in (self.project_root / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        rows = [
            ["Python", str(ri.get("python")), "outputs/run_info.json", "python"],
            ["Java", str(ri.get("java")), "outputs/run_info.json", "java"],
            ["PySpark", str(ri.get("pyspark")), "outputs/run_info.json", "pyspark"],
            ["pandas", str(ri.get("pandas")), "outputs/run_info.json", "pandas"],
            ["scikit-learn", str(ri.get("sklearn")), "outputs/run_info.json", "sklearn"],
            ["XGBoost", str(ri.get("xgboost")), "outputs/run_info.json", "xgboost"],
            ["CPU cores (physical / logical)", f"{ri['cores']['physical']} / {ri['cores']['logical']}", "outputs/run_info.json", "cores.physical, cores.logical"],
            ["Total RAM (GiB)", _fmt_num(ri["ram_gb"]["total"], 2), "outputs/run_info.json", "ram_gb.total"],
            ["Spark driver memory", str(ri["config"]["SPARK_DRIVER_MEMORY"]), "outputs/run_info.json", "config.SPARK_DRIVER_MEMORY"],
            ["Spark shuffle partitions", str(ri["config"]["SPARK_SHUFFLE_PARTITIONS"]), "outputs/run_info.json", "config.SPARK_SHUFFLE_PARTITIONS"],
            ["Max training sample rows per split", _fmt_int(ri["config"]["max_training_rows"]), "outputs/run_info.json", "config.max_training_rows"],
            ["Dataset SHA-256", str(ri["dataset"]["sha256"]), "outputs/run_info.json", "dataset.sha256"],
            ["Full-run git commit", str(ri.get("git_commit")), "outputs/run_info.json", "git_commit"],
            ["Pinned packages (`requirements.txt`)", ", ".join(req_lines), "requirements.txt", "all non-comment lines"],
        ]
        return _md_table(["Component", "Measured Value", "Source File", "Key"], rows)

    def _build_all_tables(self) -> Dict[str, str]:
        return {
            "data_counts": self._build_data_counts_table(),
            "splits_overview": self._build_splits_overview_table(),
            "models_time": self._build_model_split_table("time"),
            "models_group_dest": self._build_model_split_table("group_dest"),
            "models_random": self._build_model_split_table("random"),
            "business_impact_headline": self._build_business_impact_headline_table(),
            "business_impact_sensitivity": self._build_business_impact_sensitivity_table(),
            "real_ring": self._build_real_ring_table(),
            "planted_rings_cc": self._build_planted_ring_table("connected_components"),
            "planted_rings_lpa": self._build_planted_ring_table("label_propagation"),
            "data_scaling": self._build_data_scaling_table(),
            "core_scaling": self._build_core_scaling_table(),
            "scaling_superseded": self._build_scaling_superseded_table(),
            "pandas_vs_spark": self._build_pandas_vs_spark_table(),
            "storage_benchmark": self._build_storage_table(),
            "spark_engines": self._build_spark_engines_table(),
            "pipeline_timings": self._build_pipeline_timings_table(),
            "environment": self._build_environment_table(),
        }

    def render_template_text(self, template_text: str) -> str:
        def _repl_table(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in self.tables:
                raise KeyError(f"Unknown table placeholder: {{{{table:{name}}}}}")
            return self.tables[name]

        def _repl_num(match: re.Match[str]) -> str:
            file_ref = match.group(1)
            json_path = match.group(2)
            fmt = match.group(3)
            data = self.get_json(file_ref)
            val = _lookup_json_path(data, json_path)
            return _format_value(val, fmt)

        def _repl_text(match: re.Match[str]) -> str:
            rel_path = match.group(1)
            p = self.project_root / rel_path
            return p.read_text(encoding="utf-8").strip()

        out = TABLE_PLACEHOLDER_RE.sub(_repl_table, template_text)
        out = NUM_PLACEHOLDER_RE.sub(_repl_num, out)
        out = TEXT_PLACEHOLDER_RE.sub(_repl_text, out)
        return out


def build_results_tables_md(ctx: ReportContext) -> str:
    return f"""# Results Tables (Generated from `outputs/`)

All tables in this document are generated programmatically by `make_report.py` from `outputs/` and `docs/raw_data_facts.json`. No numbers are hand-typed.

## 1. Dataset Counts and Data Quality
Source: `outputs/data_quality.json`, `docs/raw_data_facts.json`.

{ctx.tables["data_counts"]}

## 2. Evaluation Splits and Prevalence
Source: `outputs/evaluation.json`. Models are trained independently on each split's training partition; score thresholds are selected on each split's validation partition (maximising F1 at true prevalence) and never tuned on test rows.

{ctx.tables["splits_overview"]}

## 3. Model Comparison — Primary Time Split
Source: `outputs/evaluation.json` (`splits.time.*`).

{ctx.tables["models_time"]}

## 4. Model Comparison — Secondary Group Split (Destination Account `nameDest`)
Source: `outputs/evaluation.json` (`splits.group_dest.*`).

{ctx.tables["models_group_dest"]}

## 5. Model Comparison — Reference Random Split
Source: `outputs/evaluation.json` (`splits.random.*`).

{ctx.tables["models_random"]}

## 6. Business Impact — Headline Operating Points (Time Split)
Source: `outputs/business_impact.json` (`headline_operating_table`). Operating points use the validation-selected thresholds from `outputs/evaluation.json`. Review cost = `{_fmt_int(ctx.business_impact["default_review_cost"])}` `{ctx.business_impact["currency"]}` (`{ctx.business_impact["review_cost_label"]}`). Total fraud value on time-split test set (`{_fmt_int(ctx.business_impact["test_rows"])}` rows, `{_fmt_int(ctx.business_impact["fraud_rows"])}` fraud rows) = `{float(ctx.business_impact["fraud_value_total"]):,.2f}`.

{ctx.tables["business_impact_headline"]}

## 7. Business Impact — Review-Cost Sensitivity (1,000 / 10,000 / 100,000)
Source: `outputs/business_impact.json` (`sensitivity_table`). `hindsight_best_test` rows are reported strictly for sensitivity inspection and are not recommended operating points.

{ctx.tables["business_impact_sensitivity"]}

## 8. Real-Data Fraud Subgraph Component Analysis
Source: `outputs/ring_analysis.json`.

{ctx.tables["real_ring"]}

Empirical finding (`outputs/ring_analysis.json: finding`):
> {ctx.ring_analysis["finding"]}

## 9. Planted-Ring Synthetic Injection Test — Connected Components
Source: `outputs/planted_ring_validation.json` (`algorithms.connected_components`). Label: `{ctx.planted_rings["label"]}`, sample size: `{_fmt_int(ctx.planted_rings["sample_size"])}` rows, total planted rings: `{_fmt_int(ctx.planted_rings["total_planted_rings"])}`, strict recovered: `{_fmt_int(ctx.planted_rings["algorithms"]["connected_components"]["total_recovered_rings"])}` (`{_fmt_pct(ctx.planted_rings["algorithms"]["connected_components"]["overall_recovery_rate"], 1)}`).

{ctx.tables["planted_rings_cc"]}

## 10. Planted-Ring Synthetic Injection Test — Label Propagation (LPA)
Source: `outputs/planted_ring_validation.json` (`algorithms.label_propagation`). Strict recovered: `{_fmt_int(ctx.planted_rings["algorithms"]["label_propagation"]["total_recovered_rings"])}` / `{_fmt_int(ctx.planted_rings["total_planted_rings"])}` (`{_fmt_pct(ctx.planted_rings["algorithms"]["label_propagation"]["overall_recovery_rate"], 1)}`).

{ctx.tables["planted_rings_lpa"]}

## 11. Spark Data-Size Scaling (with Superseded Prior Aborts Retained)
Source: `outputs/scaling_runs.json` (`data_size_summary`).

{ctx.tables["data_scaling"]}

### Superseded Prior Attempts Retained (`outputs/scaling_runs.json: superseded_runs`)
{ctx.tables["scaling_superseded"]}

## 12. Spark Core Scaling (Fraction `{ctx.scaling_runs["core_scaling_fraction"]:.2f}`)
Source: `outputs/scaling_runs.json` (`core_scaling_summary`).

{ctx.tables["core_scaling"]}

## 13. pandas/NetworkX vs Spark-Native Comparison (Including Full `{_fmt_int(ctx.data_quality["rows_kept"])}`-Row pandas Run)
Source: `outputs/pandas_vs_spark.json` (`summary`).

{ctx.tables["pandas_vs_spark"]}

## 14. Storage Benchmark: CSV vs Parquet
Source: `outputs/storage_benchmark.json` (`formats`, `speedup_vs_csv`).

{ctx.tables["storage_benchmark"]}

## 15. Spark Graph Engines and Full-Run Pipeline Telemetry
Source: `outputs/scalability.json`.

{ctx.tables["spark_engines"]}

{ctx.tables["pipeline_timings"]}

## 16. Machine, Spark Settings, and Software Versions
Source: `outputs/run_info.json`, `requirements.txt`.

{ctx.tables["environment"]}
"""


def build_facts_for_paper_md(ctx: ReportContext) -> str:
    glossary_text = (ctx.project_root / "docs" / "GLOSSARY.md").read_text(encoding="utf-8").strip()
    provenance_text = (ctx.outputs_dir / "PROVENANCE.txt").read_text(encoding="utf-8").strip()
    cc_algo = ctx.planted_rings["algorithms"]["connected_components"]
    lpa_algo = ctx.planted_rings["algorithms"]["label_propagation"]

    # Build complete per-run scaling status table (all 21 records)
    scaling_all_rows: List[List[str]] = []
    for idx, r in enumerate(ctx.scaling_runs.get("runs", []), start=1):
        secs = r.get("seconds") or {}
        scaling_all_rows.append(
            [
                str(idx),
                str(r.get("suite")),
                f"{float(r.get('fraction', 0)):.2f}",
                str(r.get("master")),
                str(r.get("repeat")),
                str(r.get("status")),
                "yes" if r.get("superseded") else "no",
                _fmt_num(r.get("wall_seconds"), 2),
                _fmt_num(secs.get("total"), 2),
                _fmt_bytes(r.get("peak_rss_bytes")),
                _fmt_bytes(r.get("min_available_ram_bytes")),
            ]
        )
    scaling_all_table = _md_table(
        ["#", "Suite", "Fraction", "Master", "Repeat", "Status", "Superseded", "Wall (s)", "Graph Total (s)", "Peak RSS", "Min Avail RAM"],
        scaling_all_rows,
    )

    pvs_non_ok_rows: List[List[str]] = []
    for r in ctx.pandas_vs_spark.get("non_ok_runs", []):
        pvs_non_ok_rows.append(
            [
                str(r.get("tool")),
                str(r.get("task")),
                _fmt_int(r.get("target_rows")),
                str(r.get("repeat")),
                str(r.get("status")),
                _fmt_num(r.get("wall_seconds"), 2),
                _fmt_bytes(r.get("peak_rss_bytes")),
                _fmt_bytes(r.get("min_available_ram_bytes")),
                str(r.get("reason", "")),
            ]
        )
    pvs_non_ok_table = _md_table(
        ["Tool", "Task", "Rows", "Repeat", "Status", "Wall (s)", "Peak RSS", "Min Avail RAM", "Reason"],
        pvs_non_ok_rows,
    )

    xgb_cfg = ctx.model_config.get("xgboost", {})
    xgb_rows = [
        [k, str(v), "configs/model_config.json", f"xgboost.{k}"]
        for k, v in xgb_cfg.items()
    ]
    xgb_table = _md_table(["Hyperparameter", "Value", "Source File", "Key"], xgb_rows)

    ds_025_row = next((r for r in ctx.scaling_runs.get("data_size_summary", []) if round(float(r.get("fraction", 0)), 2) == 0.25), {})
    ds_025_min_ram = _fmt_bytes(ds_025_row.get("min_available_ram_bytes"))
    sup_cnt = _fmt_int(ctx.scaling_runs.get("superseded_runs_count"))
    planted_n = _fmt_int(ctx.planted_rings.get("total_planted_rings"))
    planted_sample = _fmt_int(ctx.planted_rings.get("sample_size"))
    full_rows_str = _fmt_int(ctx.data_quality.get("rows_kept"))

    return f"""# Facts for Paper (Programmatically Extracted from `outputs/`)

Every number, table, engine name, software version, and artifact path in this file was extracted programmatically by `make_report.py`. Do not hand-edit numbers; regenerate with `python make_report.py`.

## Accepted Pipeline Provenance (`outputs/PROVENANCE.txt`)
```text
{provenance_text}
```

## 1. Data Counts (`outputs/data_quality.json`, `docs/raw_data_facts.json`)
{ctx.tables["data_counts"]}

## 2. Splits, Prevalence, and Threshold Source (`outputs/evaluation.json`)
{ctx.tables["splits_overview"]}

### 2a. Time Split (Primary) — `outputs/evaluation.json: splits.time.*`
{ctx.tables["models_time"]}

### 2b. Destination-Group Split (`nameDest`, Secondary) — `outputs/evaluation.json: splits.group_dest.*`
{ctx.tables["models_group_dest"]}

### 2c. Random Split (Reference) — `outputs/evaluation.json: splits.random.*`
{ctx.tables["models_random"]}

### 2d. Fixed Model Hyperparameters (`configs/model_config.json`)
Identical across `m1_tabular`, `m1_tabular_noerr`, `m2_graph`, and `m3_community_mule`, chosen once and never tuned on test rows.

{xgb_table}

## 3. Business Impact (`outputs/business_impact.json`)
- Split: `{ctx.business_impact["split"]}` (`outputs/business_impact.json: split`)
- Currency unit: `{ctx.business_impact["currency"]}` (`outputs/business_impact.json: currency`)
- Test rows: `{_fmt_int(ctx.business_impact["test_rows"])}` (`outputs/business_impact.json: test_rows`)
- Fraud rows: `{_fmt_int(ctx.business_impact["fraud_rows"])}` (`outputs/business_impact.json: fraud_rows`)
- Total fraud value: `{float(ctx.business_impact["fraud_value_total"]):,.2f}` (`outputs/business_impact.json: fraud_value_total`)
- Default review cost: `{_fmt_int(ctx.business_impact["default_review_cost"])}` (`{ctx.business_impact["review_cost_label"]}`) (`outputs/business_impact.json: default_review_cost`, `review_cost_label`)

### 3a. Headline Operating Points (`outputs/business_impact.json: headline_operating_table`)
{ctx.tables["business_impact_headline"]}

### 3b. Sensitivity Across Review Costs 1,000 / 10,000 / 100,000 (`outputs/business_impact.json: sensitivity_table`)
{ctx.tables["business_impact_sensitivity"]}

## 4. Real-Data Fraud Subgraph Finding (`outputs/ring_analysis.json`)
{ctx.tables["real_ring"]}

Exact finding text (`outputs/ring_analysis.json: finding`):
> {ctx.ring_analysis["finding"]}

## 5. Planted-Ring Synthetic Injection Validation (`outputs/planted_ring_validation.json`)
- Label: `{ctx.planted_rings["label"]}` (`outputs/planted_ring_validation.json: label`)
- Sample size: `{_fmt_int(ctx.planted_rings["sample_size"])}` rows (`outputs/planted_ring_validation.json: sample_size`)
- Total planted rings: `{_fmt_int(ctx.planted_rings["total_planted_rings"])}` (`outputs/planted_ring_validation.json: total_planted_rings`)
- Spark config: driver `{ctx.planted_rings["spark_config"]["driver_memory"]}`, shuffle partitions `{ctx.planted_rings["spark_config"]["shuffle_partitions"]}` (`outputs/planted_ring_validation.json: spark_config`)
- Connected Components overall strict recovery: `{_fmt_int(cc_algo["total_recovered_rings"])}` / `{_fmt_int(cc_algo["total_planted_rings"])}` (`{_fmt_pct(cc_algo["overall_recovery_rate"], 1)}`) (`outputs/planted_ring_validation.json: algorithms.connected_components.overall_recovery_rate`)
- Label Propagation (LPA) overall strict recovery: `{_fmt_int(lpa_algo["total_recovered_rings"])}` / `{_fmt_int(lpa_algo["total_planted_rings"])}` (`{_fmt_pct(lpa_algo["overall_recovery_rate"], 1)}`) (`outputs/planted_ring_validation.json: algorithms.label_propagation.overall_recovery_rate`)

### 5a. Connected Components (`outputs/planted_ring_validation.json: algorithms.connected_components.results_by_pattern_and_camouflage`)
{ctx.tables["planted_rings_cc"]}

### 5b. Label Propagation (`outputs/planted_ring_validation.json: algorithms.label_propagation.results_by_pattern_and_camouflage`)
{ctx.tables["planted_rings_lpa"]}

## 6. Spark Data-Size and Core Scaling (`outputs/scaling_runs.json`)
- Total records retained: `{_fmt_int(ctx.scaling_runs["total_runs_recorded"])}` (`{_fmt_int(ctx.scaling_runs["active_runs_count"])}` active, `{_fmt_int(ctx.scaling_runs["superseded_runs_count"])}` superseded, `{_fmt_int(ctx.scaling_runs["ok_runs_count"])}` active ok, `{_fmt_int(ctx.scaling_runs["non_ok_runs_count"])}` active non-ok) (`outputs/scaling_runs.json: total_runs_recorded, active_runs_count, superseded_runs_count, ok_runs_count, non_ok_runs_count`)
- Active benchmark wall seconds: `{_fmt_num(ctx.scaling_runs["active_benchmark_wall_seconds"], 2)}` (`outputs/scaling_runs.json: active_benchmark_wall_seconds`)
- Total wall seconds including superseded: `{_fmt_num(ctx.scaling_runs["total_benchmark_wall_seconds"], 2)}` (`outputs/scaling_runs.json: total_benchmark_wall_seconds`)
- Core scaling baseline `local[1]` mean seconds at fraction `{ctx.scaling_runs["core_scaling_fraction"]:.2f}`: `{_fmt_num(ctx.scaling_runs["core_scaling_baseline_local1_seconds"], 2)}` (`outputs/scaling_runs.json: core_scaling_baseline_local1_seconds`)

### 6a. Data-Size Scaling Summary (`outputs/scaling_runs.json: data_size_summary`)
{ctx.tables["data_scaling"]}

### 6b. Core-Scaling Summary (`outputs/scaling_runs.json: core_scaling_summary`)
{ctx.tables["core_scaling"]}

### 6c. Superseded Prior Aborts Retained (`outputs/scaling_runs.json: superseded_runs`)
{ctx.tables["scaling_superseded"]}

### 6d. Complete 21-Record Per-Run Status Table (`outputs/scaling_runs.json: runs`)
{scaling_all_table}

## 7. pandas/NetworkX vs Spark (`outputs/pandas_vs_spark.json`)
- Total runs recorded: `{_fmt_int(ctx.pandas_vs_spark["total_runs_recorded"])}` (`{_fmt_int(ctx.pandas_vs_spark["ok_runs_count"])}` ok, `{_fmt_int(ctx.pandas_vs_spark["non_ok_runs_count"])}` non-ok) (`outputs/pandas_vs_spark.json: total_runs_recorded, ok_runs_count, non_ok_runs_count`)
- Total benchmark wall seconds: `{_fmt_num(ctx.pandas_vs_spark["total_benchmark_wall_seconds"], 2)}` (`outputs/pandas_vs_spark.json: total_benchmark_wall_seconds`)

### 7a. Summary Table Including Full `{_fmt_int(ctx.data_quality["rows_kept"])}`-Row pandas Run (`outputs/pandas_vs_spark.json: summary`)
{ctx.tables["pandas_vs_spark"]}

### 7b. Non-OK Runs (`outputs/pandas_vs_spark.json: non_ok_runs`)
{pvs_non_ok_table}

## 8. Storage Benchmark: CSV vs Parquet (`outputs/storage_benchmark.json`)
- Repeats per query: `{ctx.storage_benchmark["repeats"]}` (`outputs/storage_benchmark.json: repeats`)
- Minimum available RAM recorded: `{_fmt_bytes(ctx.storage_benchmark["min_available_ram_bytes"])}` (`outputs/storage_benchmark.json: min_available_ram_bytes`)

{ctx.tables["storage_benchmark"]}

## 9. Spark Engines, Full-Run Telemetry, Machine and Software Versions
### 9a. Graph Algorithm Engines (`outputs/scalability.json: engines, algorithm_info`)
{ctx.tables["spark_engines"]}

### 9b. Full Pipeline Timings and Shuffle Totals (`outputs/scalability.json`)
{ctx.tables["pipeline_timings"]}

### 9c. Machine, Spark Configuration, and Software Versions (`outputs/run_info.json`, `requirements.txt`)
{ctx.tables["environment"]}

## 10. Metric Glossary (`docs/GLOSSARY.md`)
{glossary_text}

## 11. Limitations (Must Accompany All Claims)
1. **Single machine**: All Spark runs and benchmarks execute in `local[*]` or `local[n]` mode on one Windows 11 laptop (`{ctx.run_info["ram_gb"]["total"]:.2f} GiB` RAM, local NTFS disk); measurements do not represent multi-node network shuffle, HDFS/S3 I/O, or YARN/Kubernetes cluster scheduling.
2. **Shared RAM, memory pressure, and repeat spread**: The host OS shares `{ctx.run_info["ram_gb"]["total"]:.2f} GiB` RAM with the Spark JVM (`{ctx.run_info["config"]["SPARK_DRIVER_MEMORY"]}` driver heap plus off-heap/shuffle buffers). `{sup_cnt}` initial scaling runs hit the `< 0.5 GiB / 60s` memory guard and were re-run under `< 0.2 GiB / 120s` (original aborts retained as superseded); even among `ok` runs, memory pressure creates measurable repeat spread (at `fraction=0.25`, minimum available RAM dipped to `{ds_025_min_ram}`).
3. **Hybrid CPU cores**: The Intel Core Ultra 7 155H mixes performance and efficiency cores (`{ctx.run_info["cores"]["physical"]}` physical / `{ctx.run_info["cores"]["logical"]}` logical), causing uneven thread scaling across `local[1..16]`.
4. **Simulated data**: PaySim is a synthetic mobile-money simulator whose fraud mechanism concentrates on isolated 1-to-1 `TRANSFER` and `CASH_OUT` pairs (zero real rings of size >= 4) and exhibits strong balance-accounting signals (`errorBalanceOrig`, `errorBalanceDest`). Accordingly, `m1_tabular_noerr` is reported alongside `m1_tabular`, `m2_graph`, and `m3_community_mule`.
5. **Transductive, label-free graph features**: Account-level graph features (degrees, PageRank, triangle count, connected components, LPA, flow ratios) are label-free but computed across the full transaction graph rather than strictly past-only per split.
6. **Uniform transaction sampling in scaling benchmarks**: Sampling transactions uniformly (`seed=42`) changes graph edge density and component connectivity across fractions; node and edge counts are reported per run alongside runtime.
7. **Two repeats in long benchmarks**: Scaling and pandas-vs-Spark benchmarks use 2 repeats per configuration (3 repeats for storage); individual repeat values are always reported beside means.
8. **Planted-ring evaluation uses one `{planted_sample}`-row sample**: Synthetic ring recovery results (`outputs/planted_ring_validation.json`) come from injecting `{planted_n}` rings into a single `{planted_sample}`-row PaySim sample rather than multiple independent background samples or the full `{full_rows_str}`-row graph.
9. **pandas/NetworkX comparison scope**: The single-node comparison covers only `degree` counts and `connected_components`; it does not benchmark PageRank, triangle counting, LPA, feature joins, or end-to-end model scoring in NetworkX.
"""


def sync_results_snapshot(outputs_dir: Path, snapshot_dir: Path) -> List[str]:
    """Copy small final JSON, CSV, and PNG result artifacts into results_snapshot/ and write SHA256SUMS.txt."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    visuals_dst = snapshot_dir / "visuals"
    visuals_dst.mkdir(parents=True, exist_ok=True)

    def _copy_writable(src: Path, dst: Path) -> None:
        if dst.exists():
            dst.chmod(stat.S_IWRITE | stat.S_IREAD)
        shutil.copyfile(src, dst)
        dst.chmod(stat.S_IWRITE | stat.S_IREAD)

    small_files = [
        "business_impact.json",
        "data_quality.json",
        "evaluation.json",
        "feature_importance.json",
        "pandas_vs_spark.csv",
        "pandas_vs_spark.json",
        "planted_ring_validation.json",
        "PROVENANCE.txt",
        "ring_analysis.json",
        "rings_summary.csv",
        "run_info.json",
        "scalability.json",
        "scaling_runs.csv",
        "scaling_runs.json",
        "storage_benchmark.json",
    ]
    copied: List[str] = []
    for name in small_files:
        src = outputs_dir / name
        if src.exists() and src.is_file():
            _copy_writable(src, snapshot_dir / name)
            copied.append(name)

    visuals_src = outputs_dir / "visuals"
    if visuals_src.exists():
        for png in sorted(visuals_src.glob("*.png")):
            _copy_writable(png, visuals_dst / png.name)
            copied.append(f"visuals/{png.name}")

    write_manifest(snapshot_dir)
    return copied


def render_all(
    outputs_dir: Path,
    templates_dir: Path,
    dest_root: Path,
    project_root: Path,
    update_snapshot: bool = False,
) -> Dict[str, Path]:
    """Generate docs/RESULTS_TABLES.md, docs/FACTS_FOR_PAPER.md, and all templates under templates_dir."""
    ctx = ReportContext(outputs_dir=outputs_dir, project_root=project_root)
    generated: Dict[str, Path] = {}

    docs_dest = dest_root / "docs"
    docs_dest.mkdir(parents=True, exist_ok=True)

    results_tables_path = docs_dest / "RESULTS_TABLES.md"
    results_tables_path.write_text(build_results_tables_md(ctx), encoding="utf-8", newline="\n")
    generated["docs/RESULTS_TABLES.md"] = results_tables_path

    facts_path = docs_dest / "FACTS_FOR_PAPER.md"
    facts_path.write_text(build_facts_for_paper_md(ctx), encoding="utf-8", newline="\n")
    generated["docs/FACTS_FOR_PAPER.md"] = facts_path

    if templates_dir.exists():
        for tmpl_path in sorted(templates_dir.glob("*.tmpl")):
            raw_text = tmpl_path.read_text(encoding="utf-8")
            rendered = ctx.render_template_text(raw_text)
            target_name = tmpl_path.name[: -len(".tmpl")]
            if target_name.upper() == "README.MD":
                out_path = dest_root / "README.md"
                rel_key = "README.md"
            else:
                out_path = docs_dest / target_name
                rel_key = f"docs/{target_name}"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(rendered, encoding="utf-8", newline="\n")
            generated[rel_key] = out_path

    if update_snapshot:
        sync_results_snapshot(outputs_dir, dest_root / "results_snapshot")

    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate docs tables, facts sheet, and templated markdown documents.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--templates-dir", type=Path, default=Path("docs/templates"))
    parser.add_argument("--dest-root", type=Path, default=Path("."))
    parser.add_argument("--no-snapshot", action="store_true", help="Do not sync results_snapshot/.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    update_snap = (not args.no_snapshot) and (args.dest_root.resolve() == project_root.resolve())
    generated = render_all(
        outputs_dir=args.outputs_dir,
        templates_dir=args.templates_dir,
        dest_root=args.dest_root,
        project_root=project_root,
        update_snapshot=update_snap,
    )
    for rel, path in sorted(generated.items()):
        print(f"Generated {rel} -> {path} ({path.stat().st_size} bytes)")
    if update_snap:
        print("Synced results_snapshot/ and updated results_snapshot/SHA256SUMS.txt")


if __name__ == "__main__":
    main()
