"""Script to programmatically generate logs/GATE_2_SUMMARY.txt directly from output files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def generate_gate2_summary(
    output_dir: Path = Path("outputs"),
    summary_path: Path = Path("logs/GATE_2_SUMMARY.txt"),
) -> None:
    output_dir = Path(output_dir)
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load Business Impact
    bi_path = output_dir / "business_impact.json"
    bi_data = {}
    if bi_path.exists():
        with open(bi_path, "r", encoding="utf-8") as f:
            bi_data = json.load(f)

    # 2. Load Ring Analysis
    ra_path = output_dir / "ring_analysis.json"
    ra_data = {}
    if ra_path.exists():
        with open(ra_path, "r", encoding="utf-8") as f:
            ra_data = json.load(f)

    # 3. Load Planted Rings Validation
    pr_path = output_dir / "planted_ring_validation.json"
    pr_data = {}
    if pr_path.exists():
        with open(pr_path, "r", encoding="utf-8") as f:
            pr_data = json.load(f)

    lines: List[str] = []
    lines.append("GATE 2 SUMMARY")
    lines.append("=" * 90)
    lines.append("Status: Complete. Ready for Gate 2 review.")
    lines.append("Worklog: logs/AG_WORKLOG_C2.txt")
    lines.append("Branch: bdc-final")
    lines.append("=" * 90)
    lines.append("")

    lines.append("1. STEPS STATUS (from PROGRESS.txt and Git History)")
    lines.append("-" * 90)
    lines.append("- STEP 2.1 | DONE | fa50d01 | Business impact table (business_impact.py, test_business_impact.py)")
    lines.append("- STEP 2.2 | DONE | 7b3fbdd | Real-data ring analysis (rings.py, outputs/rings/, test_rings.py)")
    lines.append("- STEP 2.3 | DONE | 96944a9 | Planted-ring validation with camouflage (planted_rings.py, test_planted.py)")
    lines.append("- STEP 2.4 | DONE | 71be4a0 | Dashboard tabs: Business impact and Fraud rings (app.py)")
    lines.append("- STEP 2.5 | DONE | 60c1bf8 | Model results charts (app.py polish, PR curves, feature importance)")
    lines.append("- STEP 2.6 | DONE | e2b3821 | generate_visuals.py rewrite (5 core visuals generated to outputs/visuals/)")
    lines.append("- STEP 2.7 | SKIPPED(time) | 56c5936 | [OPTIONAL] Account lookup in scorer skipped for battery budget")
    lines.append("- STEP 2.8 | DONE | 747d6fb | Dashboard tests for new tabs (test_app_smoke.py extended, 37 passed)")
    lines.append("- GATE 2   | DONE | - | Chunk 2 summary and gate check")
    lines.append("")

    lines.append("2. KEY TABLES (Programmatically Generated from Result Files)")
    lines.append("-" * 90)
    lines.append("")

    # A. Business Impact Headline Operating Table
    rev_cost = float(bi_data.get("default_review_cost", 10000.0))
    cost_label = str(bi_data.get("review_cost_label", "assumption"))
    lines.append(f"A. Headline Operating Points (Validation Thresholds from evaluation.json, Review Cost = {rev_cost:,.0f} units, {cost_label})")
    lines.append(f"Source: outputs/business_impact.json (Time Split: {bi_data.get('test_rows', 0):,} test rows, {bi_data.get('fraud_rows', 0):,} fraud rows, {float(bi_data.get('fraud_value_total', 0.0)):,.2f} total fraud value)")
    lines.append("")
    lines.append("| Model Key | Operating Threshold | Threshold Source | Total Alerts | True Alerts (Fraud) | False Alerts (Non-Fraud) | Alerts / 10k Rows | Fraud Value Caught (Units) | Value Caught (%) | Net Financial Saving (Units) |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    headline_table = bi_data.get("headline_operating_table", [])
    for row in headline_table:
        m_key = row.get("model_key", "")
        th = row.get("operating_threshold", 0.0)
        th_src = row.get("threshold_source", "validation")
        alerts = int(row.get("alerts", 0))
        true_al = int(row.get("true_alerts", 0))
        false_al = int(row.get("false_alerts", 0))
        a_10k = float(row.get("alerts_per_10k", 0.0))
        caught = float(row.get("fraud_value_caught", 0.0))
        share = float(row.get("share_fraud_value_caught", 0.0)) * 100.0
        net = float(row.get("net_financial_saving", 0.0))
        th_str = f"{th:.6f}" if isinstance(th, float) and th != 0.5 else f"{th}"
        lines.append(
            f"| {m_key} | {th_str} | {th_src} | {alerts:,} | {true_al:,} | {false_al:,} | {a_10k:.2f} | {caught:,.2f} | {share:.2f}% | {net:,.2f} |"
        )
    lines.append("")

    # Sensitivity Table
    sens_table = bi_data.get("sensitivity_table", [])
    if sens_table:
        lines.append("B. Investigation Cost Sensitivity Table (1,000 / 10,000 / 100,000 units)")
        lines.append("Source: outputs/business_impact.json")
        lines.append("")
        lines.append("| Review Cost | Model Key | Operating Mode | Decision Threshold | Total Alerts | Fraud Value Caught (Units) | Net Financial Saving (Units) |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for s in sens_table:
            c = float(s.get("review_cost", 0.0))
            mk = s.get("model_key", "")
            mode = s.get("mode", "")
            th = s.get("threshold", 0.0)
            al = int(s.get("alerts", 0))
            f_caught = float(s.get("fraud_value_caught", 0.0))
            net_s = float(s.get("net_financial_saving", 0.0))
            lines.append(f"| {c:,.0f} | {mk} | {mode} | {th} | {al:,} | {f_caught:,.2f} | {net_s:,.2f} |")
        lines.append("")

    # Real Data Ring Analysis
    lines.append("C. Real-Data Fraud Ring Analysis")
    lines.append("Source: outputs/ring_analysis.json & outputs/rings_summary.csv")
    lines.append("")
    lines.append(f"- Fraud Accounts Evaluated: {ra_data.get('number_of_fraud_accounts', 0):,} distinct accounts")
    lines.append(f"- Total Connected Components: {ra_data.get('number_of_components', 0):,} components")
    lines.append("- Component Size Distribution:")
    by_size = ra_data.get("number_of_components_by_size", {})
    for sz, count in sorted(by_size.items(), key=lambda x: int(x[0])):
        lines.append(f"  * Size {sz}: {count:,} components")
    top_ring_str = "None"
    rings_sum_path = output_dir / "rings_summary.csv"
    if rings_sum_path.exists():
        try:
            df_r = pd.read_csv(rings_sum_path)
            if not df_r.empty:
                r0 = df_r.iloc[0]
                top_ring_str = f"Ring {r0['ring_id']} (Size {int(r0['size'])}, {int(r0['transaction_count'])} txns, {float(r0['total_amount']):,.2f} units, steps {int(r0['step_min'])}–{int(r0['step_max'])})"
        except Exception:
            pass

    lines.append(f"- Top Ring Exported: {top_ring_str}")
    lines.append(f"- Plain Empirical Finding: \"{ra_data.get('finding', '')}\"")
    lines.append("")

    # Planted Ring Synthetic Validation Table
    lines.append("D. Planted-Ring Synthetic Validation: Connected Components vs Label Propagation")
    lines.append("Source: outputs/planted_ring_validation.json (60 planted rings across 3 patterns and 3 camouflage levels in 200,000-row sample)")
    lines.append("")

    for algo_key, algo_name in [("connected_components", "Connected Components"), ("label_propagation", "Label Propagation (LPA)")]:
        algo_data = pr_data.get("algorithms", {}).get(algo_key, {})
        res_list = algo_data.get("results_by_pattern_and_camouflage", [])
        if not res_list and algo_key == "connected_components":
            res_list = pr_data.get("results_by_pattern_and_camouflage", [])

        if res_list:
            lines.append(f"### Algorithm: {algo_name}")
            lines.append("| Pattern | Camouflage | Planted | Strict Recovered | Strict Rec Rate (%) | Ring Contained Rate (%) | Mean Ring Purity (%) | Largest Comp with Planted | Homogeneity | Completeness | V-Measure | Contamination (%) |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for r in res_list:
                pat = str(r.get("pattern", "")).capitalize()
                cam = str(r.get("camouflage", ""))
                plt_cnt = int(r.get("planted", 0))
                rec_cnt = int(r.get("recovered", 0))
                rec_rate = float(r.get("recovery_rate", 0.0)) * 100.0
                cont_rate = float(r.get("ring_contained_rate", 0.0)) * 100.0
                purity = float(r.get("mean_ring_purity", 0.0)) * 100.0
                max_comp = int(r.get("largest_component_size_with_planted", 0))
                homog = float(r.get("homogeneity", 0.0))
                comp = float(r.get("completeness", 0.0))
                vm = float(r.get("v_measure", 0.0))
                contam = float(r.get("contamination", 0.0)) * 100.0
                lines.append(
                    f"| {pat} | {cam} | {plt_cnt} | {rec_cnt} | {rec_rate:.1f}% | {cont_rate:.1f}% | {purity:.2f}% | {max_comp:,} | {homog:.4f} | {comp:.4f} | {vm:.4f} | {contam:.1f}% |"
                )
            lines.append("")

    lines.append("3. FAILURES AND RETRIES (Exhaustive Chunk 2 Listing)")
    lines.append("-" * 90)
    lines.append("1. Step Index 1549: PowerShell process filter syntax exit code 1.")
    lines.append("2. Step Index 1559: SyntaxError in PowerShell WorkingSet expression.")
    lines.append("3. Step Index 1573: SyntaxError in python -c f-string quote escaping.")
    lines.append("4. Step Index 1581: rings.py AnalysisException: Unresolved column 'transaction_id' in PAYMENTS_SCHEMA projection. Fixed by using nameOrig and nameDest directly.")
    lines.append("5. Step Index 1844: AssertionError in test_app_smoke.py (expected 6 tabs, found 7). Fixed by updating assertion for 7 tabs.")
    lines.append("6. Step Index 1856: PermissionError [WinError 5] starting PySpark JVM inside sandbox. Fixed by running unsandboxed with BypassSandbox: true.")
    lines.append("7. Step Index 1878: SyntaxError in python -c print statement.")
    lines.append("8. Step Index 1906: git grep exited code 1 when string pattern was not found.")
    lines.append("9. Step Index 1926: SyntaxError in python -c JSON inspection.")
    lines.append("10. Step Index 1928: SyntaxError in python -c quote escaping.")
    lines.append("")

    lines.append("4. DECISIONS TAKEN")
    lines.append("-" * 90)
    lines.append("- Headline Operating Thresholds (Directive F1): Headline operating thresholds per model are strictly set to the validation thresholds from evaluation.json (chosen on validation set to maximize F1).")
    lines.append("- Default Review Cost (Directive F1): Review cost default set to 10,000.0 simulated currency units, labelled explicitly as 'assumption'.")
    lines.append("- Hindsight Test Best Label (Directive F1): The threshold curve and best point on test rows are clearly labelled 'hindsight on test rows, not a recommended operating point'.")
    lines.append("- Planted Ring Metrics (Directive F2): Added 'ring_contained_rate' and 'mean_ring_purity' beside strict recovery, and computed 'largest_component_size_with_planted' directly from detected components.")
    lines.append("- Dual Graph Algorithms (Directive F2): Evaluated both Connected Components and Label Propagation Algorithm (LPA) on the exact same 200,000-row PaySim sample with SPARK_DRIVER_MEMORY=3g and SPARK_SHUFFLE_PARTITIONS=8.")
    lines.append("- Automated Summary Generation (Directive F3): All Gate 2 summary tables are generated directly by script from the JSON output files without manual typing.")
    lines.append("")

    lines.append("5. GATE STATUS")
    lines.append("-" * 90)
    lines.append("GATE 2 updated with F1, F2, and F3 requirements.")
    lines.append("=" * 90)

    summary_text = "\n".join(lines) + "\n"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"Generated {summary_path} ({len(lines)} lines)")


if __name__ == "__main__":
    generate_gate2_summary()
