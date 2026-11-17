import hashlib
import json
import os
import stat
from pathlib import Path

RUN_DIR = Path('runs/20261117_162714_c1-full-v2')

# 1. Read metrics from JSON files directly
with open(RUN_DIR / 'data_quality.json') as f:
    dq = json.load(f)

with open(RUN_DIR / 'evaluation.json') as f:
    ev = json.load(f)

with open(RUN_DIR / 'scalability.json') as f:
    sc = json.load(f)

# Build CARD.md
splits_ev = ev.get('splits', {})
phases = sc.get('phases') or sc.get('timings_seconds', {})
pipeline_total = phases.get('pipeline_total', 0.0)

card_lines = [
    "# Results Card: 20261117_162714_c1-full-v2",
    "",
    "## Purpose",
    "Full pipeline execution across 6.36M PaySim transactions evaluating comparative models across time, destination group, and random splits with per-split model training.",
    "",
    "## Execution Command",
    "```bash",
    "python -u main.py --output-dir runs/20261117_162714_c1-full-v2",
    "```",
    "",
    "## Inputs",
    "- **raw_dataset**: `data/PS_20174392719_1491204439457_log.csv`",
    "",
    "## Outputs",
    "- **evaluation**: `evaluation.json`",
    "- **scalability**: `scalability.json`",
    "- **data_quality**: `data_quality.json`",
    "- **pr_curves**: `pr_curves.csv`",
    "- **models**: `models/`",
    "- **enriched_features**: `enriched_features.parquet`",
    "",
    "## Key Measured Numbers (Read from Output Artifacts)",
    "| Metric / Label | Value | Source File | JSON Path |",
    "| --- | --- | --- | --- |",
    f"| Transactions Read | `{dq.get('rows_read')}` | `data_quality.json` | `rows_read` |",
    f"| Transactions Kept | `{dq.get('rows_kept')}` | `data_quality.json` | `rows_kept` |",
    f"| Distinct Accounts | `{dq.get('distinct_accounts')}` | `data_quality.json` | `distinct_accounts` |",
    f"| Fraud Transactions | `{dq.get('fraud_rows')}` | `data_quality.json` | `fraud_rows` |",
]

for split_name in ['time', 'group_dest', 'random']:
    split_models = splits_ev.get(split_name, {})
    for m in ['m1_tabular', 'm1_tabular_noerr', 'm2_graph', 'm3_community_mule']:
        m_data = split_models.get(m, {})
        f1 = m_data.get('at_threshold', {}).get('f1')
        pr_auc = m_data.get('average_precision')
        roc_auc = m_data.get('roc_auc')
        card_lines.append(f"| {split_name} {m} F1 | `{f1}` | `evaluation.json` | `splits.{split_name}.{m}.at_threshold.f1` |")
        card_lines.append(f"| {split_name} {m} PR-AUC | `{pr_auc}` | `evaluation.json` | `splits.{split_name}.{m}.average_precision` |")

card_lines.extend([
    f"| Pipeline Total Seconds | `{pipeline_total:.2f}` | `scalability.json` | `phases.pipeline_total` |",
    f"| Graph Engineering Seconds | `{phases.get('graph_feature_engineering')}` | `scalability.json` | `phases.graph_feature_engineering` |",
    f"| Total Shuffle Read Bytes | `{sc.get('shuffle_read_bytes')}` | `scalability.json` | `shuffle_read_bytes` |",
    f"| Total Shuffle Write Bytes | `{sc.get('shuffle_write_bytes')}` | `scalability.json` | `shuffle_write_bytes` |",
    f"| Memory Warning | `{sc.get('memory_warning')}` | `scalability.json` | `memory_warning` |",
    "",
    "## Caveats and Limitations",
    "- Single-node local execution in PySpark 4.2.0 local[*] mode.",
    "- LPA reached 5 iterations without convergence flag set to True (expected bounded iteration behavior).",
    "- Simulator balance discrepancy artifacts inform M1 Tabular baseline behavior.",
    "",
    "## What This Result Does NOT Show",
    "- Does not show multi-machine production cluster behavior.",
    "- Does not show ground-truth real-world criminal syndicate identities (synthetic data labels only).",
    ""
])

card_text = "\n".join(card_lines)
with open(RUN_DIR / 'CARD.md', 'w') as f:
    f.write(card_text)

print("CARD.md written.")

# 2. SHA256SUMS.txt
sha_lines = []
for root, dirs, files in os.walk(RUN_DIR):
    for f in sorted(files):
        if f == 'SHA256SUMS.txt':
            continue
        filepath = Path(root) / f
        h = hashlib.sha256()
        with open(filepath, 'rb') as fp:
            while chunk := fp.read(65536):
                h.update(chunk)
        rel_path = filepath.relative_to(RUN_DIR).as_posix()
        sha_lines.append(f"{h.hexdigest()}  {rel_path}")

sha_lines.sort()
with open(RUN_DIR / 'SHA256SUMS.txt', 'w') as f:
    f.write("\n".join(sha_lines) + "\n")

print(f"SHA256SUMS.txt written with {len(sha_lines)} file entries.")

# 3. Make run folder read-only
for root, dirs, files in os.walk(RUN_DIR):
    for f in files:
        p = Path(root) / f
        p.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)

print("Run folder set to read-only.")
