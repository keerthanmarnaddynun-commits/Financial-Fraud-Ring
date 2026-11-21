# PaySim Financial Fraud-Ring Analytics (PySpark, ML, Streamlit)

End-to-end big-data fraud analytics pipeline and interactive Streamlit dashboard built on Apache Spark (`PySpark 4.2.0`) over the PaySim mobile-money transaction dataset (`6,362,620` kept transactions across `9,073,900` distinct accounts and `8,213` ground-truth fraud rows).

All metrics, tables, and environment details in this README are generated programmatically from `outputs/` and `docs/` via `python make_report.py` from `docs/templates/README.md.tmpl`.

---

## 1. Problem Statement

Rule-based fraud filters (`isFlaggedFraud` on `TRANSFER` transactions above 200,000 units) evaluate transactions in isolation and miss the majority of fraudulent value (`9` alerts catching `1.35%` of fraud value on the primary time-split test set). This project evaluates:

1. **Spark-native distributed graph feature engineering**: in/out degrees, iterative PageRank, undirected triangle counts, iterative connected components, iterative label propagation (LPA), and account-level flow features computed natively on Spark DataFrames (see [docs/engine_findings.txt](docs/engine_findings.txt)).
2. **Per-split model comparison (`M0`–`M3`)**: models are trained per split (`time`, `group_dest`, `random`) on each split's own training rows, score thresholds are chosen on validation rows only (maximising F1 at true prevalence), test rows are scored distributively via Spark `mapInPandas`, and graph features are label-free but transductive (computed over the full graph).
3. **Real-data vs synthetic ring structure**: empirical connected-component analysis on PaySim's ground-truth fraud subgraph paired with a controlled synthetic ring injection test (`outputs/planted_ring_validation.json`) across `0%`, `20%`, and `50%` camouflage for both connected components and LPA.
4. **Big-data systems benchmarks**: CSV vs Parquet storage benchmark, Spark data-size and core-scaling benchmarks, and single-node `pandas`/NetworkX versus Spark comparisons.

---

## 2. Pipeline Architecture

Data flows through the following stages (a dedicated Mermaid architecture diagram in `docs/ARCHITECTURE.md` is deferred to a later documentation step):

1. **Storage & Ingestion** (`data_ingestion.py`, `to_parquet.py`): Reads PaySim CSV with explicit schema (`PAYMENTS_SCHEMA`) or Parquet (`--input-format parquet`), retains merchant destinations (`is_merchant` flag), validates schema and numeric columns, and writes `outputs/data_quality.json`.
2. **Graph Construction & Feature Engineering** (`graph_construction.py`, `feature_engineering.py`, `graph_pipeline.py`): Runs Spark-native DataFrame algorithms (`degrees`, `pagerank`, `triangles`, `components`, `lpa`, `behaviour_features`), joins account features onto transactions (`add_transaction_graph_features`), and persists `enriched_features.parquet`.
3. **Per-Split Model Training & Distributed Evaluation** (`splits.py`, `models.py`, `evaluation.py`, `configs/model_config.json`): Independently trains `m1_tabular`, `m1_tabular_noerr`, `m2_graph`, and `m3_community_mule` per split (`time`, `group_dest`, `random`), selects thresholds on validation rows, scores all test rows distributively, and writes `outputs/evaluation.json`, `outputs/pr_curves.csv`, and `outputs/scalability.json`.
4. **Downstream Analytics & Visuals** (`business_impact.py`, `rings.py`, `planted_rings.py`, `generate_visuals.py`): Computes financial impact at validation thresholds plus review-cost sensitivity, inspects ground-truth fraud components, runs planted-ring injection tests, and generates PNG figures in `outputs/visuals/`.
5. **Interactive Dashboard & Report Generation** (`app.py`, `make_report.py`, `check_docs.py`): Streamlit 7-tab dashboard reading strictly from `outputs/` (`FRAUD_OUTPUT_DIR`), plus deterministic Markdown table and facts generation ([docs/RESULTS_TABLES.md](docs/RESULTS_TABLES.md), [docs/FACTS_FOR_PAPER.md](docs/FACTS_FOR_PAPER.md), [docs/GLOSSARY.md](docs/GLOSSARY.md)).

---

## 3. Exact PowerShell Run Commands

### Environment & Dependencies
```powershell
pip install --no-input -r requirements.txt
pip install --no-input -r requirements-dev.txt
```

### Fast Sample Pipeline Run (2% Sample)
```powershell
python main.py --sample-fraction 0.02 --output-dir outputs_test
```

### Full End-to-End Pipeline Run (All Rows, Default `6g` / `48` Partitions)
```powershell
python main.py --output-dir runs\full_run
# Or reading from Parquet after conversion:
python main.py --input-format parquet --output-dir runs\full_run_parquet
```

### Downstream Analytics & Charts
```powershell
python business_impact.py
python rings.py
python planted_rings.py
python generate_visuals.py
```

### Big-Data Benchmarks
```powershell
# 1. Convert CSV to partitioned and unpartitioned Parquet (keeps original CSV)
python to_parquet.py

# 2. Storage benchmark (CSV vs Parquet, 3 repeats)
python storage_benchmark.py

# 3. Spark data-size and core scaling benchmark (resumable)
python benchmark_scaling.py --run-dir runs\c3-scaling
python aggregate_scaling.py --run-dir runs\c3-scaling --finalize

# 4. pandas/NetworkX vs Spark comparison (resumable)
python pandas_vs_spark.py --run-dir runs\c3-pandas-vs-spark
```

### Interactive Dashboard & Spark UI Screenshot Helper
```powershell
# Launch the 7-tab Streamlit dashboard (reads outputs\ by default)
streamlit run app.py

# Run 2% sample pipeline and hold Spark UI open at http://localhost:4040 for screenshots
python demo_run.py
```

### Documentation Generation, Consistency Check, and Tests
```powershell
# Regenerate README.md, docs\RESULTS_TABLES.md, docs\FACTS_FOR_PAPER.md, and results_snapshot\
python make_report.py

# Run read-only consistency and link check
python check_docs.py

# Run dashboard smoke tests
python -m pytest tests/test_app_smoke.py -v

# Run full pytest suite
python -m pytest -q
```

---

## 4. Pinned Environment and Software Versions

Measured on the project machine (`outputs/run_info.json`):
- **Python**: `3.13.7 (tags/v3.13.7:bcee1c3, Aug 14 2025, 14:15:11) [MSC v.1944 64 bit (AMD64)]`
- **Java**: `openjdk version "20.0.2" 2023-07-18`
- **CPU**: `16` physical / `22` logical cores
- **RAM**: `15.37 GiB` total
- **Default Spark Settings**: `spark.driver.memory=6g`, `spark.sql.shuffle.partitions=48`

Pinned Python runtime packages ([requirements.txt](requirements.txt)):
```text
# Pinned runtime dependencies (Section 2.2)
pyspark==4.2.0
pandas==3.0.0
numpy==2.2.4
scikit-learn==1.8.0
xgboost==3.4.1
pyarrow==26.0.0
networkx==3.6.1
plotly==6.5.2
streamlit==1.56.0
psutil==7.2.2
joblib==1.5.3
# graphframes and graphframes-py are excluded: STEP 1.1 empirical tests showed GraphFrames 0.8.3 fails on Spark 4.2 with NoSuchMethodError. Spark-native DataFrame graph algorithms are used instead.
```

---

## 5. Repository Folder Map

```text
Financial-Fraud-Ring/
  main.py, graph_pipeline.py, spark_session.py, data_ingestion.py,
  graph_construction.py, feature_engineering.py, feature_sets.py,
  splits.py, models.py, evaluation.py, stage_timer.py, run_info.py,
  provenance.py, business_impact.py, rings.py, planted_rings.py,
  to_parquet.py, storage_benchmark.py, benchmark_scaling.py,
  aggregate_scaling.py, pandas_vs_spark.py, generate_visuals.py,
  make_report.py, check_docs.py, demo_run.py, app.py,
  model_names.json, requirements.txt, requirements-dev.txt,
  README.md, PROGRESS.txt, NOTES_FOR_LATER.txt
  configs/             model_config.json
  tests/               pytest suite and tests/data/sample_1000.csv
  docs/                GLOSSARY.md, env_snapshot.txt, raw_data_facts.json,
                       engine_findings.txt, RESULTS_TABLES.md,
                       FACTS_FOR_PAPER.md, templates/
  logs/                AG_WORKLOG_C1..C4.txt, GATE_*_SUMMARY.txt, raw/ (gitignored)
  runs/                immutable per-run folders with CARD.md + SHA256SUMS.txt (gitignored)
  outputs/             accepted copy of results + PROVENANCE.txt (gitignored)
  results_snapshot/    committed small JSON/CSV/PNG results + SHA256SUMS.txt
  archive/             archived legacy outputs and archived legacy web dashboard (gitignored)
```

---

## 6. Key Results (Generated from `outputs/`)

For full tables across all three splits (`time`, `group_dest`, `random`), review-cost sensitivity, per-run benchmark logs, and metric definitions, see [docs/RESULTS_TABLES.md](docs/RESULTS_TABLES.md), [docs/FACTS_FOR_PAPER.md](docs/FACTS_FOR_PAPER.md), and [docs/GLOSSARY.md](docs/GLOSSARY.md).

### 6.0 What the results do and do not show

- **Tree models with balance-error features (`m1_tabular`, `m2_graph`, `m3_community_mule`)**: On the primary `time` split (Table 6.1, `outputs/evaluation.json: splits.time`), the three models differ by only a few test rows (`m1_tabular`: TP `2,058`, FP `6`, FN `4`; `m2_graph`: TP `2,057`, FP `6`, FN `5`; `m3_community_mule`: TP `2,053`, FP `7`, FN `9`). Each model was trained once (`random_state=42`, `configs/model_config.json`), so no performance difference among these three models is claimed.
- **Role of balance-error features (`m1_tabular_noerr`)**: Removing `errorBalanceOrig` and `errorBalanceDest` drops `time`-split F1 from `0.9976` (`m1_tabular`) to `0.9163` (`m1_tabular_noerr`), with false positives rising from `6` to `133` and false negatives rising from `4` to `206` (Table 6.1), showing that the balance-error features carry most of the predictive signal in PaySim.
- **Real PaySim fraud subgraph structure**: The ground-truth fraud subgraph contains `0` components of size 4 or more (`8,125` isolated pairs of size 2 and `44` components of size 3; Table 6.3, `outputs/ring_analysis.json`).
- **Graph-feature runtime bottleneck**: In the full pipeline run (`outputs/scalability.json`), iterative connected components take `2042.20 s` of the `2908.38 s` `graph_feature_engineering` phase (`70.2%`).
- **Planted-ring sample size per cell**: The planted-ring tables (Table 6.4, `outputs/planted_ring_validation.json`) contain `6 or 7` planted rings per `(pattern, camouflage)` cell (`60` rings total injected into one `200,000`-row sample).
- **Strict ring recovery definition**: Strict recovery (`docs/GLOSSARY.md`) requires at least `80%` of a planted ring's accounts in one detected community AND at least `80%` of that community's accounts belonging to that ring; rings whose accounts stay together in a larger noisy component show high `ring_contained_rate` with lower `mean_ring_purity` and lower strict recovery.

### 6.1 Primary Time-Split Model Results (`outputs/evaluation.json: splits.time`)
Test set size: `193,939` rows, fraud prevalence `1.0632%`. Thresholds for `m1_tabular`–`m3_community_mule` are selected strictly on validation rows; `m0_rule` uses its fixed `rule_definition`.

| Model | Threshold | Threshold Source | Precision | Recall | F1 | PR-AUC (AP) | ROC-AUC | False Alarms / 10k | TP | FP | FN | TN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m0_rule | 0.500000 | rule_definition | 1.0000 | 0.0044 | 0.0087 | 0.0150 | 0.502182 | 0.00 | 9 | 0 | 2,053 | 191,877 |
| m1_tabular | 0.998236 | validation | 0.9971 | 0.9981 | 0.9976 | 0.9997 | 0.999997 | 0.31 | 2,058 | 6 | 4 | 191,871 |
| m1_tabular_noerr | 0.984859 | validation | 0.9331 | 0.9001 | 0.9163 | 0.9835 | 0.999812 | 6.86 | 1,856 | 133 | 206 | 191,744 |
| m2_graph | 0.996715 | validation | 0.9971 | 0.9976 | 0.9973 | 0.9995 | 0.999995 | 0.31 | 2,057 | 6 | 5 | 191,871 |
| m3_community_mule | 0.996618 | validation | 0.9966 | 0.9956 | 0.9961 | 0.9995 | 0.999994 | 0.36 | 2,053 | 7 | 9 | 191,870 |

### 6.2 Business Impact at Validation Operating Thresholds (`outputs/business_impact.json`)
Time-split test set total fraud value: `3,181,396,658.88` `PaySim simulated currency units` across `2,062` fraud transactions. Default review cost: `10,000` (`assumption`).

| Model | Operating Threshold | Threshold Source | Alerts | True Alerts | False Alerts | Alerts / 10k | Fraud Value Caught | Fraud Value Missed | Share Caught | Net Saving (Review Cost 10,000 assumption) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rule_baseline | 0.500000 | rule_definition | 9 | 9 | 0 | 0.46 | 42,898,513.56 | 3,138,498,145.32 | 1.35% | 42,808,513.56 |
| m1_tabular | 0.998236 | validation | 2,064 | 2,058 | 6 | 106.43 | 3,180,643,739.58 | 752,919.30 | 99.98% | 3,160,003,739.58 |
| m1_tabular_noerr | 0.984859 | validation | 1,989 | 1,856 | 133 | 102.56 | 3,160,064,870.70 | 21,331,788.18 | 99.33% | 3,140,174,870.70 |
| m2_graph | 0.996715 | validation | 2,063 | 2,057 | 6 | 106.37 | 3,180,642,653.42 | 754,005.46 | 99.98% | 3,160,012,653.42 |
| m3_community_mule | 0.996618 | validation | 2,060 | 2,053 | 7 | 106.22 | 3,180,642,415.69 | 754,243.19 | 99.98% | 3,160,042,415.69 |

### 6.3 Real-Data Fraud Subgraph Finding (`outputs/ring_analysis.json`)
> In the PaySim dataset, real fraud transactions are overwhelmingly isolated 1-to-1 interactions (8125 components of size 2). Exactly 44 components have size >= 3 (all of size 3, where 2 distinct origin accounts transferred funds to the same destination account). There are zero multi-hop chains, cycles, or larger fraud rings of size >= 4 in the ground-truth fraud data.

| Metric | Value | Source File | Key |
| --- | --- | --- | --- |
| Fraud accounts in fraud subgraph | 16,382 | outputs/ring_analysis.json | number_of_fraud_accounts |
| Total fraud-subgraph components | 8,169 | outputs/ring_analysis.json | number_of_components |
| Components of size 2 (1-to-1 pairs) | 8,125 | outputs/ring_analysis.json | number_of_components_by_size.2 |
| Components of size 3 (2-to-1 funnels) | 44 | outputs/ring_analysis.json | number_of_components_by_size.3 |
| Components of size >= 3 | 44 | outputs/ring_analysis.json | count_size_ge_3 |
| Components of size >= 4 | 0 | outputs/ring_analysis.json | number_of_components_by_size (>=4) |
| Exported ring summary rows | 20 | outputs/ring_analysis.json | exported_rings_count |

### 6.4 Planted-Ring Synthetic Injection Test (`outputs/planted_ring_validation.json`)
Label: `synthetic injection for validation`; `60` rings injected into a `200,000`-row sample.

**Connected Components** (overall strict recovery: `35.00%`):

| Pattern | Camouflage | Planted | Recovered | Strict Recovery | Ring Contained | Mean Purity | Largest Component w/ Planted | Homogeneity | Completeness | V-Measure | Contamination |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chain | 0% | 7 | 7 | 100.0% | 100.0% | 1.0000 | 6 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| chain | 20% | 7 | 0 | 0.0% | 100.0% | 0.3373 | 22 | 1.0000 | 1.0000 | 1.0000 | 0.6627 |
| chain | 50% | 6 | 0 | 0.0% | 100.0% | 0.1736 | 43 | 1.0000 | 1.0000 | 1.0000 | 0.8264 |
| cycle | 0% | 7 | 7 | 100.0% | 100.0% | 1.0000 | 5 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| cycle | 20% | 7 | 0 | 0.0% | 100.0% | 0.3314 | 33 | 1.0000 | 1.0000 | 1.0000 | 0.6686 |
| cycle | 50% | 6 | 0 | 0.0% | 100.0% | 0.2120 | 31 | 1.0000 | 1.0000 | 1.0000 | 0.7880 |
| funnel | 0% | 7 | 7 | 100.0% | 100.0% | 1.0000 | 17 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| funnel | 20% | 7 | 0 | 0.0% | 100.0% | 0.3384 | 55 | 1.0000 | 1.0000 | 1.0000 | 0.6616 |
| funnel | 50% | 6 | 0 | 0.0% | 100.0% | 0.2026 | 104 | 1.0000 | 1.0000 | 1.0000 | 0.7974 |

**Label Propagation (LPA)** (overall strict recovery: `21.67%`):

| Pattern | Camouflage | Planted | Recovered | Strict Recovery | Ring Contained | Mean Purity | Largest Component w/ Planted | Homogeneity | Completeness | V-Measure | Contamination |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chain | 0% | 7 | 0 | 0.0% | 54.3% | 1.0000 | 3 | 1.0000 | 0.7386 | 0.8497 | 0.0000 |
| chain | 20% | 7 | 0 | 0.0% | 49.0% | 0.6179 | 10 | 1.0000 | 0.6759 | 0.8066 | 0.3821 |
| chain | 50% | 6 | 0 | 0.0% | 40.6% | 0.4319 | 9 | 1.0000 | 0.5928 | 0.7443 | 0.5681 |
| cycle | 0% | 7 | 4 | 57.1% | 78.6% | 1.0000 | 5 | 1.0000 | 0.8818 | 0.9372 | 0.0000 |
| cycle | 20% | 7 | 1 | 14.3% | 78.1% | 0.5310 | 12 | 1.0000 | 0.8287 | 0.9063 | 0.4690 |
| cycle | 50% | 6 | 0 | 0.0% | 57.8% | 0.4296 | 9 | 1.0000 | 0.7129 | 0.8324 | 0.5704 |
| funnel | 0% | 7 | 7 | 100.0% | 92.1% | 1.0000 | 16 | 1.0000 | 0.8789 | 0.9355 | 0.0000 |
| funnel | 20% | 7 | 1 | 14.3% | 72.4% | 0.9768 | 14 | 1.0000 | 0.6745 | 0.8056 | 0.0232 |
| funnel | 50% | 6 | 0 | 0.0% | 50.8% | 0.9056 | 15 | 1.0000 | 0.5288 | 0.6918 | 0.0944 |

### 6.5 Spark Data-Size Scaling (`outputs/scaling_runs.json: data_size_summary`)
Default Spark settings (`6g` driver memory, `48` shuffle partitions, `local[*]`, 2 repeats). `3` initial runs hit the `< 0.5 GiB / 60 s` memory guard, were retained as `superseded: true`, and were re-run under `< 0.2 GiB / 120 s`.

| Fraction | Active OK/Total | Active Statuses | Superseded Prior | Rows | Nodes | Edges | Mean Total (s) | Individual Repeats (s) | Mean Peak RSS | Min Avail RAM | Mean Shuffle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.10 | 2/2 | ok, ok | 0 | 635,856 | 1,092,959 | 635,856 | 493.36 | 490.71, 496.00 | 6.18 GiB | 1.15 GiB | 8.26 GiB |
| 0.25 | 2/2 | ok, ok | 2 | 1,590,820 | 2,509,524 | 1,590,820 | 923.50 | 1132.07, 714.93 | 6.76 GiB | 76.73 MiB | 20.19 GiB |
| 0.50 | 2/2 | ok, ok | 1 | 3,180,590 | 4,737,083 | 3,180,590 | 1364.98 | 1262.74, 1467.22 | 6.91 GiB | 461.73 MiB | 50.50 GiB |
| 1.00 | 2/2 | ok, ok | 0 | 6,362,620 | 9,073,900 | 6,362,620 | 2781.71 | 2799.15, 2764.28 | 6.98 GiB | 444.07 MiB | 126.98 GiB |

### 6.6 Spark Core Scaling (`outputs/scaling_runs.json: core_scaling_summary`)
Measured at sample fraction `0.10` across `local[1..16]` (2 repeats each).

| Master | Fraction | Statuses | Mean Total (s) | Individual Repeats (s) | Speedup | Individual Speedups | Efficiency | Mean Peak RSS | Min Avail RAM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| local[1] | 0.10 | ok, ok | 926.94 | 957.10, 896.78 | 1.000x | 0.969x, 1.034x | 1.000 | 3.90 GiB | 3.05 GiB |
| local[2] | 0.10 | ok, ok | 787.97 | 713.36, 862.58 | 1.176x | 1.299x, 1.075x | 0.588 | 4.90 GiB | 1.72 GiB |
| local[4] | 0.10 | ok, ok | 742.21 | 764.09, 720.33 | 1.249x | 1.213x, 1.287x | 0.312 | 4.56 GiB | 2.34 GiB |
| local[8] | 0.10 | ok, ok | 690.28 | 689.16, 691.40 | 1.343x | 1.345x, 1.341x | 0.168 | 4.72 GiB | 2.11 GiB |
| local[16] | 0.10 | ok, ok | 667.05 | 661.96, 672.15 | 1.390x | 1.400x, 1.379x | 0.087 | 5.50 GiB | 1.33 GiB |

### 6.7 pandas/NetworkX vs Spark (`outputs/pandas_vs_spark.json: summary`)
Uniform Parquet samples (`100,000`–`4,000,000` rows, `seed=42`) plus full PaySim Parquet (`6,362,620` rows) for `pandas_networkx`; 10-minute timeout, 9 GiB RSS cap, 2 repeats.

| Task | Tool | Rows | Statuses | Load Mean (s) | Compute Mean (s) | Compute Repeats (s) | Peak RSS | Min Avail RAM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| components | pandas_networkx | 100,000 | ok, ok | 0.032 | 0.500 | 0.499, 0.502 | 290.60 MiB | 6.85 GiB |
| components | spark | 100,000 | ok, ok | 5.280 | 88.872 | 88.192, 89.551 | 3.99 GiB | 3.33 GiB |
| components | pandas_networkx | 250,000 | ok, ok | 0.033 | 1.808 | 1.711, 1.906 | 458.71 MiB | 6.71 GiB |
| components | spark | 250,000 | ok, ok | 5.936 | 157.088 | 157.638, 156.538 | 3.45 GiB | 3.70 GiB |
| components | pandas_networkx | 500,000 | ok, ok | 0.050 | 4.178 | 4.114, 4.242 | 698.56 MiB | 6.46 GiB |
| components | spark | 500,000 | ok, ok | 6.583 | 233.167 | 233.794, 232.540 | 4.02 GiB | 2.99 GiB |
| components | pandas_networkx | 1,000,000 | ok, ok | 0.065 | 8.300 | 8.262, 8.337 | 1.14 GiB | 6.08 GiB |
| components | spark | 1,000,000 | ok, ok | 9.264 | 294.049 | 290.234, 297.864 | 4.37 GiB | 2.70 GiB |
| components | pandas_networkx | 2,000,000 | ok, ok | 0.103 | 16.367 | 16.191, 16.542 | 2.05 GiB | 5.15 GiB |
| components | spark | 2,000,000 | timeout | not available | not available | none | 6.20 GiB | 1.01 GiB |
| components | pandas_networkx | 4,000,000 | ok, ok | 0.177 | 36.047 | 35.626, 36.467 | 3.80 GiB | 3.75 GiB |
| components | pandas_networkx | 6,362,620 | ok, ok | 0.179 | 49.936 | 50.735, 49.137 | 5.41 GiB | 2.91 GiB |
| degree | pandas_networkx | 100,000 | ok, ok | 0.034 | 0.293 | 0.284, 0.302 | 235.93 MiB | 6.59 GiB |
| degree | spark | 100,000 | ok, ok | 4.702 | 8.118 | 8.632, 7.605 | 1.30 GiB | 6.23 GiB |
| degree | pandas_networkx | 250,000 | ok, ok | 0.041 | 0.813 | 0.753, 0.873 | 308.55 MiB | 6.53 GiB |
| degree | spark | 250,000 | ok, ok | 5.486 | 9.537 | 9.176, 9.897 | 1.41 GiB | 6.10 GiB |
| degree | pandas_networkx | 500,000 | ok, ok | 0.059 | 2.068 | 2.060, 2.077 | 429.42 MiB | 6.45 GiB |
| degree | spark | 500,000 | ok, ok | 5.972 | 13.093 | 13.259, 12.927 | 2.04 GiB | 5.26 GiB |
| degree | pandas_networkx | 1,000,000 | ok, ok | 0.075 | 4.701 | 4.769, 4.634 | 648.66 MiB | 6.26 GiB |
| degree | spark | 1,000,000 | ok, ok | 7.481 | 13.954 | 13.627, 14.281 | 2.21 GiB | 5.18 GiB |
| degree | pandas_networkx | 2,000,000 | ok, ok | 0.107 | 10.554 | 10.259, 10.849 | 1.03 GiB | 5.92 GiB |
| degree | spark | 2,000,000 | ok, ok | 9.033 | 17.963 | 17.907, 18.020 | 3.09 GiB | 4.15 GiB |
| degree | pandas_networkx | 4,000,000 | ok, ok | 0.169 | 23.329 | 23.735, 22.923 | 1.80 GiB | 5.25 GiB |
| degree | spark | 4,000,000 | ok, ok | 11.946 | 23.252 | 22.480, 24.024 | 4.12 GiB | 3.06 GiB |
| degree | pandas_networkx | 6,362,620 | ok, ok | 0.585 | 35.331 | 35.877, 34.786 | 2.56 GiB | 5.59 GiB |

**Reading guide (`outputs/pandas_vs_spark.json: summary`):**
- **Connected components**: `pandas_networkx` finishes faster than Spark at every size where both ran (`100,000` to `1,000,000` rows; at `1,000,000` rows, compute mean is `8.30 s` vs `294.05 s`), and Spark recorded `timeout` at `2,000,000` rows (`600 s` limit). `pandas_networkx` completed `6,362,620` rows (`49.94 s` compute mean, `5.41 GiB` peak RSS).
- **Degree counts**: At `4,000,000` rows, compute means are `23.25 s` (Spark) and `23.33 s` (`pandas_networkx`), with Spark load time listed separately (`11.95 s` vs `0.17 s`). Spark was not run at `6,362,620` rows in this comparison (`pandas_networkx` compute mean `35.33 s`). Consequently, `pandas_networkx` is not faster than Spark for degree compute at every size.

### 6.8 Storage Benchmark: CSV vs Parquet (`outputs/storage_benchmark.json`)
3 repeats each on the full `6,362,620`-row dataset.

| Format | Size (MiB) | Compression vs CSV | Operation | All-3 Mean (s) | Repeats 1-3 (s) | Warm Mean (Repeats 2-3) (s) | All-3 Speedup vs CSV | Warm Speedup vs CSV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| csv | 470.67 | 1.000x | count_rows | 4.9775 | 12.20, 1.49, 1.24 | 1.3647 | 1.00x | 1.00x |
| csv | 470.67 | 1.000x | read_two_columns | 4.4425 | 7.37, 3.06, 2.90 | 2.9797 | 1.00x | 1.00x |
| csv | 470.67 | 1.000x | filter_transfer_fraud | 3.0483 | 3.94, 2.64, 2.57 | 2.6042 | 1.00x | 1.00x |
| parquet_partitioned | 249.64 | 1.885x | count_rows | 1.8054 | 2.73, 1.54, 1.15 | 1.3431 | 2.76x | 1.02x |
| parquet_partitioned | 249.64 | 1.885x | read_two_columns | 1.8561 | 2.77, 1.63, 1.17 | 1.4015 | 2.39x | 2.13x |
| parquet_partitioned | 249.64 | 1.885x | filter_transfer_fraud | 1.2015 | 1.68, 1.14, 0.78 | 0.9620 | 2.54x | 2.71x |
| parquet_unpartitioned | 273.11 | 1.723x | count_rows | 0.7076 | 0.69, 0.75, 0.68 | 0.7140 | 7.03x | 1.91x |
| parquet_unpartitioned | 273.11 | 1.723x | read_two_columns | 0.7764 | 0.74, 0.86, 0.73 | 0.7932 | 5.72x | 3.76x |
| parquet_unpartitioned | 273.11 | 1.723x | filter_transfer_fraud | 1.0209 | 1.44, 0.83, 0.79 | 0.8109 | 2.99x | 3.21x |

*Cold-cache note (`outputs/storage_benchmark.json`): the first CSV repeat is a cold-cache read (`count_rows` repeat 1 = `12.20 s` vs repeats 2-3 = `1.49 s`, `1.24 s`; `read_two_columns` repeat 1 = `7.37 s` vs `3.06 s`, `2.90 s`), which inflates the all-3 CSV means and all-3 speedups; warm means (repeats 2-3) and warm speedups are therefore shown beside them.*

---

## 7. Limitations

1. **Single machine**: All Spark runs and benchmarks execute in `local[*]` or `local[n]` mode on one laptop (`15.37 GiB` total RAM, `16` physical / `22` logical cores, local disk); measurements do not represent multi-node network shuffle, HDFS/S3 I/O, or YARN/Kubernetes cluster scheduling.
2. **Shared RAM, memory guard, and repeat spread**: The host OS shares `15.37 GiB` RAM with the Spark JVM (`6g` driver heap plus shuffle/off-heap buffers). `3` initial scaling runs hit the `< 0.5 GiB / 60 s` memory guard and were retained as `superseded: true`; the relaxed memory guard aborts only when available RAM stays below `0.2 GiB` for `120 s` continuously, so shorter dips still occur in `ok` runs (the minimum observed in an `ok` run was `76.73 MiB` at `fraction=0.25`, `outputs/scaling_runs.json: data_size_summary.1.min_available_ram_bytes`), contributing to spread between repeats.
3. **Hybrid CPU cores**: The machine's `16` physical / `22` logical cores (`outputs/run_info.json: cores`) mix performance and efficiency cores, producing uneven speedup across `local[1..16]`.
4. **Single training run per model and split**: Each model is trained once per split with fixed `random_state=42` (`configs/model_config.json: xgboost.random_state`); no multi-seed confidence intervals are computed.
5. **Training sampling**: Per `splits.py` (`draw_training_sample`), each split's training set keeps training-split fraud rows plus sampled non-fraud rows at a 10:1 negative-to-positive ratio capped at `max_training_rows=100,000` per split (`outputs/run_info.json: config.max_training_rows`), totaling `sampled_training_rows=147,532` rows across the three splits (`outputs/scalability.json: sampled_training_rows`), while validation threshold selection and test evaluation run on all validation and test rows at true prevalence.
6. **Simulated data and balance-accounting signal**: PaySim is a synthetic mobile-money simulator whose ground-truth fraud consists almost entirely of isolated 1-to-1 `TRANSFER` and `CASH_OUT` pairs (zero real components of size >= 4) and carries strong balance-accounting signals (`errorBalanceOrig`, `errorBalanceDest`). `m1_tabular_noerr` is reported alongside `m1_tabular`, `m2_graph`, and `m3_community_mule`.
7. **Transductive, label-free graph features**: Account-level graph features (degrees, PageRank, triangle count, connected components, LPA, flow ratios) are label-free but computed across the full transaction graph rather than strictly past-only per split.
8. **Uniform transaction sampling in scaling benchmarks**: Sampling transactions uniformly (`seed=42`) changes graph edge density and component connectivity across fractions; node and edge counts are reported per run alongside runtime.
9. **Two repeats in long benchmarks**: Scaling and pandas-vs-Spark benchmarks use 2 repeats per configuration (3 repeats for storage); individual repeat values are reported alongside means.
10. **Planted-ring evaluation uses one `200,000`-row sample**: Synthetic ring recovery results (`outputs/planted_ring_validation.json`) come from injecting `60` rings into a single `200,000`-row PaySim sample rather than multiple independent samples or the full `6,362,620`-row graph.
11. **pandas/NetworkX comparison scope**: The single-node comparison covers only `degree` counts and `connected_components`; it does not benchmark PageRank, triangle counting, LPA, feature joins, or end-to-end model scoring in NetworkX.
