# Results Tables (Generated from `outputs/`)

All tables in this document are generated programmatically by `make_report.py` from `outputs/` and `docs/raw_data_facts.json`. No numbers are hand-typed.

## 1. Dataset Counts and Data Quality
Source: `outputs/data_quality.json`, `docs/raw_data_facts.json`.

| Metric | Value | Source File | Key |
| --- | --- | --- | --- |
| Dataset file | PS_20174392719_1491204439457_log.csv | docs/raw_data_facts.json | file |
| Rows read | 6,362,620 | outputs/data_quality.json | rows_read |
| Rows kept after validation | 6,362,620 | outputs/data_quality.json | rows_kept |
| Dropped: missing/null | 0 | outputs/data_quality.json | dropped_by_rule.missing_required_columns_or_nulls |
| Dropped: non-numeric | 0 | outputs/data_quality.json | dropped_by_rule.non_numeric_values |
| Dropped: invalid type/account | 0 | outputs/data_quality.json | dropped_by_rule.invalid_type_or_account_filter |
| Total fraud rows (isFraud=1) | 8,213 | outputs/data_quality.json | fraud_rows |
| Whole-file fraud prevalence | 0.1291% | outputs/data_quality.json | fraud_rows / rows_kept |
| Fraud rows: CASH_OUT | 4,116 | docs/raw_data_facts.json | fraud_rows_by_type.CASH_OUT |
| Fraud rows: TRANSFER | 4,097 | docs/raw_data_facts.json | fraud_rows_by_type.TRANSFER |
| Time steps (min .. max) | 1 .. 743 | docs/raw_data_facts.json | step_min, step_max |
| Distinct origin accounts (nameOrig) | 6,353,307 | docs/raw_data_facts.json | distinct_nameOrig_count |
| Distinct destination accounts (nameDest) | 2,722,362 | docs/raw_data_facts.json | distinct_nameDest_count |
| Distinct accounts (union) | 9,073,900 | outputs/data_quality.json | distinct_accounts |
| Destination prefix C (customer) | 4,211,125 | docs/raw_data_facts.json | rows_by_dest_prefix.C |
| Destination prefix M (merchant) | 2,151,495 | docs/raw_data_facts.json | rows_by_dest_prefix.M |
| Rows by type: CASH_IN | 1,399,284 | outputs/data_quality.json | by_type.CASH_IN |
| Rows by type: CASH_OUT | 2,237,500 | outputs/data_quality.json | by_type.CASH_OUT |
| Rows by type: DEBIT | 41,432 | outputs/data_quality.json | by_type.DEBIT |
| Rows by type: PAYMENT | 2,151,495 | outputs/data_quality.json | by_type.PAYMENT |
| Rows by type: TRANSFER | 532,909 | outputs/data_quality.json | by_type.TRANSFER |

## 2. Evaluation Splits and Prevalence
Source: `outputs/evaluation.json`. Models are trained independently on each split's training partition; score thresholds are selected on each split's validation partition (maximising F1 at true prevalence) and never tuned on test rows.

| Split | Test Rows | Test Fraud Rows (tp+fn) | Test Prevalence | Threshold Source | Source Key (`outputs/evaluation.json`) |
| --- | --- | --- | --- | --- | --- |
| time | 193,939 | 2,062 | 1.0632% | validation | splits.time.m1_tabular.(rows, prevalence, at_threshold.confusion) |
| group_dest | 1,600,192 | 2,078 | 0.1299% | validation | splits.group_dest.m1_tabular.(rows, prevalence, at_threshold.confusion) |
| random | 1,589,069 | 2,101 | 0.1322% | validation | splits.random.m1_tabular.(rows, prevalence, at_threshold.confusion) |

## 3. Model Comparison — Primary Time Split
Source: `outputs/evaluation.json` (`splits.time.*`).

| Model | Threshold | Threshold Source | Precision | Recall | F1 | PR-AUC (AP) | ROC-AUC | False Alarms / 10k | TP | FP | FN | TN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m0_rule | 0.500000 | rule_definition | 1.0000 | 0.0044 | 0.0087 | 0.0150 | 0.502182 | 0.00 | 9 | 0 | 2,053 | 191,877 |
| m1_tabular | 0.998236 | validation | 0.9971 | 0.9981 | 0.9976 | 0.9997 | 0.999997 | 0.31 | 2,058 | 6 | 4 | 191,871 |
| m1_tabular_noerr | 0.984859 | validation | 0.9331 | 0.9001 | 0.9163 | 0.9835 | 0.999812 | 6.86 | 1,856 | 133 | 206 | 191,744 |
| m2_graph | 0.996715 | validation | 0.9971 | 0.9976 | 0.9973 | 0.9995 | 0.999995 | 0.31 | 2,057 | 6 | 5 | 191,871 |
| m3_community_mule | 0.996618 | validation | 0.9966 | 0.9956 | 0.9961 | 0.9995 | 0.999994 | 0.36 | 2,053 | 7 | 9 | 191,870 |

## 4. Model Comparison — Secondary Group Split (Destination Account `nameDest`)
Source: `outputs/evaluation.json` (`splits.group_dest.*`).

| Model | Threshold | Threshold Source | Precision | Recall | F1 | PR-AUC (AP) | ROC-AUC | False Alarms / 10k | TP | FP | FN | TN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m0_rule | 0.500000 | rule_definition | 1.0000 | 0.0010 | 0.0019 | 0.0023 | 0.500481 | 0.00 | 2 | 0 | 2,076 | 1,598,114 |
| m1_tabular | 0.996006 | validation | 0.9741 | 0.9947 | 0.9843 | 0.9945 | 0.999269 | 0.34 | 2,067 | 55 | 11 | 1,598,059 |
| m1_tabular_noerr | 0.997866 | validation | 0.9244 | 0.7478 | 0.8268 | 0.9068 | 0.998554 | 0.79 | 1,554 | 127 | 524 | 1,597,987 |
| m2_graph | 0.985269 | validation | 0.9718 | 0.9952 | 0.9834 | 0.9940 | 0.999433 | 0.37 | 2,068 | 60 | 10 | 1,598,054 |
| m3_community_mule | 0.996180 | validation | 0.9727 | 0.9942 | 0.9833 | 0.9940 | 0.999401 | 0.36 | 2,066 | 58 | 12 | 1,598,056 |

## 5. Model Comparison — Reference Random Split
Source: `outputs/evaluation.json` (`splits.random.*`).

| Model | Threshold | Threshold Source | Precision | Recall | F1 | PR-AUC (AP) | ROC-AUC | False Alarms / 10k | TP | FP | FN | TN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m0_rule | 0.500000 | rule_definition | 1.0000 | 0.0014 | 0.0029 | 0.0027 | 0.500714 | 0.00 | 3 | 0 | 2,098 | 1,586,968 |
| m1_tabular | 0.996603 | validation | 0.9817 | 0.9957 | 0.9887 | 0.9954 | 0.999397 | 0.25 | 2,092 | 39 | 9 | 1,586,929 |
| m1_tabular_noerr | 0.997947 | validation | 0.9013 | 0.7692 | 0.8300 | 0.9062 | 0.999090 | 1.11 | 1,616 | 177 | 485 | 1,586,791 |
| m2_graph | 0.997427 | validation | 0.9817 | 0.9952 | 0.9884 | 0.9953 | 0.999571 | 0.25 | 2,091 | 39 | 10 | 1,586,929 |
| m3_community_mule | 0.998688 | validation | 0.9812 | 0.9948 | 0.9879 | 0.9953 | 0.999427 | 0.25 | 2,090 | 40 | 11 | 1,586,928 |

## 6. Business Impact — Headline Operating Points (Time Split)
Source: `outputs/business_impact.json` (`headline_operating_table`). Operating points use the validation-selected thresholds from `outputs/evaluation.json`. Review cost = `10,000` `PaySim simulated currency units` (`assumption`). Total fraud value on time-split test set (`193,939` rows, `2,062` fraud rows) = `3,181,396,658.88`.

| Model | Operating Threshold | Threshold Source | Alerts | True Alerts | False Alerts | Alerts / 10k | Fraud Value Caught | Fraud Value Missed | Share Caught | Net Saving (Review Cost 10,000 assumption) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rule_baseline | 0.500000 | rule_definition | 9 | 9 | 0 | 0.46 | 42,898,513.56 | 3,138,498,145.32 | 1.35% | 42,808,513.56 |
| m1_tabular | 0.998236 | validation | 2,064 | 2,058 | 6 | 106.43 | 3,180,643,739.58 | 752,919.30 | 99.98% | 3,160,003,739.58 |
| m1_tabular_noerr | 0.984859 | validation | 1,989 | 1,856 | 133 | 102.56 | 3,160,064,870.70 | 21,331,788.18 | 99.33% | 3,140,174,870.70 |
| m2_graph | 0.996715 | validation | 2,063 | 2,057 | 6 | 106.37 | 3,180,642,653.42 | 754,005.46 | 99.98% | 3,160,012,653.42 |
| m3_community_mule | 0.996618 | validation | 2,060 | 2,053 | 7 | 106.22 | 3,180,642,415.69 | 754,243.19 | 99.98% | 3,160,042,415.69 |

## 7. Business Impact — Review-Cost Sensitivity (1,000 / 10,000 / 100,000)
Source: `outputs/business_impact.json` (`sensitivity_table`). `hindsight_best_test` rows are reported strictly for sensitivity inspection and are not recommended operating points.

| Review Cost (assumption) | Model | Threshold Mode | Threshold | Alerts | Fraud Value Caught | Net Financial Saving |
| --- | --- | --- | --- | --- | --- | --- |
| 1,000 | rule_baseline | hindsight_best_test (not a recommended operating point) | 0.500000 | 9 | 42,898,513.56 | 42,889,513.56 |
| 1,000 | m1_tabular | validation_threshold | 0.998236 | 2,064 | 3,180,643,739.58 | 3,178,579,739.58 |
| 1,000 | m1_tabular | hindsight_best_test (not a recommended operating point) | 0.100000 | 2,142 | 3,181,042,784.66 | 3,178,900,784.66 |
| 1,000 | m1_tabular_noerr | validation_threshold | 0.984859 | 1,989 | 3,160,064,870.70 | 3,158,075,870.70 |
| 1,000 | m1_tabular_noerr | hindsight_best_test (not a recommended operating point) | 0.500000 | 2,911 | 3,181,394,694.31 | 3,178,483,694.31 |
| 1,000 | m2_graph | validation_threshold | 0.996715 | 2,063 | 3,180,642,653.42 | 3,178,579,653.42 |
| 1,000 | m2_graph | hindsight_best_test (not a recommended operating point) | 0.900000 | 2,069 | 3,180,643,739.58 | 3,178,574,739.58 |
| 1,000 | m3_community_mule | validation_threshold | 0.996618 | 2,060 | 3,180,642,415.69 | 3,178,582,415.69 |
| 1,000 | m3_community_mule | hindsight_best_test (not a recommended operating point) | 0.900000 | 2,069 | 3,180,643,739.58 | 3,178,574,739.58 |
| 10,000 | rule_baseline | hindsight_best_test (not a recommended operating point) | 0.500000 | 9 | 42,898,513.56 | 42,808,513.56 |
| 10,000 | m1_tabular | validation_threshold | 0.998236 | 2,064 | 3,180,643,739.58 | 3,160,003,739.58 |
| 10,000 | m1_tabular | hindsight_best_test (not a recommended operating point) | 0.950000 | 2,070 | 3,180,643,739.58 | 3,159,943,739.58 |
| 10,000 | m1_tabular_noerr | validation_threshold | 0.984859 | 1,989 | 3,160,064,870.70 | 3,140,174,870.70 |
| 10,000 | m1_tabular_noerr | hindsight_best_test (not a recommended operating point) | 0.850000 | 2,507 | 3,179,692,152.91 | 3,154,622,152.91 |
| 10,000 | m2_graph | validation_threshold | 0.996715 | 2,063 | 3,180,642,653.42 | 3,160,012,653.42 |
| 10,000 | m2_graph | hindsight_best_test (not a recommended operating point) | 0.900000 | 2,069 | 3,180,643,739.58 | 3,159,953,739.58 |
| 10,000 | m3_community_mule | validation_threshold | 0.996618 | 2,060 | 3,180,642,415.69 | 3,160,042,415.69 |
| 10,000 | m3_community_mule | hindsight_best_test (not a recommended operating point) | 0.900000 | 2,069 | 3,180,643,739.58 | 3,159,953,739.58 |
| 100,000 | rule_baseline | hindsight_best_test (not a recommended operating point) | 0.500000 | 9 | 42,898,513.56 | 41,998,513.56 |
| 100,000 | m1_tabular | validation_threshold | 0.998236 | 2,064 | 3,180,643,739.58 | 2,974,243,739.58 |
| 100,000 | m1_tabular | hindsight_best_test (not a recommended operating point) | 0.950000 | 2,070 | 3,180,643,739.58 | 2,973,643,739.58 |
| 100,000 | m1_tabular_noerr | validation_threshold | 0.984859 | 1,989 | 3,160,064,870.70 | 2,961,164,870.70 |
| 100,000 | m1_tabular_noerr | hindsight_best_test (not a recommended operating point) | 0.986297 | 1,959 | 3,158,482,836.87 | 2,962,582,836.87 |
| 100,000 | m2_graph | validation_threshold | 0.996715 | 2,063 | 3,180,642,653.42 | 2,974,342,653.42 |
| 100,000 | m2_graph | hindsight_best_test (not a recommended operating point) | 0.900000 | 2,069 | 3,180,643,739.58 | 2,973,743,739.58 |
| 100,000 | m3_community_mule | validation_threshold | 0.996618 | 2,060 | 3,180,642,415.69 | 2,974,642,415.69 |
| 100,000 | m3_community_mule | hindsight_best_test (not a recommended operating point) | 0.900000 | 2,069 | 3,180,643,739.58 | 2,973,743,739.58 |

## 8. Real-Data Fraud Subgraph Component Analysis
Source: `outputs/ring_analysis.json`.

| Metric | Value | Source File | Key |
| --- | --- | --- | --- |
| Fraud accounts in fraud subgraph | 16,382 | outputs/ring_analysis.json | number_of_fraud_accounts |
| Total fraud-subgraph components | 8,169 | outputs/ring_analysis.json | number_of_components |
| Components of size 2 (1-to-1 pairs) | 8,125 | outputs/ring_analysis.json | number_of_components_by_size.2 |
| Components of size 3 (2-to-1 funnels) | 44 | outputs/ring_analysis.json | number_of_components_by_size.3 |
| Components of size >= 3 | 44 | outputs/ring_analysis.json | count_size_ge_3 |
| Components of size >= 4 | 0 | outputs/ring_analysis.json | number_of_components_by_size (>=4) |
| Exported ring summary rows | 20 | outputs/ring_analysis.json | exported_rings_count |

Empirical finding (`outputs/ring_analysis.json: finding`):
> In the PaySim dataset, real fraud transactions are overwhelmingly isolated 1-to-1 interactions (8125 components of size 2). Exactly 44 components have size >= 3 (all of size 3, where 2 distinct origin accounts transferred funds to the same destination account). There are zero multi-hop chains, cycles, or larger fraud rings of size >= 4 in the ground-truth fraud data.

## 9. Planted-Ring Synthetic Injection Test — Connected Components
Source: `outputs/planted_ring_validation.json` (`algorithms.connected_components`). Label: `synthetic injection for validation`, sample size: `200,000` rows, total planted rings: `60`, strict recovered: `21` (`35.0%`).

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

## 10. Planted-Ring Synthetic Injection Test — Label Propagation (LPA)
Source: `outputs/planted_ring_validation.json` (`algorithms.label_propagation`). Strict recovered: `13` / `60` (`21.7%`).

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

## 11. Spark Data-Size Scaling (with Superseded Prior Aborts Retained)
Source: `outputs/scaling_runs.json` (`data_size_summary`).

| Fraction | Active OK/Total | Active Statuses | Superseded Prior | Rows | Nodes | Edges | Mean Total (s) | Individual Repeats (s) | Mean Peak RSS | Min Avail RAM | Mean Shuffle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.10 | 2/2 | ok, ok | 0 | 635,856 | 1,092,959 | 635,856 | 493.36 | 490.71, 496.00 | 6.18 GiB | 1.15 GiB | 8.26 GiB |
| 0.25 | 2/2 | ok, ok | 2 | 1,590,820 | 2,509,524 | 1,590,820 | 923.50 | 1132.07, 714.93 | 6.76 GiB | 76.73 MiB | 20.19 GiB |
| 0.50 | 2/2 | ok, ok | 1 | 3,180,590 | 4,737,083 | 3,180,590 | 1364.98 | 1262.74, 1467.22 | 6.91 GiB | 461.73 MiB | 50.50 GiB |
| 1.00 | 2/2 | ok, ok | 0 | 6,362,620 | 9,073,900 | 6,362,620 | 2781.71 | 2799.15, 2764.28 | 6.98 GiB | 444.07 MiB | 126.98 GiB |

### Superseded Prior Attempts Retained (`outputs/scaling_runs.json: superseded_runs`)
| Suite | Fraction | Master | Repeat | Status | Wall (s) | Peak RSS | Min Avail RAM | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| data_scaling | 0.25 | local[*] | 1 | aborted_low_memory | 329.37 | 6.85 GiB | 282.91 MiB | Available RAM remained below 536870912 bytes for 60.4s (min_available_ram_bytes=296656896). |
| data_scaling | 0.25 | local[*] | 2 | aborted_low_memory | 849.44 | 6.88 GiB | 367.68 MiB | Available RAM remained below 536870912 bytes for 60.6s (min_available_ram_bytes=385540096). |
| data_scaling | 0.50 | local[*] | 1 | aborted_low_memory | 872.04 | 6.89 GiB | 408.79 MiB | Available RAM remained below 536870912 bytes for 60.1s (min_available_ram_bytes=428642304). |

## 12. Spark Core Scaling (Fraction `0.10`)
Source: `outputs/scaling_runs.json` (`core_scaling_summary`).

| Master | Fraction | Statuses | Mean Total (s) | Individual Repeats (s) | Speedup | Individual Speedups | Efficiency | Mean Peak RSS | Min Avail RAM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| local[1] | 0.10 | ok, ok | 926.94 | 957.10, 896.78 | 1.000x | 0.969x, 1.034x | 1.000 | 3.90 GiB | 3.05 GiB |
| local[2] | 0.10 | ok, ok | 787.97 | 713.36, 862.58 | 1.176x | 1.299x, 1.075x | 0.588 | 4.90 GiB | 1.72 GiB |
| local[4] | 0.10 | ok, ok | 742.21 | 764.09, 720.33 | 1.249x | 1.213x, 1.287x | 0.312 | 4.56 GiB | 2.34 GiB |
| local[8] | 0.10 | ok, ok | 690.28 | 689.16, 691.40 | 1.343x | 1.345x, 1.341x | 0.168 | 4.72 GiB | 2.11 GiB |
| local[16] | 0.10 | ok, ok | 667.05 | 661.96, 672.15 | 1.390x | 1.400x, 1.379x | 0.087 | 5.50 GiB | 1.33 GiB |

## 13. pandas/NetworkX vs Spark-Native Comparison (Including Full `6,362,620`-Row pandas Run)
Source: `outputs/pandas_vs_spark.json` (`summary`).

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

## 14. Storage Benchmark: CSV vs Parquet
Source: `outputs/storage_benchmark.json` (`formats`, `speedup_vs_csv`).

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

## 15. Spark Graph Engines and Full-Run Pipeline Telemetry
Source: `outputs/scalability.json`.

| Algorithm | Engine | Iterations | Converged | Full-Run Stage Time (s) | Source Key |
| --- | --- | --- | --- | --- | --- |
| degrees | spark-native | 1 | True | 30.33 | outputs/scalability.json: algorithm_info.degrees, timings_seconds.degrees |
| pagerank | spark-native | 10 | True | 144.30 | outputs/scalability.json: algorithm_info.pagerank, timings_seconds.pagerank |
| triangles | spark-native | 1 | True | 78.46 | outputs/scalability.json: algorithm_info.triangles, timings_seconds.triangles |
| components | spark-native | 9 | True | 2042.20 | outputs/scalability.json: algorithm_info.components, timings_seconds.components |
| lpa | spark-native | 5 | False | 487.88 | outputs/scalability.json: algorithm_info.lpa, timings_seconds.lpa |
| behaviour | spark-native | 1 | True | 57.24 | outputs/scalability.json: algorithm_info.behaviour, timings_seconds.behaviour_features |

| Phase / Metric | Value | Source File | Key |
| --- | --- | --- | --- |
| ingestion_and_validation | 60.09 | outputs/scalability.json | timings_seconds.ingestion_and_validation |
| graph_construction | 15.04 | outputs/scalability.json | timings_seconds.graph_construction |
| graph_feature_engineering (phase) | 2908.38 | outputs/scalability.json | timings_seconds.graph_feature_engineering |
| model_training (3 splits) | 99.71 | outputs/scalability.json | timings_seconds.model_training |
| distributed_scoring_and_eval (3 splits) | 691.35 | outputs/scalability.json | timings_seconds.distributed_scoring_and_eval |
| Total Spark shuffle read | 68.41 GiB | outputs/scalability.json | shuffle_read_bytes |
| Total Spark shuffle write | 61.67 GiB | outputs/scalability.json | shuffle_write_bytes |
| Spark completed stage count | 945 | outputs/scalability.json | stage_count |
| Sampled training rows (3 splits sum) | 147,532 | outputs/scalability.json | sampled_training_rows |
| Memory warning recorded | Low memory warning: available RAM (1.82 GB) dropped below 2.0 GB threshold. | outputs/scalability.json | memory_warning |

## 16. Machine, Spark Settings, and Software Versions
Source: `outputs/run_info.json`, `requirements.txt`.

| Component | Measured Value | Source File | Key |
| --- | --- | --- | --- |
| Python | 3.13.7 (tags/v3.13.7:bcee1c3, Aug 14 2025, 14:15:11) [MSC v.1944 64 bit (AMD64)] | outputs/run_info.json | python |
| Java | openjdk version "20.0.2" 2023-07-18 | outputs/run_info.json | java |
| PySpark | 4.2.0 | outputs/run_info.json | pyspark |
| pandas | 3.0.0 | outputs/run_info.json | pandas |
| scikit-learn | 1.8.0 | outputs/run_info.json | sklearn |
| XGBoost | 3.4.1 | outputs/run_info.json | xgboost |
| CPU cores (physical / logical) | 16 / 22 | outputs/run_info.json | cores.physical, cores.logical |
| Total RAM (GiB) | 15.37 | outputs/run_info.json | ram_gb.total |
| Spark driver memory | 6g | outputs/run_info.json | config.SPARK_DRIVER_MEMORY |
| Spark shuffle partitions | 48 | outputs/run_info.json | config.SPARK_SHUFFLE_PARTITIONS |
| Max training sample rows per split | 100,000 | outputs/run_info.json | config.max_training_rows |
| Dataset SHA-256 | 16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b | outputs/run_info.json | dataset.sha256 |
| Full-run git commit | 81c9ad4490bcefa60619177cc21c2f4157eadedf | outputs/run_info.json | git_commit |
| Pinned packages (`requirements.txt`) | pyspark==4.2.0, pandas==3.0.0, numpy==2.2.4, scikit-learn==1.8.0, xgboost==3.4.1, pyarrow==26.0.0, networkx==3.6.1, plotly==6.5.2, streamlit==1.56.0, psutil==7.2.2, joblib==1.5.3 | requirements.txt | all non-comment lines |
