# PaySim Financial Fraud Analytics

This repository contains a local PySpark pipeline for the PaySim fraud dataset. The project is intentionally kept honest: dashboards and documentation read from `outputs\` only, and any missing values are shown as `n/a` instead of hard-coded defaults.

## Environment & Prerequisites

- **Python**: 3.13.7
- **Java**: OpenJDK 20.0.2 (Temurin). *Note: Apache Spark 4.2.0 officially targets Java 17 and Java 21 LTS; Temurin 20.0.2 executes successfully with incubator vector module warnings.*
- **PySpark**: 4.2.0 (local mode)

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt

# Run sample pipeline (fast test)
python main.py --sample-fraction 0.02 --output-dir outputs_test

# Run full pipeline
python main.py --output-dir outputs

# Run Streamlit dashboard
streamlit run app.py

# Run test suite
python -m pytest -q
```

The data loader keeps merchant destinations (`nameDest` starting with `M`) as valid graph nodes, rather than silently dropping all merchant payments. That was a known issue in the original implementation and is now corrected.

## Model Comparison Ladder

All machine learning models are trained using XGBoost with identical hyperparameters (`configs/model_config.json`) evaluated across time, destination group, and random splits:

| Model | Feature Set | Description |
|---|---|---|
| **M0: Rule Baseline** | None (`isFlaggedFraud`) | Bank rule (`isFlaggedFraud`: single transfer > 200,000) evaluated as a decision rule |
| **M1: Tabular Baseline** | BASE | Amounts, balances, simulator balance error accounting, merchant flag, type one-hot |
| **M1: Tabular (no error)**| BASE_NOERR | Ablation removing simulator balance errors (`errorBalanceOrig`, `errorBalanceDest`) |
| **M2: Graph + Structural**| BASE + STRUCT | In/out degree, PageRank, triangle count, connected component size |
| **M3: Community + Mule**  | BASE + STRUCT + COMM + BEHAV | Graph features + LPA community size, co-membership, and mule behavioural flow ratios |

### Leakage Prevention
Neither `isFlaggedFraud` nor `step` is included in any ML model feature list to prevent target and temporal leakage. `isFlaggedFraud` is evaluated purely as the M0 rule baseline comparison.

## Key Files

- `data_ingestion.py`: schema validation, merchant retention, and data-quality reporting
- `main.py`: end-to-end pipeline runner
- `feature_sets.py`: formal feature set definitions (BASE, STRUCT, COMM, BEHAV)
- `feature_engineering.py`: Spark-native graph algorithms (degrees, PageRank, triangles, components, LPA)
- `splits.py`: leakage-free time, group_dest, and random split generators and threshold optimization
- `models.py`: XGBoost pipeline creation, model training, and artefact persistence
- `evaluation.py`: distributed batch scoring (`mapInPandas`), multi-split metrics, and PR curves
- `stage_timer.py`: wall-clock and process-tree RSS memory delta tracking
- `spark_session.py`: unified Spark session builder
- `app.py`: interactive Streamlit dashboard reading empirical results strictly from `outputs\`
- `outputs\`: single source of truth for evaluation results

## Output Files

The project expects results in:

- `outputs\evaluation.json`: multi-split performance metrics (precision, recall, f1, confusion, PR-AUC)
- `outputs\scalability.json`: stage timings, peak RSS bytes, shuffle volume, engine reporting
- `outputs\run_info.json`: complete execution metadata, git commit, dataset hash, software versions
- `outputs\data_quality.json`: raw rows read, rules-based drops, cleaned rows kept
- `outputs\pr_curves.csv`: measured PR curve coordinates across models and splits
- `outputs\models\`: persisted joblib model pipelines and feature dictionaries
