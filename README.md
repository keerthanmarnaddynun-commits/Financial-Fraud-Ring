# Financial Fraud Analytics with PySpark GraphFrames & Machine Learning

An end-to-end distributed graph analytics and machine learning pipeline designed to detect financial transaction fraud and organized fraud rings (collusion syndicates). Built using Apache Spark (`GraphFrames`), NetworkX, Scikit-Learn, and XGBoost on the PaySim financial dataset.

---

## 🏛 Architecture Overview

```
                      Raw Data / PaySim Dataset
                                │
                                ▼
                   ┌─────────────────────────┐
                   │   Data Ingestion & ETL  │
                   │ (Validation, Parquet)   │
                   └────────────┬────────────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │   Graph Construction    │
                   │ (Nodes: Accounts)       │
                   │ (Edges: Transactions)   │
                   └────────────┬────────────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │   Feature Engineering   │
                   ├─────────────────────────┤
                   │ • In/Out Degrees        │
                   │ • PageRank Centrality   │
                   │ • Connected Components  │
                   │ • Triangle Counting     │
                   │ • LPA Fraud Ring Detect │
                   │ • Node2Vec Embeddings   │
                   └────────────┬────────────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │  Comparative Modeling   │
                   ├─────────────────────────┤
                   │ 1. Baseline (Tabular)   │
                   │ 2. Model 1 (+ Node2Vec) │
                   │ 3. Model 2 (+ GraphX)   │
                   │ 4. Model 3 (+ Ring/LPA) │
                   └────────────┬────────────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │ Comprehensive Eval &    │
                   │ Systems Benchmarking    │
                   └─────────────────────────┘
```

---

## 📂 Modular Structure

```
├── requirements.txt               # Project dependencies
├── README.md                      # Architecture and user documentation
├── run_pipeline.py                # Master CLI execution pipeline
├── data/
│   ├── raw/                       # Raw CSV transaction inputs
│   └── processed/                 # Partitioned and indexed Parquet files
├── output/
│   ├── models/                    # Serialized ML models (.joblib)
│   ├── metrics/                   # Transaction, Ring, and Spark scaling reports
│   └── checkpoints/               # Spark GraphFrames iterative checkpoints
└── src/
    ├── __init__.py
    ├── utils.py                   # Spark session factory, hadoop configuration, and performance tracker
    ├── data_ingestion.py          # Kaggle PaySim ingestion, fallback synthetic generator, cleaning
    ├── graph_construction.py      # Entity vertices and transaction edges GraphFrame network
    ├── feature_engineering.py     # GraphX features, LPA communities, PPMI-SVD Node2Vec embeddings
    ├── model_training.py          # 4 comparative classifiers with class imbalance compensation
    └── evaluation.py              # Transaction metrics, ring purity/completeness, Spark scaling analysis
```

---

## 🚀 Getting Started

### 1. Requirements & Dependencies
Ensure Python 3.10+ and Java 11/17 (JDK) are installed. Install Python requirements:
```bash
pip install -r requirements.txt
```

### 2. Running the Complete Pipeline
Execute the master pipeline:
```bash
python run_pipeline.py --num-records 35000 --model-type xgboost
```

#### CLI Options:
- `--num-records <int>`: Number of transactions to process/synthesize (default: `30000`).
- `--model-type <xgboost|random_forest>`: Classifier model architecture (default: `xgboost`).
- `--force-synthetic`: Force generate fresh synthetic PaySim data with ground-truth fraud rings.
- `--skip-scaling`: Skip the multi-point Spark system scaling benchmark for quicker execution.

---

## 🔬 Comparative Models

| Model | Feature Representation | Primary Objective |
| :--- | :--- | :--- |
| **Baseline** | Raw tabular amounts, initial/final balances, transaction types, error discrepancies | Isolate baseline tabular detection power |
| **Model 1 (Node2Vec)** | Baseline + 16D continuous graph embedding vectors + cosine/L2 similarities | Test continuous latent topological proximity |
| **Model 2 (Structural GraphX)** | Baseline + In/Out degrees, degree ratio, PageRank, triangle counts, connected components | Explicit graph topological centrality & cycle density |
| **Model 3 (Fraud Rings / LPA)** | Baseline + Community Detection (LPA), cluster sizes, intra-community transactions | Multi-account ring-level syndicate identification |

---

## 📊 Evaluation Dimensions

1. **Transaction-Level**: Precision, Recall, F1-Score, AUC-ROC, PR-AUC, Confusion Matrix.
2. **Ring-Level**:
   - *Ring Detection Rate*: Percentage of syndicated fraud communities flagged.
   - *Cluster Purity*: Concentration of fraudulent transactions within flagged clusters.
   - *Cluster Completeness & V-Measure*: Overlap between detected communities and actual fraud syndicates.
3. **Systems Performance**:
   - Execution duration per GraphX algorithm (PageRank vs TriangleCount vs ConnectedComponents vs LPA).
   - JVM Heap memory overhead and shuffle data scaling across varying graph sizes.
