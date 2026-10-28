# PaySim Financial Fraud Analytics

An end-to-end local PySpark and GraphFrames pipeline for detecting suspicious
transactions and coordinated account communities in the Kaggle PaySim dataset
(`ealaxi/paysim1`).

## Requirements

- Python 3.10 or newer
- Java 8, 11, or 17 available on `PATH` (required by Spark)
- Kaggle API credentials for automatic download, unless the CSV is supplied
  locally

Install Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For automatic download, install/configure the Kaggle CLI credentials using
Kaggle's documented `kaggle.json` setup. Alternatively, place the dataset CSV
in `data\` or supply its path with `--data`. The expected CSV columns are the
standard PaySim fields: `step`, `type`, `amount`, origin/destination account
names and balances, `isFraud`, and `isFlaggedFraud`.

## Run

```powershell
python main.py
```

Examples:

```powershell
python main.py --data data\PS_20174392719_1491204439457_log.csv
python main.py --max-training-rows 50000 --output-dir outputs
```

The first run downloads the GraphFrames JVM package through Spark's package
resolver. Set `GRAPHFRAMES_PACKAGE` to override the Maven coordinate (the
default GraphFrames 0.9.3 coordinate matches the Python API package) or
`SPARK_SHUFFLE_PARTITIONS` / `SPARK_DEFAULT_PARALLELISM` to tune local
parallelism. `SPARK_DRIVER_MEMORY` defaults to `8g` for the full PaySim graph
and can be lowered or raised to match available RAM.

## Pipeline

1. `data_ingestion.py` starts Spark with `local[*]`, downloads/discovers the
   CSV, applies an explicit schema, and removes malformed transactions.
2. `graph_construction.py` creates directed transaction edges between account
   vertices.
3. `feature_engineering.py` calculates in/out-degree, PageRank, triangle count,
   GraphX connected components, and GraphFrames Label Propagation communities,
   then joins account metrics back to transactions.
4. `models.py` trains three scikit-learn random-forest classifiers: a tabular
   baseline, a structural-graph model, and a community/ring model. A
   class-aware sample is collected to the driver; the default maximum is
   100,000 rows.
5. `evaluation.py` reports transaction precision, recall, F1, and ROC AUC;
   ring metrics aggregate labels and predictions by discovered origin-account
   community. Ring ground truth is operationally defined as a discovered
   community containing at least one labeled fraudulent transaction.
6. `main.py` records stage runtimes and Spark UI executor-memory and shuffle
   metrics.

JSON reports are written to `outputs\evaluation.json` and
`outputs\scalability.json`. Spark's package resolver must be able to reach
Maven Central for the GraphFrames JVM dependency. PaySim's `isFlaggedFraud`
column is deliberately excluded from model features to avoid target leakage.

## Web dashboard

After running the analytics pipeline, start the local dashboard:

```powershell
python web_app.py --open
```

Open `http://127.0.0.1:8000` if it does not open automatically. The page shows
the latest saved transaction/ring evaluation and Spark scalability reports,
the five pipeline stages, dataset and run status, and a searchable function
catalog with each function's signature, inputs, and outputs. Use **Refresh
data** to reload reports after another pipeline run. The dashboard only
displays existing results; start or rerun analytics separately with
`python main.py`. Stop the local web server with Ctrl+C.
