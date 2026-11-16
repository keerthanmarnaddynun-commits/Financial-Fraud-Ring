# PaySim Financial Fraud Analytics

This repository contains a local PySpark pipeline for the PaySim fraud dataset. The project is intentionally kept honest: dashboards and documentation read from `outputs\` only, and any missing values are shown as `n/a` instead of hard-coded defaults.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py --sample-fraction 0.01
streamlit run app.py
```

The data loader now keeps merchant destinations (`nameDest` starting with `M`) as valid graph nodes, rather than silently dropping all merchant payments. That was a known issue in the original implementation and is now corrected.

## Key files

- `data_ingestion.py`: schema validation and data-quality logic
- `main.py`: pipeline runner and JSON output writer
- `feature_engineering.py`: account graph features
- `models.py`: model training logic
- `evaluation.py`: metrics and confusion-matrix reporting
- `app.py`: dashboard that reads from `outputs\` only
- `outputs\`: single source of truth for evaluation results

## Output files

The project expects results in:

- `outputs\evaluation.json`
- `outputs\scalability.json`
- `outputs\run_info.json`
- `outputs\data_quality.json`

If those files do not exist, the dashboard shows a friendly message instead of inventing numbers.

## Notes

- `isFlaggedFraud` is not treated as a feature in the model pipeline.
- The current project is a real-data Spark workflow; it does not rely on synthetic fake results.
- The legacy `output\` folder is no longer the source of truth for the dashboard.
