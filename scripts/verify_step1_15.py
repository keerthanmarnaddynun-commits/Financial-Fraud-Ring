"""Script to verify STEP 1.15 outputs and exact confusion counts."""

import json
from pathlib import Path
import pandas as pd


def verify_outputs(output_dir: Path = Path("outputs_test")) -> None:
    output_dir = Path(output_dir)
    print(f"=== VERIFYING STEP 1.15 OUTPUTS IN {output_dir} ===")

    # 1. Check enriched features parquet
    enriched_parquet = output_dir / "enriched_features.parquet"
    print(f"1. Enriched features parquet exists: {enriched_parquet.exists()}")
    assert enriched_parquet.exists(), f"Missing {enriched_parquet}"

    # 2. Check models saved per split
    print("\n2. Checking per-split model directories:")
    for split in ["time", "group_dest", "random"]:
        split_dir = output_dir / "models" / split
        exists = split_dir.exists()
        models_found = list(split_dir.glob("*.joblib")) if exists else []
        print(f"   - Split '{split}': exists={exists}, joblib_files={[f.name for f in models_found]}")
        assert exists, f"Missing models directory for split: {split}"
        assert len(models_found) >= 4, f"Expected at least 4 models in {split_dir}"

    # 3. Check scalability.json for flat timings, clean model_names, and memory_warning
    scalability_path = output_dir / "scalability.json"
    print(f"\n3. Checking scalability.json ({scalability_path.exists()}):")
    assert scalability_path.exists(), f"Missing {scalability_path}"
    with open(scalability_path, "r", encoding="utf-8") as f:
        scal = json.load(f)

    timings = scal.get("timings_seconds", {})
    has_nested_timings = isinstance(timings.get("timings_seconds"), dict)
    print(f"   - Timings seconds flat (no nested 'timings_seconds'): {not has_nested_timings}")
    print(f"   - Timing keys: {list(timings.keys())}")
    print(f"   - Model names in scalability: {scal.get('model_names')}")
    print(f"   - Memory warning recorded: {scal.get('memory_warning')}")
    assert not has_nested_timings, "timings_seconds contains nested timings_seconds!"
    for legacy_alias in ["tabular_baseline", "Baseline", "graph_features", "community_ring"]:
        assert legacy_alias not in scal.get("model_names", []), f"Found legacy alias '{legacy_alias}' in model_names"

    # 4. Check exact confusion counts across all splits
    eval_path = output_dir / "evaluation.json"
    print(f"\n4. Checking evaluation.json ({eval_path.exists()}):")
    assert eval_path.exists(), f"Missing {eval_path}"
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    splits = eval_data.get("splits", {})
    for split_name in ["time", "group_dest", "random"]:
        parquet_file = output_dir / f"scored_{split_name}.parquet"
        assert parquet_file.exists(), f"Missing {parquet_file}"
        df_scored = pd.read_parquet(parquet_file)
        actual_rows = len(df_scored)
        actual_fraud = int((df_scored["isFraud"] == 1).sum())

        print(f"\n   === Split: {split_name} (Scored Parquet: {actual_rows} rows, {actual_fraud} fraud) ===")
        split_eval = splits.get(split_name, {})
        for m_name, m_info in split_eval.items():
            conf = m_info.get("at_threshold", {}).get("confusion", {})
            tp = int(conf.get("tp", 0))
            fp = int(conf.get("fp", 0))
            fn = int(conf.get("fn", 0))
            tn = int(conf.get("tn", 0))
            thresh = m_info.get("threshold")

            tp_fn_sum = tp + fn
            total_sum = tp + fp + fn + tn
            tp_fn_equal = (tp_fn_sum == actual_fraud)
            total_equal = (total_sum == actual_rows)

            print(
                f"   Model {m_name:<18} (th={thresh:.4f}): "
                f"tp={tp:<4} fn={fn:<4} -> tp+fn={tp_fn_sum:<4} (equals fraud: {tp_fn_equal}) | "
                f"total={total_sum} (equals rows: {total_equal})"
            )
            assert tp_fn_equal, f"Mismatch in {split_name}/{m_name}: tp+fn ({tp_fn_sum}) != actual fraud ({actual_fraud})"
            assert total_equal, f"Mismatch in {split_name}/{m_name}: total ({total_sum}) != actual rows ({actual_rows})"

    print("\n[SUCCESS] All checks passed cleanly with exact count conservation!")


if __name__ == "__main__":
    verify_outputs()
