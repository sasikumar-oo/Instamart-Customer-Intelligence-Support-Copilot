"""src/validate_part_b.py -- Automated validation of all Part B outputs and checkpoints.
Must print 'ALL PART B CHECKPOINTS PASSED' when all hold.
"""
import os
import joblib
import pandas as pd
import numpy as np

OUTPUTS_DIR = "outputs"

def validate_part_b():
    seg_path = os.path.join(OUTPUTS_DIR, "customer_segments.csv")
    model_path = os.path.join(OUTPUTS_DIR, "lapse_model.joblib")
    ret_path = os.path.join(OUTPUTS_DIR, "retention_list.csv")

    # 1. Validate customer_segments.csv
    assert os.path.exists(seg_path), f"Missing {seg_path}"
    df_seg = pd.read_csv(seg_path)
    assert len(df_seg) == 1175, f"Expected 1175 rows, got {len(df_seg)}"
    expected_cols = ["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]
    assert list(df_seg.columns) == expected_cols, f"Columns mismatch: {list(df_seg.columns)}"

    # Check is_premium flag
    premium_counts = df_seg["is_premium"].value_counts().to_dict()
    assert premium_counts.get(True, 0) == 138, f"Expected 138 premium customers, got {premium_counts.get(True)}"
    premium_mean_money = df_seg[df_seg["is_premium"] == True]["monetary"].mean()
    non_premium_mean_money = df_seg[df_seg["is_premium"] == False]["monetary"].mean()
    assert premium_mean_money > non_premium_mean_money, "Premium segment is not the highest-mean-monetary segment"
    assert abs(premium_mean_money - 7129.70) < 50.0, f"Unexpected premium monetary mean: {premium_mean_money}"

    # 2. Validate lapse_model.joblib
    assert os.path.exists(model_path), f"Missing {model_path}"
    model = joblib.load(model_path)
    assert hasattr(model, "predict"), "Loaded model has no predict method"
    assert hasattr(model, "predict_proba"), "Loaded model has no predict_proba method"

    # 3. Validate retention_list.csv
    assert os.path.exists(ret_path), f"Missing {ret_path}"
    df_ret = pd.read_csv(ret_path)
    assert list(df_ret.columns) == ["customer_id", "lapse_probability", "actually_lapsed"]
    assert len(df_ret) <= 10 and len(df_ret) > 0, f"Expected up to 10 rows, got {len(df_ret)}"
    # Verify sorted descending by lapse_probability
    assert (df_ret["lapse_probability"].diff().dropna() <= 0).all(), "Retention list not sorted descending by probability"
    # Verify all in retention list are in premium segment
    premium_set = set(df_seg[df_seg["is_premium"] == True]["customer_id"])
    assert all(cid in premium_set for cid in df_ret["customer_id"]), "Non-premium customer found in retention list"

    print("ALL PART B CHECKPOINTS PASSED")

if __name__ == "__main__":
    validate_part_b()
