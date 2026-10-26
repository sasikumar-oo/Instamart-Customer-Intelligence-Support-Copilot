"""src/validate_part_c.py -- Automated validation of Part C triage results and checkpoints.
Must print 'ALL PART C CHECKPOINTS PASSED' when all hold.
"""
import os
import re
import pandas as pd

OUTPUTS_DIR = "outputs"
RAW_DIR = os.path.join("data", "raw")

def validate_part_c():
    demo_path = os.path.join(OUTPUTS_DIR, "triage_budget_demo.csv")
    results_path = os.path.join(OUTPUTS_DIR, "triage_results.csv")
    tickets_path = os.path.join(RAW_DIR, "tickets.csv")

    assert os.path.exists(demo_path), f"Missing {demo_path}"
    assert os.path.exists(results_path), f"Missing {results_path}"

    df_demo = pd.read_csv(demo_path)
    df_res = pd.read_csv(results_path, keep_default_na=False)
    df_raw = pd.read_csv(tickets_path, keep_default_na=False)

    # Check 1: 164 total rows
    assert len(df_res) == 164, f"Expected 164 rows in triage_results.csv, got {len(df_res)}"
    assert len(df_demo) == 164, f"Expected 164 rows in triage_budget_demo.csv, got {len(df_demo)}"

    # Check 2: Columns format
    expected_cols = ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]
    assert list(df_res.columns) == expected_cols, f"Columns mismatch: {list(df_res.columns)}"

    # Check 3: Status counts in full run
    status_counts = df_res["status"].value_counts().to_dict()
    assert status_counts.get("skipped_empty") == 4, f"Expected 4 skipped_empty, got {status_counts.get('skipped_empty')}"
    assert status_counts.get("flagged_long") == 4, f"Expected 4 flagged_long, got {status_counts.get('flagged_long')}"
    assert status_counts.get("classified_json") == 135, f"Expected 135 classified_json, got {status_counts.get('classified_json')}"
    assert status_counts.get("classified_keyword") == 12, f"Expected 12 classified_keyword, got {status_counts.get('classified_keyword')}"
    assert status_counts.get("flagged_invalid_output") == 9, f"Expected 9 flagged_invalid_output, got {status_counts.get('flagged_invalid_output')}"

    # Check 4: Budget exhausted count in demo
    demo_exhausted = (df_demo["status"] == "budget_exhausted").sum()
    assert demo_exhausted >= 1, f"Expected at least 1 budget_exhausted ticket in demo, got {demo_exhausted}"

    # Check 5: Routing rules consistency
    for _, row in df_res.iterrows():
        st = row["status"]
        rt = row["route"]
        pr = row["priority"]
        cat = row["category"]

        if st in ["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"]:
            assert rt == "human_review" and pr == "NORMAL", f"Routing anomaly for {row['ticket_id']}: {rt}, {pr}"
            assert cat == "", f"Category should be blank for anomalous ticket {row['ticket_id']}"
        elif cat == "fraud":
            assert rt == "security_team" and pr == "HIGH", f"Fraud routing mismatch: {rt}, {pr}"
        elif rt == "premium_desk":
            assert pr == "HIGH", f"Premium desk priority mismatch: {pr}"
        else:
            assert rt == "support_queue" and pr == "NORMAL", f"Support queue mismatch: {rt}, {pr}"

    print("ALL PART C CHECKPOINTS PASSED")

if __name__ == "__main__":
    validate_part_c()
