"""src/validate_part_d.py -- Automated validation of Part D RAG and Agent outputs and checkpoints.
Must print 'ALL PART D CHECKPOINTS PASSED' when all hold.
"""
import os
import json
import pandas as pd

OUTPUTS_DIR = "outputs"
STATE_DIR = "state"
POLICIES_DIR = os.path.join("data", "policies")

def validate_part_d():
    runs_file = os.path.join(OUTPUTS_DIR, "agent_runs.jsonl")
    ledger_file = os.path.join(OUTPUTS_DIR, "refund_ledger.jsonl")

    # 1. Validate outputs/agent_runs.jsonl
    assert os.path.exists(runs_file), f"Missing {runs_file}"
    with open(runs_file, "r", encoding="utf-8") as f:
        runs = [json.loads(line) for line in f]

    assert len(runs) == 7, f"Expected 7 scenario runs, got {len(runs)}"
    scenario_ids = [r["scenario_id"] for r in runs]
    assert scenario_ids == ["S1", "S2", "S3", "S4", "S5", "S6", "S7"], f"Scenario mismatch: {scenario_ids}"

    # Status checks
    expected_statuses = {
        "S1": "REFUND",
        "S2": "REFUND",
        "S3": "ESCALATE",
        "S4": "DENIED",
        "S5": "DENIED",
        "S6": "NOT_FOUND",
        "S7": "FALLBACK_MAX_ITERATIONS"
    }
    for r in runs:
        sid = r["scenario_id"]
        exp_st = expected_statuses[sid]
        assert r["status"] == exp_st, f"Scenario {sid} status mismatch: {r['status']} vs {exp_st}"

    # S1 model calls check
    s1_run = next(r for r in runs if r["scenario_id"] == "S1")
    assert s1_run["llm_calls"] == 5, f"Expected 5 calls for S1, got {s1_run['llm_calls']}"

    # 2. Validate outputs/refund_ledger.jsonl
    assert os.path.exists(ledger_file), f"Missing {ledger_file}"
    with open(ledger_file, "r", encoding="utf-8") as f:
        ledger_entries = [json.loads(line) for line in f]
    assert len(ledger_entries) >= 3, f"Expected at least 3 ledger entries, got {len(ledger_entries)}"
    tools_used = {e["tool"] for e in ledger_entries}
    assert "issue_refund" in tools_used and "escalate_to_human" in tools_used, "Missing issue_refund or escalate_to_human in ledger"

    # 3. Validate state/ checkpoints
    assert os.path.exists(STATE_DIR), "Missing state/ directory"
    checkpoints = [f for f in os.listdir(STATE_DIR) if f.endswith(".json")]
    assert len(checkpoints) >= 1, "No checkpoint files found in state/"

    print("ALL PART D CHECKPOINTS PASSED")

if __name__ == "__main__":
    validate_part_d()
