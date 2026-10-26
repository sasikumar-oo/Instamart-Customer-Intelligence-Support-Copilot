"""src/run_sql.py -- Execute queries from sql/queries.sql against data/instamart.db and verify reference outputs.
"""
import os
import sqlite3
import pandas as pd

DB_PATH = os.path.join("data", "instamart.db")
SQL_FILE = os.path.join("sql", "queries.sql")

def run_queries():
    with open(SQL_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    query_blocks = content.split("-- BREAK_QUERY --")
    conn = sqlite3.connect(DB_PATH)

    results = {}

    print("================ EXECUTING SQL QUERIES ================\n")

    for i, block in enumerate(query_blocks, start=1):
        clean_sql = block.strip()
        df = pd.read_sql_query(clean_sql, conn)
        results[f"Q{i}"] = df
        print(f"--- Q{i} Results (Shape: {df.shape[0]} rows, {df.shape[1]} cols) ---")
        print(df.head(5).to_string())
        print("...\n")

    conn.close()

    # Verification against checkpoints
    print("================ SQL CHECKPOINT VERIFICATION ================")

    # Q1 Check: 30 customers
    q1_count = len(results["Q1"])
    print(f"Q1: Expected 30 customers, got {q1_count} -> Match: {q1_count == 30}")
    assert q1_count == 30

    # Q2 Check: 30 rows, sum = 2,998,668.80, largest cell Bengaluru / Electronics = 233,909.45
    df_q2 = results["Q2"]
    q2_rows = len(df_q2)
    q2_sum = round(df_q2["revenue"].sum(), 2)
    beng_elec = df_q2[(df_q2["city"] == "Bengaluru") & (df_q2["category"] == "Electronics")]["revenue"].values[0]
    print(f"Q2: Expected 30 rows, got {q2_rows} -> Match: {q2_rows == 30}")
    print(f"Q2: Expected sum 2998668.80, got {q2_sum:.2f} -> Match: {abs(q2_sum - 2998668.80) < 0.01}")
    print(f"Q2: Expected Bengaluru/Electronics 233909.45, got {beng_elec:.2f} -> Match: {abs(beng_elec - 233909.45) < 0.01}")
    assert q2_rows == 30
    assert abs(q2_sum - 2998668.80) < 0.01
    assert abs(beng_elec - 233909.45) < 0.01

    # Q3 Check: 11 products
    q3_count = len(results["Q3"])
    print(f"Q3: Expected 11 products, got {q3_count} -> Match: {q3_count == 11}")
    assert q3_count == 11

    # Q4 Check: 25 customers
    q4_count = len(results["Q4"])
    print(f"Q4: Expected 25 customers, got {q4_count} -> Match: {q4_count == 25}")
    assert q4_count == 25

    # Q5 Check: 550 customers, avg 2552.06
    df_q5 = results["Q5"]
    q5_count = len(df_q5)
    q5_avg = df_q5["avg_benchmark"].values[0]
    print(f"Q5: Expected 550 customers, got {q5_count} -> Match: {q5_count == 550}")
    print(f"Q5: Expected avg 2552.06, got {q5_avg:.2f} -> Match: {abs(q5_avg - 2552.06) < 0.01}")
    assert q5_count == 550
    assert abs(q5_avg - 2552.06) < 0.01

    # Q6 Check: Top customer per city and 8 rows where RANK != ROW_NUMBER
    df_q6 = results["Q6"]
    top_per_city = df_q6[df_q6["spend_rank"] == 1].set_index("city")
    ref_tops = {
        "Bengaluru": ("C0082", 12856.20),
        "Chennai": ("C0930", 10151.40),
        "Delhi": ("C0291", 13742.60),
        "Hyderabad": ("C0743", 12730.05),
        "Mumbai": ("C0693", 14766.55),
        "Pune": ("C0426", 10377.10)
    }
    for city, (ref_cid, ref_spend) in ref_tops.items():
        obs_cid = top_per_city.loc[city, "customer_id"]
        obs_spend = top_per_city.loc[city, "total_spend"]
        match_city = (obs_cid == ref_cid) and abs(obs_spend - ref_spend) < 0.01
        print(f"Q6 top {city}: Expected {ref_cid} ({ref_spend}), got {obs_cid} ({obs_spend}) -> Match: {match_city}")
        assert match_city

    rank_diff_count = (df_q6["spend_rank"] != df_q6["spend_row_number"]).sum()
    print(f"Q6: Exactly 8 rows where RANK() != ROW_NUMBER(): got {rank_diff_count} -> Match: {rank_diff_count == 8}")
    assert rank_diff_count == 8

    # Q7 Check: 183 daily rows, last cumulative revenue 2,998,668.80
    df_q7 = results["Q7"]
    q7_count = len(df_q7)
    last_cum_rev = df_q7["cumulative_revenue"].iloc[-1]
    print(f"Q7: Expected 183 daily rows, got {q7_count} -> Match: {q7_count == 183}")
    print(f"Q7: Expected last cumulative 2998668.80, got {last_cum_rev:.2f} -> Match: {abs(last_cum_rev - 2998668.80) < 0.01}")
    assert q7_count == 183
    assert abs(last_cum_rev - 2998668.80) < 0.01

    print("\nALL SQL QUERIES AND CHECKPOINTS PASSED PERFECTLY!")

if __name__ == "__main__":
    run_queries()
