"""src/audit.py -- Audit raw CSV datasets and verify anomaly reference checkpoints.
"""
import os
import pandas as pd
import numpy as np

RAW_DIR = os.path.join("data", "raw")

def audit_raw_data():
    customers_path = os.path.join(RAW_DIR, "customers.csv")
    products_path = os.path.join(RAW_DIR, "products.csv")
    orders_path = os.path.join(RAW_DIR, "orders.csv")
    tickets_path = os.path.join(RAW_DIR, "tickets.csv")

    df_customers = pd.read_csv(customers_path)
    df_products = pd.read_csv(products_path)
    df_orders = pd.read_csv(orders_path, dtype=str)  # read as str to inspect raw unparsed fields
    df_tickets = pd.read_csv(tickets_path, keep_default_na=False)

    print("================ RAW DATA AUDIT REPORT ================\n")

    tables = {
        "customers.csv": df_customers,
        "products.csv": df_products,
        "orders.csv": df_orders,
        "tickets.csv": df_tickets
    }

    for name, df in tables.items():
        print(f"--- Table: {name} (Shape: {df.shape[0]} rows, {df.shape[1]} cols) ---")
        null_counts = df.isna().sum()
        null_pcts = (null_counts / len(df)) * 100
        dtypes = df.dtypes
        audit_df = pd.DataFrame({
            "dtype": dtypes,
            "missing_count": null_counts,
            "missing_pct": null_pcts.round(2)
        })
        print(audit_df.to_string())
        print()

    # Specific Audit Figures for orders.csv
    # Re-read orders without forcing str for calculations if needed, but let's check with str and raw inspections
    raw_orders_rows = len(df_orders)
    exact_duplicate_rows = df_orders.duplicated().sum()
    pending_rows = (df_orders["status"] == "pending").sum()
    distinct_order_ids = df_orders["order_id"].nunique()

    # order_amount non-numeric
    numeric_amounts = pd.to_numeric(df_orders["order_amount"], errors="coerce")
    non_numeric_amounts = numeric_amounts.isna().sum()

    # delivery_minutes checks
    numeric_deliv = pd.to_numeric(df_orders["delivery_minutes"], errors="coerce")
    negative_delivery = (numeric_deliv < 0).sum()
    blank_delivery = df_orders["delivery_minutes"].isna() | (df_orders["delivery_minutes"].str.strip() == "")
    blank_delivery_count = blank_delivery.sum()

    # Customers checks
    raw_cust_rows = len(df_customers)
    # Blank age
    blank_age = df_customers["age"].isna()
    blank_age_count = blank_age.sum()

    # Age values outside 10-100 (among non-blank numeric ages)
    numeric_age = pd.to_numeric(df_customers["age"], errors="coerce")
    outside_age_count = ((numeric_age < 10) | (numeric_age > 100)).sum()

    # Distinct raw city spellings
    distinct_raw_cities = df_customers["city"].nunique()

    # Distinct non-blank raw gender spellings
    non_blank_gender = df_customers["gender"].dropna().loc[lambda s: s.str.strip() != ""]
    distinct_non_blank_genders = non_blank_gender.nunique()
    blank_gender_count = (df_customers["gender"].isna() | (df_customers["gender"].astype(str).str.strip() == "")).sum()

    audit_summary = [
        ("Raw orders.csv rows", raw_orders_rows, 13084),
        ("Exact duplicate rows (orders.csv)", exact_duplicate_rows, 119),
        ("Pending status rows (orders.csv)", pending_rows, 656),
        ("Distinct order_id values", distinct_order_ids, 12309),
        ("Non-numeric order_amount count", non_numeric_amounts, 120),
        ("Negative delivery_minutes count", negative_delivery, 106),
        ("Blank delivery_minutes count", blank_delivery_count, 2541),
        ("Blank age count (customers.csv)", blank_age_count, 61),
        ("Age outside 10-100 count (customers.csv)", outside_age_count, 6),
        ("Distinct raw city spellings", distinct_raw_cities, 24),
        ("Distinct non-blank raw gender spellings", distinct_non_blank_genders, 10),
        ("Blank gender count (customers.csv)", blank_gender_count, 30)
    ]

    print("================ BENCHMARK CHECKPOINTS ================")
    summary_df = pd.DataFrame(audit_summary, columns=["Metric", "Observed", "Reference"])
    summary_df["Match"] = summary_df["Observed"] == summary_df["Reference"]
    print(summary_df.to_string(index=False))
    print()

    all_match = summary_df["Match"].all()
    if all_match:
        print("ALL AUDIT CHECKPOINTS MATCH REFERENCE EXACTLY!")
    else:
        print("WARNING: Some audit metrics do not match reference values!")

    return summary_df

if __name__ == "__main__":
    audit_raw_data()
