"""src/validate_part_a.py -- Validate post-cleaning checkpoints for Part A.
Must print exactly 'ALL CHECKPOINTS PASSED' when all hold.
"""
import os
import sqlite3
import pandas as pd
import numpy as np

CLEAN_DIR = os.path.join("data", "clean")
DB_PATH = os.path.join("data", "instamart.db")

def validate_part_a():
    df_customers = pd.read_csv(os.path.join(CLEAN_DIR, "customers.csv"))
    df_products = pd.read_csv(os.path.join(CLEAN_DIR, "products.csv"))
    df_orders = pd.read_csv(os.path.join(CLEAN_DIR, "orders.csv"))

    # Also connect to SQLite database
    conn = sqlite3.connect(DB_PATH)
    db_customers = pd.read_sql_query("SELECT * FROM customers", conn)
    db_products = pd.read_sql_query("SELECT * FROM products", conn)
    db_orders = pd.read_sql_query("SELECT * FROM orders", conn)
    conn.close()

    # Verify both CSV and SQLite have identical row counts
    assert len(df_customers) == len(db_customers) == 1200, f"Customer row count mismatch: {len(df_customers)}"
    assert len(df_products) == len(db_products) == 40, f"Product row count mismatch: {len(df_products)}"
    assert len(df_orders) == len(db_orders) == 12309, f"Order row count mismatch: {len(df_orders)}"

    # Checkpoint 1: Customers city distribution
    city_counts = df_customers["city"].value_counts().to_dict()
    ref_cities = {
        "Pune": 227,
        "Delhi": 210,
        "Bengaluru": 197,
        "Chennai": 191,
        "Hyderabad": 191,
        "Mumbai": 184
    }
    assert city_counts == ref_cities, f"City counts mismatch: {city_counts} vs {ref_cities}"

    # Checkpoint 2: Customers gender distribution
    gender_counts = df_customers["gender"].value_counts().to_dict()
    ref_genders = {
        "female": 610,
        "male": 560,
        "unknown": 30
    }
    assert gender_counts == ref_genders, f"Gender counts mismatch: {gender_counts} vs {ref_genders}"

    # Checkpoint 3: Customers age: 0 missing, every age between 18 and 60
    assert df_customers["age"].isna().sum() == 0, "Missing values found in customer age"
    assert (df_customers["age"] >= 18).all() and (df_customers["age"] <= 60).all(), "Customer age outside 18-60"

    # Checkpoint 4: Orders row count & distinct order_id
    assert len(df_orders) == 12309, f"Expected 12309 orders, got {len(df_orders)}"
    assert df_orders["order_id"].nunique() == 12309, f"Expected 12309 unique order_ids, got {df_orders['order_id'].nunique()}"

    # Checkpoint 5: Orders status counts (completed: 11110, cancelled: 1199, pending: 0)
    status_counts = df_orders["status"].value_counts().to_dict()
    assert status_counts.get("completed", 0) == 11110, f"Completed orders mismatch: {status_counts.get('completed')}"
    assert status_counts.get("cancelled", 0) == 1199, f"Cancelled orders mismatch: {status_counts.get('cancelled')}"
    assert status_counts.get("pending", 0) == 0, f"Pending orders found: {status_counts.get('pending')}"

    # Checkpoint 6: 0 missing order_amount
    assert df_orders["order_amount"].isna().sum() == 0, "Missing order_amount values found"

    # Checkpoint 7: Missing delivery_minutes (0 among completed, 1199 among cancelled)
    completed_orders = df_orders[df_orders["status"] == "completed"]
    cancelled_orders = df_orders[df_orders["status"] == "cancelled"]
    assert completed_orders["delivery_minutes"].isna().sum() == 0, "Missing delivery_minutes among completed orders"
    assert cancelled_orders["delivery_minutes"].isna().sum() == 1199, "Missing delivery_minutes among cancelled orders not 1199"

    # Checkpoint 8: Sum of order_amount over completed orders is INR 2,998,668.80
    completed_revenue = round(completed_orders["order_amount"].sum(), 2)
    assert abs(completed_revenue - 2998668.80) < 0.01, f"Completed revenue mismatch: {completed_revenue} vs 2998668.80"

    # Checkpoint 9: order_ts format check
    ts_lengths = df_orders["order_ts"].str.len().unique()
    assert len(ts_lengths) == 1 and ts_lengths[0] == 19, "order_ts not in YYYY-MM-DD HH:MM:SS format"

    print("ALL CHECKPOINTS PASSED")

if __name__ == "__main__":
    validate_part_a()
