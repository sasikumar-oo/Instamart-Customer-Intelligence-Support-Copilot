"""src/clean.py -- Clean raw datasets and store to CSV and SQLite database.
"""
import os
import sqlite3
import pandas as pd
import numpy as np

RAW_DIR = os.path.join("data", "raw")
CLEAN_DIR = os.path.join("data", "clean")
DB_PATH = os.path.join("data", "instamart.db")

os.makedirs(CLEAN_DIR, exist_ok=True)

def clean_customers():
    raw_path = os.path.join(RAW_DIR, "customers.csv")
    df = pd.read_csv(raw_path)

    # (a) Standardise city: strip whitespace, title case -> exactly 6 cities
    df["city"] = df["city"].astype(str).str.strip().str.title()

    # (b) Map gender: male, female, unknown (blank -> unknown)
    def normalize_gender(val):
        if pd.isna(val) or str(val).strip() == "":
            return "unknown"
        val_clean = str(val).strip().lower()
        if val_clean in ["male", "m"]:
            return "male"
        elif val_clean in ["female", "f"]:
            return "female"
        return "unknown"

    df["gender"] = df["gender"].apply(normalize_gender)

    # (c) Age: values outside 10-100 treated as missing, fill missing with city median
    age_numeric = pd.to_numeric(df["age"], errors="coerce")
    age_numeric = age_numeric.where((age_numeric >= 10) & (age_numeric <= 100), np.nan)
    df["age"] = age_numeric

    # Impute missing with city median
    city_medians = df.groupby("city")["age"].transform("median")
    df["age"] = df["age"].fillna(city_medians).astype(int)

    # (d) Convert signup_date to datetime
    df["signup_date"] = pd.to_datetime(df["signup_date"]).dt.strftime("%Y-%m-%d")

    return df

def clean_orders(df_products):
    raw_path = os.path.join(RAW_DIR, "orders.csv")
    df = pd.read_csv(raw_path)

    # 1. Convert order_ts to datetime
    df["order_ts"] = pd.to_datetime(df["order_ts"])

    # 2. Remove exact duplicate rows
    df = df.drop_duplicates()

    # 3. For each order_id keep only the record with the latest order_ts
    df = df.sort_values("order_ts").drop_duplicates(subset=["order_id"], keep="last")

    # 4. Convert order_amount to numeric, fill missing from product price formula:
    # order_amount = round(price_inr * quantity * (1 - discount_pct / 100), 2)
    df["order_amount"] = pd.to_numeric(df["order_amount"], errors="coerce")
    
    # Merge price_inr from products
    df = df.merge(df_products[["product_id", "price_inr", "category"]], on="product_id", how="left")

    calc_amount = (df["price_inr"] * df["quantity"] * (1 - df["discount_pct"] / 100.0)).round(2)
    df["order_amount"] = df["order_amount"].fillna(calc_amount)

    # 5. Set negative delivery_minutes to missing
    df["delivery_minutes"] = pd.to_numeric(df["delivery_minutes"], errors="coerce")
    df.loc[df["delivery_minutes"] < 0, "delivery_minutes"] = np.nan

    # For completed orders: fill missing delivery_minutes with median of product category among completed orders
    completed_mask = df["status"] == "completed"
    completed_cat_medians = df[completed_mask].groupby("category")["delivery_minutes"].transform("median")
    
    # Fill only completed orders that are missing delivery_minutes
    missing_completed_mask = completed_mask & df["delivery_minutes"].isna()
    df.loc[missing_completed_mask, "delivery_minutes"] = completed_cat_medians[missing_completed_mask]

    # For cancelled orders: leave delivery_minutes missing (NaN)
    df.loc[df["status"] == "cancelled", "delivery_minutes"] = np.nan

    # Format order_ts as string YYYY-MM-DD HH:MM:SS
    df["order_ts"] = df["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Drop the temporary merged columns (price_inr, category) so schema matches exactly
    df = df.drop(columns=["price_inr", "category"])

    return df

def store_clean_data(df_customers, df_products, df_orders):
    # Save to CSV
    df_customers.to_csv(os.path.join(CLEAN_DIR, "customers.csv"), index=False)
    df_products.to_csv(os.path.join(CLEAN_DIR, "products.csv"), index=False)
    df_orders.to_csv(os.path.join(CLEAN_DIR, "orders.csv"), index=False)
    print("Saved clean CSVs to data/clean/")

    # Save to SQLite database
    conn = sqlite3.connect(DB_PATH)
    df_customers.to_sql("customers", conn, if_exists="replace", index=False)
    df_products.to_sql("products", conn, if_exists="replace", index=False)
    df_orders.to_sql("orders", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    print(f"Stored clean tables in SQLite database: {DB_PATH}")

def run_cleaning_pipeline():
    df_products = pd.read_csv(os.path.join(RAW_DIR, "products.csv"))
    df_customers = clean_customers()
    df_orders = clean_orders(df_products)

    store_clean_data(df_customers, df_products, df_orders)
    return df_customers, df_products, df_orders

if __name__ == "__main__":
    run_cleaning_pipeline()
