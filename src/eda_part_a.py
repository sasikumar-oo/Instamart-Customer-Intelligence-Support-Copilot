"""src/eda_part_a.py -- Pandas cross-check, summary statistics, outlier detection, and correlation analysis.
"""
import os
import pandas as pd
import numpy as np

CLEAN_DIR = os.path.join("data", "clean")

def run_eda():
    df_customers = pd.read_csv(os.path.join(CLEAN_DIR, "customers.csv"))
    df_products = pd.read_csv(os.path.join(CLEAN_DIR, "products.csv"))
    df_orders = pd.read_csv(os.path.join(CLEAN_DIR, "orders.csv"))

    # Completed orders merged with customers and products
    df_completed = df_orders[df_orders["status"] == "completed"].copy()
    merged = df_completed.merge(df_customers, on="customer_id", how="inner")
    merged = merged.merge(df_products, on="product_id", how="inner")

    print("================ (A) PANDAS CROSS-CHECK ================\n")

    # Pivot table rebuilding Q2
    pivot_q2 = pd.pivot_table(
        merged,
        values="order_amount",
        index="city",
        columns="category",
        aggfunc="sum",
        margins=True,
        margins_name="Total"
    )
    print("--- Q2 Pivot Table (Revenues in INR) ---")
    print(pivot_q2.round(2).to_string())
    print()

    grand_total = round(pivot_q2.loc["Total", "Total"], 2)
    print(f"Pivot Grand Total: INR {grand_total:,.2f} (Expected: INR 2,998,668.80) -> Match: {abs(grand_total - 2998668.80) < 0.01}")

    # Groupby with named aggregation per city
    city_agg = df_completed.merge(df_customers[["customer_id", "city"]], on="customer_id").groupby("city").agg(
        sum_order_amount=("order_amount", "sum"),
        mean_order_amount=("order_amount", "mean"),
        count_order_amount=("order_amount", "count")
    ).round(2)

    print("\n--- City Completed Order Aggregations ---")
    print(city_agg.to_string())
    print()

    # Reference checks for Bengaluru and Mumbai
    beng_sum = city_agg.loc["Bengaluru", "sum_order_amount"]
    beng_mean = city_agg.loc["Bengaluru", "mean_order_amount"]
    beng_count = city_agg.loc["Bengaluru", "count_order_amount"]
    print(f"Bengaluru: Sum={beng_sum} (ref 523643.40), Mean={beng_mean} (ref 276.47), Count={beng_count} (ref 1894)")

    mum_sum = city_agg.loc["Mumbai", "sum_order_amount"]
    mum_mean = city_agg.loc["Mumbai", "mean_order_amount"]
    mum_count = city_agg.loc["Mumbai", "count_order_amount"]
    print(f"Mumbai: Sum={mum_sum} (ref 479058.50), Mean={mum_mean} (ref 297.00), Count={mum_count} (ref 1613)")

    print("\n================ (B) DISTRIBUTION & OUTLIERS ================\n")

    amounts = df_completed["order_amount"]
    mean_val = round(amounts.mean(), 2)
    median_val = round(amounts.median(), 2)
    skewness = round(amounts.skew(), 2)
    q1 = round(amounts.quantile(0.25), 2)
    q3 = round(amounts.quantile(0.75), 2)
    iqr = round(q3 - q1, 2)
    lower_fence = round(q1 - 1.5 * iqr, 2)
    upper_fence = round(q3 + 1.5 * iqr, 2)

    outliers_below = (amounts < lower_fence).sum()
    outliers_above = (amounts > upper_fence).sum()

    print(f"Mean: INR {mean_val:.2f} (ref 269.91)")
    print(f"Median: INR {median_val:.2f} (ref 135.00)")
    print(f"Skewness: {skewness:.2f} (ref ~4.61)")
    print(f"Q1: INR {q1:.2f} (ref 64.80)")
    print(f"Q3: INR {q3:.2f} (ref 297.00)")
    print(f"IQR: INR {iqr:.2f}")
    print(f"Lower Fence: INR {lower_fence:.2f}")
    print(f"Upper Fence: INR {upper_fence:.2f} (ref 645.30)")
    print(f"Outliers Below Lower Fence: {outliers_below} (ref 0)")
    print(f"Outliers Above Upper Fence: {outliers_above} (ref 897)")

    print("\n================ CORRELATION MATRIX ================\n")
    corr_cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
    corr_matrix = df_completed[corr_cols].corr()
    print(corr_matrix.round(4).to_string())

    print("\nPair with largest absolute correlation:")
    # Unstack and drop self correlation
    corr_pairs = corr_matrix.unstack()
    corr_pairs_no_self = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)]
    abs_corr = corr_pairs_no_self.abs().sort_values(ascending=False)
    top_pair = abs_corr.index[0]
    print(f"{top_pair[0]} and {top_pair[1]}: {corr_matrix.loc[top_pair[0], top_pair[1]]:.4f}")

    return {
        "grand_total": grand_total,
        "city_agg": city_agg,
        "mean": mean_val,
        "median": median_val,
        "skewness": skewness,
        "q1": q1,
        "q3": q3,
        "upper_fence": upper_fence,
        "outliers_above": outliers_above,
        "outliers_below": outliers_below,
        "corr_matrix": corr_matrix
    }

if __name__ == "__main__":
    run_eda()
