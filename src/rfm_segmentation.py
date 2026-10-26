"""src/rfm_segmentation.py -- RFM computation, K-Means clustering, PCA, profiling and segment output.
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

CLEAN_DIR = os.path.join("data", "clean")
OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

def compute_rfm():
    df_orders = pd.read_csv(os.path.join(CLEAN_DIR, "orders.csv"))
    df_customers = pd.read_csv(os.path.join(CLEAN_DIR, "customers.csv"))

    # Filter completed orders only
    completed = df_orders[df_orders["status"] == "completed"].copy()
    completed["order_ts"] = pd.to_datetime(completed["order_ts"])

    snapshot = pd.Timestamp("2025-10-01 00:00:00")

    # Group by customer_id
    rfm = completed.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    ).reset_index()

    # Recency = whole days between snapshot and most recent completed order
    rfm["recency"] = (snapshot - rfm["last_order_ts"]).dt.days
    rfm["monetary"] = rfm["monetary"].round(2)

    rfm = rfm[["customer_id", "recency", "frequency", "monetary"]].sort_values("customer_id").reset_index(drop=True)

    print("================ RFM TABLE SUMMARY ================")
    print(f"Total customers with completed orders: {len(rfm)} (Reference: 1175)")
    print(f"Recency: min={rfm['recency'].min()}, median={rfm['recency'].median()}, max={rfm['recency'].max()} (Ref: median 19, min 0, max 180)")
    print(f"Frequency: mean={rfm['frequency'].mean():.2f}, max={rfm['frequency'].max()} (Ref: mean 9.46, max 33)")
    print(f"Monetary: mean={rfm['monetary'].mean():.2f}, max={rfm['monetary'].max():.2f} (Ref: mean 2552.06, max 14766.55)")
    print()

    return rfm

def cluster_rfm(rfm):
    feature_cols = ["recency", "frequency", "monetary"]
    X = rfm[feature_cols].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("================ K-MEANS GRID EVALUATION (K=2 to 8) ================")
    results = []
    kmeans_models = {}

    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        results.append({
            "K": k,
            "Inertia": round(km.inertia_, 1),
            "Silhouette": round(sil, 3)
        })
        kmeans_models[k] = (km, labels)

    eval_df = pd.DataFrame(results)
    print(eval_df.to_string(index=False))
    print()

    # Chosen K = 4
    chosen_k = 4
    km_4, labels_4 = kmeans_models[chosen_k]
    rfm["segment_id"] = labels_4

    # PCA (2 components)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_ratios = pca.explained_variance_ratio_
    print("================ PCA EXPLAINED VARIANCE ================")
    print(f"PC1 Variance Ratio: {var_ratios[0]:.3f} (Ref: ~0.634)")
    print(f"PC2 Variance Ratio: {var_ratios[1]:.3f} (Ref: ~0.218)")
    print(f"Total Explained Variance: {var_ratios.sum():.3f} (Ref: ~0.852)")
    print()

    # Profile segments
    profiles = rfm.groupby("segment_id").agg(
        size=("customer_id", "count"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean")
    ).reset_index()

    # Identify premium segment (highest mean monetary)
    max_monetary_idx = profiles["mean_monetary"].idxmax()
    premium_cluster_id = profiles.loc[max_monetary_idx, "segment_id"]

    # Assign business names and recommended actions
    # Segment characteristics for K=4:
    # 1. Highest monetary (~7130), moderate freq (~6.8) -> Tech/Occasional High-Spenders (Premium)
    # 2. Highest frequency (~19.4), lowest recency (~8.8 days) -> Daily Grocery Loyalists
    # 3. High recency (long inactivity), low freq, low monetary -> Inactive / At-Risk
    # 4. Moderate recency, moderate freq, moderate monetary -> Steady Regulars
    segment_names = {}
    recommended_actions = {}

    for _, row in profiles.iterrows():
        sid = int(row["segment_id"])
        m = row["mean_monetary"]
        f = row["mean_frequency"]
        r = row["mean_recency"]

        if sid == premium_cluster_id:
            name = "High-Value Premium Shoppers"
            action = "Exclusive concierge support, early access to new tech gadgets, and VIP delivery slots."
        elif f > 15:
            name = "Daily Grocery Loyalists"
            action = "Subscription loyalty perks, free delivery passes, and weekly replenishment baskets."
        elif r > 60:
            name = "Lapsed / At-Risk Occasionals"
            action = "Win-back discount vouchers, category re-discovery campaigns, and SMS reminders."
        else:
            name = "Steady Moderate Shoppers"
            action = "Personalized category bundle cross-sells to increase basket size and order frequency."

        segment_names[sid] = name
        recommended_actions[sid] = action

    rfm["segment_name"] = rfm["segment_id"].map(segment_names)
    rfm["is_premium"] = rfm["segment_id"] == premium_cluster_id

    profiles["segment_name"] = profiles["segment_id"].map(segment_names)
    profiles["is_premium"] = profiles["segment_id"] == premium_cluster_id
    profiles["recommended_action"] = profiles["segment_id"].map(recommended_actions)

    print("================ SEGMENT PROFILES (K=4) ================")
    print(profiles.to_string(index=False))
    print()

    # Save to outputs/customer_segments.csv
    out_cols = ["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]
    out_path = os.path.join(OUTPUTS_DIR, "customer_segments.csv")
    rfm[out_cols].to_csv(out_path, index=False)
    print(f"Saved customer segments to {out_path} ({len(rfm)} rows)")

    return rfm, profiles, eval_df, var_ratios

if __name__ == "__main__":
    rfm = compute_rfm()
    cluster_rfm(rfm)
