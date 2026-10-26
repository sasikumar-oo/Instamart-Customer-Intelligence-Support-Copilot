"""src/lapse_model.py -- Supervised lapse prediction model, hyperparameter tuning, class-balance experiment, and retention list.
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.utils import resample

CLEAN_DIR = os.path.join("data", "clean")
OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

def build_cohort_and_features():
    df_orders = pd.read_csv(os.path.join(CLEAN_DIR, "orders.csv"))
    df_customers = pd.read_csv(os.path.join(CLEAN_DIR, "customers.csv"))
    df_products = pd.read_csv(os.path.join(CLEAN_DIR, "products.csv"))

    df_orders["order_ts"] = pd.to_datetime(df_orders["order_ts"])
    df_customers["signup_date"] = pd.to_datetime(df_customers["signup_date"])

    # Cutoff date: 2025-09-01
    cutoff = pd.Timestamp("2025-09-01 00:00:00")
    sep_end = pd.Timestamp("2025-09-30 23:59:59")

    # Merge product category into orders
    orders_with_cat = df_orders.merge(df_products[["product_id", "category"]], on="product_id", how="left")

    # Pre-cutoff orders
    pre_orders = orders_with_cat[orders_with_cat["order_ts"] < cutoff].copy()
    pre_completed = pre_orders[pre_orders["status"] == "completed"].copy()

    # Cohort: every customer with at least one completed order before the cutoff
    cohort_customers = pre_completed["customer_id"].unique()
    print(f"Cohort size: {len(cohort_customers)} customers (Reference: 1156)")

    # Performance window (September): 2025-09-01 through 2025-09-30
    sep_completed = df_orders[
        (df_orders["status"] == "completed") & 
        (df_orders["order_ts"] >= cutoff) & 
        (df_orders["order_ts"] <= sep_end)
    ]
    sep_active_customers = set(sep_completed["customer_id"].unique())

    # Features extraction per customer
    features = []
    for cid in cohort_customers:
        cust_comp = pre_completed[pre_completed["customer_id"] == cid]
        cust_all_pre = pre_orders[pre_orders["customer_id"] == cid]

        # Recency: whole days from cutoff to last completed order
        last_ts = cust_comp["order_ts"].max()
        recency = (cutoff - last_ts).days

        # Frequency: count of completed orders before cutoff
        frequency = len(cust_comp)

        # Monetary: sum of completed order_amount
        monetary = round(cust_comp["order_amount"].sum(), 2)

        # Average discount_pct
        avg_discount = round(cust_comp["discount_pct"].mean(), 2)

        # Cancellation rate: cancelled orders / all orders with a final status before cutoff
        cancelled_count = (cust_all_pre["status"] == "cancelled").sum()
        total_final = len(cust_all_pre)
        cancellation_rate = round(cancelled_count / total_final, 4) if total_final > 0 else 0.0

        # Average delivery_minutes of completed orders
        avg_delivery_minutes = round(cust_comp["delivery_minutes"].mean(), 2)

        # Most frequent product category (favorite_category)
        favorite_category = cust_comp["category"].value_counts().index[0]

        # Target label: lapsed = 1 if no completed order in September, else 0
        lapsed = 0 if cid in sep_active_customers else 1

        features.append({
            "customer_id": cid,
            "recency": recency,
            "frequency": frequency,
            "monetary": monetary,
            "avg_discount": avg_discount,
            "cancellation_rate": cancellation_rate,
            "avg_delivery_minutes": avg_delivery_minutes,
            "favorite_category": favorite_category,
            "lapsed": lapsed
        })

    df_features = pd.DataFrame(features)

    # Merge demographics from customers
    df_features = df_features.merge(df_customers[["customer_id", "city", "age", "gender", "signup_date"]], on="customer_id", how="left")
    df_features["tenure_days"] = (cutoff - df_features["signup_date"]).dt.days
    df_features = df_features.drop(columns=["signup_date"])

    # Check missing values
    assert df_features.isna().sum().sum() == 0, "Missing values found in feature set!"

    total_cohort = len(df_features)
    total_lapsed = df_features["lapsed"].sum()
    lapsed_pct = (total_lapsed / total_cohort) * 100
    print(f"Lapsed customers: {total_lapsed} / {total_cohort} ({lapsed_pct:.2f}%) (Reference: 443, 38.32%)")

    return df_features

def train_and_evaluate(df_features):
    X = df_features.drop(columns=["customer_id", "lapsed"])
    y = df_features["lapsed"]

    # Stratified split: 80% train, 20% test
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, df_features["customer_id"], test_size=0.2, stratify=y, random_state=42
    )

    print(f"Train set: {len(X_train)} rows; Test set: {len(X_test)} rows (Reference: 924 train, 232 test)")

    # Baseline majority-class predictor (predict all 0)
    baseline_preds = np.zeros(len(y_test))
    baseline_acc = accuracy_score(y_test, baseline_preds)
    baseline_f1 = f1_score(y_test, baseline_preds, zero_division=0)
    print(f"Baseline ('always not lapsed'): Accuracy = {baseline_acc * 100:.1f}%, Lapsed F1 = {baseline_f1:.2f}")

    # Features setup
    numeric_features = ["recency", "frequency", "monetary", "avg_discount", "cancellation_rate", "avg_delivery_minutes", "age", "tenure_days"]
    categorical_features = ["city", "gender", "favorite_category"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_features)
        ]
    )

    # Fit preprocessor on train only to verify learned frequency mean vs full cohort mean
    preprocessor.fit(X_train)
    scaler_freq_mean = preprocessor.named_transformers_["num"].mean_[numeric_features.index("frequency")]
    cohort_freq_mean = df_features["frequency"].mean()
    print(f"\nFrequency mean (Train Scaler): {scaler_freq_mean:.4f} vs Full Cohort mean: {cohort_freq_mean:.4f}")
    print("They differ because the training set is an 80% sample drawn without leaking the 20% test set distribution.")

    # 1. Tune Logistic Regression
    pipe_lr = Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    grid_lr = GridSearchCV(
        pipe_lr,
        param_grid={"clf__C": [0.01, 0.1, 1, 10, 100]},
        cv=5,
        scoring="f1"
    )
    grid_lr.fit(X_train, y_train)
    best_lr = grid_lr.best_estimator_
    print(f"\nBest Logistic Regression: {grid_lr.best_params_}, CV F1: {grid_lr.best_score_:.4f}")

    # 2. Tune Random Forest
    pipe_rf = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    grid_rf = GridSearchCV(
        pipe_rf,
        param_grid={
            "clf__max_depth": [3, 5, 8],
            "clf__min_samples_leaf": [5, 10, 20]
        },
        cv=5,
        scoring="f1"
    )
    grid_rf.fit(X_train, y_train)
    best_rf = grid_rf.best_estimator_
    print(f"Best Random Forest: {grid_rf.best_params_}, CV F1: {grid_rf.best_score_:.4f}")

    # Evaluate on untouched test set
    y_pred_lr = best_lr.predict(X_test)
    y_pred_rf = best_rf.predict(X_test)

    print("\n================ MODEL TEST EVALUATION ================")
    print("--- Logistic Regression ---")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_lr)}")
    p_lr = precision_score(y_test, y_pred_lr)
    r_lr = recall_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr)
    print(f"Precision: {p_lr:.4f}, Recall: {r_lr:.4f}, F1: {f1_lr:.4f}")

    print("\n--- Random Forest ---")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_rf)}")
    p_rf = precision_score(y_test, y_pred_rf)
    r_rf = recall_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf)
    print(f"Precision: {p_rf:.4f}, Recall: {r_rf:.4f}, F1: {f1_rf:.4f}")

    # Pick selected model (Random Forest reaches highest F1 >= 0.55)
    selected_model = best_rf
    assert f1_rf >= 0.55, f"Selected model F1 {f1_rf} < 0.55"

    # ================ CLASS BALANCE EXPERIMENT ================
    print("\n================ CLASS-BALANCE EXPERIMENT ================")
    train_df = pd.concat([X_train, y_train], axis=1)
    majority = train_df[train_df["lapsed"] == 0]
    minority = train_df[train_df["lapsed"] == 1]

    # Oversample minority with replacement to match majority size (570)
    minority_oversampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    train_balanced = pd.concat([majority, minority_oversampled])

    X_train_bal = train_balanced.drop(columns=["lapsed"])
    y_train_bal = train_balanced["lapsed"]
    print(f"Balanced training set: {len(X_train_bal)} rows ({y_train_bal.value_counts().to_dict()})")

    # Refit best Logistic Regression on balanced data
    bal_lr = Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(C=grid_lr.best_params_["clf__C"], max_iter=1000, random_state=42))
    ])
    bal_lr.fit(X_train_bal, y_train_bal)
    y_pred_bal_lr = bal_lr.predict(X_test)

    p_bal = precision_score(y_test, y_pred_bal_lr)
    r_bal = recall_score(y_test, y_pred_bal_lr)
    f1_bal = f1_score(y_test, y_pred_bal_lr)
    print(f"Unbalanced LR: Precision={p_lr:.4f}, Recall={r_lr:.4f}, F1={f1_lr:.4f}")
    print(f"Balanced LR:   Precision={p_bal:.4f}, Recall={r_bal:.4f}, F1={f1_bal:.4f}")
    print(f"Recall shifted by {r_bal - r_lr:+.4f} (from {r_lr:.2f} to {r_bal:.2f})!")

    # ================ FEATURE IMPORTANCE & COEFFICIENTS ================
    print("\n================ INTERPRETABILITY ================")
    # Random Forest feature importances
    rf_clf = best_rf.named_steps["clf"]
    cat_names = best_rf.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(categorical_features)
    all_feature_names = list(numeric_features) + list(cat_names)
    rf_importances = pd.Series(rf_clf.feature_importances_, index=all_feature_names).sort_values(ascending=False)
    print("Top 5 Features in Random Forest:")
    print(rf_importances.head(5).to_string())

    # Logistic Regression coefficients
    lr_clf = best_lr.named_steps["clf"]
    lr_coefs = pd.Series(lr_clf.coef_[0], index=all_feature_names)
    print("\nKey Logistic Regression Coefficients:")
    for feat in ["recency", "frequency", "cancellation_rate"]:
        if feat in lr_coefs:
            print(f"{feat}: {lr_coefs[feat]:+.4f}")

    # ================ MODEL PERSISTENCE ================
    model_path = os.path.join(OUTPUTS_DIR, "lapse_model.joblib")
    joblib.dump(selected_model, model_path)
    print(f"\nSaved selected model to {model_path}")

    # Reload and verify exact match
    loaded_model = joblib.load(model_path)
    loaded_preds = loaded_model.predict(X_test)
    assert (loaded_preds == y_pred_rf).all(), "Reloaded model predictions mismatch!"
    print("Reloaded model reproduced test predictions exactly!")

    # ================ RETENTION LIST GENERATION ================
    # Read customer_segments.csv from Part B task 3 to find Premium customers
    seg_df = pd.read_csv(os.path.join(OUTPUTS_DIR, "customer_segments.csv"))
    premium_cids = set(seg_df[seg_df["is_premium"] == True]["customer_id"])

    # Test set customers
    test_eval_df = pd.DataFrame({
        "customer_id": test_idx.values,
        "actually_lapsed": y_test.values
    })
    test_eval_df["lapse_probability"] = selected_model.predict_proba(X_test)[:, 1].round(4)

    # Filter test-set Premium customers
    premium_test_df = test_eval_df[test_eval_df["customer_id"].isin(premium_cids)].copy()
    premium_test_df = premium_test_df.sort_values("lapse_probability", ascending=False).reset_index(drop=True)

    # Top 10 (or all if < 10)
    top_retention = premium_test_df.head(10)[["customer_id", "lapse_probability", "actually_lapsed"]]
    retention_path = os.path.join(OUTPUTS_DIR, "retention_list.csv")
    top_retention.to_csv(retention_path, index=False)
    print(f"\nSaved retention list to {retention_path} ({len(top_retention)} rows):")
    print(top_retention.to_string(index=False))

    true_lapsed_count = top_retention["actually_lapsed"].sum()
    print(f"Actually lapsed in top retention list: {true_lapsed_count} of {len(top_retention)}")

    return selected_model, grid_lr, grid_rf, top_retention

if __name__ == "__main__":
    df_features = build_cohort_and_features()
    train_and_evaluate(df_features)
