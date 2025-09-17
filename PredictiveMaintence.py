# PredictiveMaintenance_CycleLevel.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import joblib

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("hydraulic_press_dataset_cleaned.csv")
print("Initial Data Types:\n", df.dtypes)
print("\nShape of dataset:", df.shape)

# -----------------------------
# 2. Aggregate per cycle
# -----------------------------
if "cycle_id" not in df.columns:
    raise ValueError("Dataset must have 'cycle_id' column for cycle-level aggregation.")

# Identify numeric features (exclude targets)
target_cols = ["binary_failure_T50", "multi_failure_mode", "RUL_cycles"]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in target_cols + ["cycle_id"]]

# Aggregate numeric features per cycle
df_cycle = df.groupby("cycle_id")[feature_cols].agg(['mean', 'max', 'min', 'std']).reset_index()
df_cycle.columns = ['_'.join(col).strip() if type(col) is tuple else col for col in df_cycle.columns]

# Aggregate labels per cycle
labels = df.groupby("cycle_id")[target_cols].max().reset_index()
df_cycle = df_cycle.merge(labels, left_on='cycle_id_', right_on='cycle_id').drop(columns=['cycle_id_'])

# Save feature columns for consistent testing
feature_cols_cycle = [c for c in df_cycle.columns if c not in target_cols + ['cycle_id']]
joblib.dump(feature_cols_cycle, "features_cycle.pkl")

# -----------------------------
# 3. Correlation Analysis
# -----------------------------
plt.figure(figsize=(12, 8))
sns.heatmap(df_cycle[feature_cols_cycle + target_cols].corr(), cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap (Cycle Level)")
plt.show()

# -----------------------------
# 4. Evaluation Functions
# -----------------------------
def evaluate_classification(y_test, y_pred, y_proba=None, task="Classification"):
    print(f"\n📌 {task} Metrics:")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.3f}")
    print(f"Recall   : {recall_score(y_test, y_pred, average='weighted'):.3f}")
    print(f"F1-Score : {f1_score(y_test, y_pred, average='weighted'):.3f}")
    if y_proba is not None and len(np.unique(y_test)) == 2:
        print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba[:,1]):.3f}")

def evaluate_regression(y_test, y_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("\n📌 Regression Metrics (RUL):")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAE  : {mae:.2f}")
    print(f"R²   : {r2:.3f}")

# -----------------------------
# 5. Binary Classification (per cycle)
# -----------------------------
if "binary_failure_T50" in df_cycle.columns:
    X = df_cycle[feature_cols_cycle]
    y = df_cycle["binary_failure_T50"]

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # stratify removed
)
    clf_bin = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_bin.fit(X_train, y_train)

    y_pred_bin = clf_bin.predict(X_test)
    y_proba_bin = clf_bin.predict_proba(X_test)

    evaluate_classification(y_test, y_pred_bin, y_proba_bin, task="Binary Classification")
    joblib.dump(clf_bin, "binary_failure_model_cycle.pkl")

# -----------------------------
# 6. Multi-Class Classification (per cycle)
# -----------------------------
if "multi_failure_mode" in df_cycle.columns:
    X = df_cycle[feature_cols_cycle]
    y = df_cycle["multi_failure_mode"]

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # stratify removed
)

    clf_multi = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_multi.fit(X_train, y_train)

    y_pred_multi = clf_multi.predict(X_test)
    evaluate_classification(y_test, y_pred_multi, task="Multi-Class Classification")
    joblib.dump(clf_multi, "multi_failure_model_cycle.pkl")

# -----------------------------
# 7. Regression (RUL per cycle)
# -----------------------------
if "RUL_cycles" in df_cycle.columns:
    X = df_cycle[feature_cols_cycle]
    y = df_cycle["RUL_cycles"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    reg_rul = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_rul.fit(X_train, y_train)

    y_pred_rul = reg_rul.predict(X_test)
    evaluate_regression(y_test, y_pred_rul)

    joblib.dump(reg_rul, "rul_regression_model_cycle.pkl")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_rul, alpha=0.6)
    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.title("RUL Prediction: Actual vs Predicted (Cycle Level)")
    plt.show()
