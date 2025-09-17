# TestModels_CycleLevel.py

import pandas as pd
import joblib
import numpy as np

# -------------------------------
# 1. Load Models & Features
# -------------------------------
clf_bin = joblib.load("binary_failure_model_cycle.pkl")
clf_multi = joblib.load("multi_failure_model_cycle.pkl")
reg_rul = joblib.load("rul_regression_model_cycle.pkl")
feature_cols_cycle = joblib.load("features_cycle.pkl")

# -------------------------------
# 2. Load New Dataset
# -------------------------------
df_new = pd.read_csv("hydraulic_press_dataset_uncleaned.csv")

# -------------------------------
# 3. Encode categorical columns
# -------------------------------
categorical_cols = ['phase', 'motion_type', 'anomaly_type']  # update based on your dataset
for col in categorical_cols:
    if col in df_new.columns:
        df_new[col] = pd.factorize(df_new[col])[0]

# -------------------------------
# 4. Aggregate per cycle
# -------------------------------
target_cols = ['binary_failure_T50','multi_failure_mode','RUL_cycles']

numeric_cols = [c for c in df_new.select_dtypes(include=np.number).columns 
                if c not in ['cycle_id'] + target_cols]

# Aggregate numeric features
df_cycle_new = df_new.groupby('cycle_id')[numeric_cols].agg(['mean','max','min','std']).reset_index()
df_cycle_new.columns = ['_'.join(col).strip() if type(col) is tuple else col for col in df_cycle_new.columns]

# Aggregate labels if present
labels = df_new.groupby('cycle_id')[target_cols].max().reset_index()
df_cycle_new = df_cycle_new.merge(labels, left_on='cycle_id_', right_on='cycle_id', how='left').drop(columns=['cycle_id_'])

# -------------------------------
# 5. Select features for prediction
# -------------------------------
X_cycle_new = df_cycle_new.reindex(columns=feature_cols_cycle, fill_value=0)

# -------------------------------
# 6. Choose cycle range to predict
# -------------------------------
# Example: predict cycles 10 to 20
start_cycle = 10
end_cycle = 20
X_range = X_cycle_new.iloc[start_cycle:end_cycle+1]  # iloc upper bound exclusive

# -------------------------------
# 7. Make predictions
# -------------------------------
# Binary failure
bin_pred = clf_bin.predict(X_range)
bin_prob = clf_bin.predict_proba(X_range)[:,1]  # probability of failure

# Multi-failure
multi_pred = clf_multi.predict(X_range)

# RUL
rul_pred = reg_rul.predict(X_range)

# -------------------------------
# 8. Print results in table
# -------------------------------
results = pd.DataFrame({
    'Cycle_ID': df_cycle_new['cycle_id'].iloc[start_cycle:end_cycle+1].values,
    'Binary_Failure': bin_pred,
    'Failure_Prob': np.round(bin_prob, 2),
    'Multi_Failure_Mode': multi_pred,
    'RUL_Predicted': np.round(rul_pred, 2)
})

print("\n🔹 Prediction Results (Cycle Level)")
print(results.to_string(index=False))
