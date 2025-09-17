import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv(r"D:\Danfoss\hydraulic_press_realtime_dataset_small.csv")

# Aggregate per cycle
agg_funcs = {
    "pressure_bar":["mean","max","std"],
    "flow_lpm":["mean","max","std"],
    "speed_mm_s":["mean","max","std"],
    "motor_power_kw":["mean","max","std"],
    "actuator_power_kw":["mean","max","std"],
    "efficiency":["mean","min","std"],
    "temperature_C":["mean","max","std"],
    "vibration_rms":["mean","max","std"],
    "energy_kJ_incremental":["sum"],
    "is_anomaly":["max"]
}
per_cycle = df.groupby("cycle_id").agg(agg_funcs)
per_cycle.columns = ["_".join(c) for c in per_cycle.columns]
per_cycle = per_cycle.reset_index()

# Target = efficiency_mean of NEXT cycle
per_cycle = per_cycle.sort_values("cycle_id").reset_index(drop=True)
per_cycle["next_cycle_efficiency"] = per_cycle["efficiency_mean"].shift(-1).fillna(method="ffill")
per_cycle = per_cycle.iloc[:-1, :]  # drop last

# -----------------------------
# 2. Features + Target
# -----------------------------
feature_cols = [c for c in per_cycle.columns if c not in ["cycle_id","next_cycle_efficiency"]]
X = per_cycle[feature_cols].fillna(0.0)
y = per_cycle["next_cycle_efficiency"]

# Train-test split (time-aware)
n = len(per_cycle)
split_idx = int(n*0.7)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# -----------------------------
# 3. Train regression model
# -----------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor(n_estimators=200, random_state=42))
])
pipeline.fit(X_train, y_train)

# -----------------------------
# 4. Evaluate
# -----------------------------
y_pred = pipeline.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))



plt.figure(figsize=(12,6))
plt.plot(y_test.values, label="Actual Efficiency", marker='o')
plt.plot(y_pred, label="Predicted Efficiency", marker='x')
plt.xlabel("Cycle Index")
plt.ylabel("Efficiency (%)")
plt.title("Actual vs Predicted Next Cycle Efficiency")
plt.legend()
plt.tight_layout()
plt.show()


# -----------------------------
# 5. Save model
# -----------------------------
joblib.dump({"model": pipeline, "feature_cols": feature_cols}, "next_cycle_efficiency_rf.pkl")
print(" Model saved as next_cycle_efficiency_rf.pkl")
