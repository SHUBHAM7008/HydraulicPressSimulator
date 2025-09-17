# save as train_per_cycle.py and run
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# ---------- CONFIG ----------
CSV_PATH = r"D:\Danfoss\hydraulic_press_realtime_dataset_small.csv"  # change if needed
SAVE_DIR = r"D:\Danfoss\models"  # where to save trained models
os.makedirs(SAVE_DIR, exist_ok=True)

# thresholds / assumptions
DEFAULT_CYCLE_TIME_S = 1.0  # if no cycle time info, assume 1 second per row
ENERGY_TIME_UNIT_HOURS = 1/3600.0  # to convert seconds -> hours for kWh calculation

# ---------- 1. Load ----------
df = pd.read_csv(CSV_PATH)
# Quick look (uncomment to inspect)
# print(df.head()); print(df.columns)

# ---------- 2. Create per-cycle records ----------
# Case A: data already has `cycle_id` (one id per cycle -> many rows per cycle)
if "cycle_id" in df.columns:
    group = df.groupby("cycle_id")
    per_cycle = group.agg({
        "pressure_bar": "mean",
        "flow_lpm": "mean",
        "speed_mm_s": "mean",
        "motor_power_kw": "mean",
        "temperature_C": "max",     # or mean, choose what matters
        "vibration_rms": "max",
        "efficiency": "mean",
        # If you have timestamps inside cycle, compute cycle_time separately below
    }).reset_index()

    # If timestamps exist, compute duration per cycle
    if "timestamp" in df.columns:
        # ensure timestamp dtype
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cycle_time = group["timestamp"].agg(lambda x: (x.max()-x.min()).total_seconds())
        per_cycle = per_cycle.merge(cycle_time.rename("cycle_time_s"), left_on="cycle_id", right_index=True).reset_index(drop=True)
    else:
        per_cycle["cycle_time_s"] = DEFAULT_CYCLE_TIME_S

# Case B: data has `cycle_time_s` or `cycle_duration_s` and already one row per cycle
elif "cycle_time_s" in df.columns or "cycle_duration_s" in df.columns:
    col = "cycle_time_s" if "cycle_time_s" in df.columns else "cycle_duration_s"
    per_cycle = df.copy()
    per_cycle = per_cycle.rename(columns={col: "cycle_time_s"})

# Case C: no cycle_id and no cycle_time -> assume each row is already one cycle
else:
    per_cycle = df.copy()
    per_cycle["cycle_time_s"] = DEFAULT_CYCLE_TIME_S

# ---------- 3. Compute energy per cycle ----------
# Energy (kWh) = motor_power_kw * (cycle_time_hours)
per_cycle["cycle_time_h"] = per_cycle["cycle_time_s"] * ENERGY_TIME_UNIT_HOURS
per_cycle["energy_kwh"] = per_cycle["motor_power_kw"] * per_cycle["cycle_time_h"]

# ---------- 4. Feature engineering ----------
# Use the main control variables as features; you can add more aggregated features if helpful
feature_cols = ["pressure_bar", "flow_lpm", "speed_mm_s", "cycle_time_s"]
target_energy = "energy_kwh"
target_eff = "efficiency"
target_temp = "temperature_C"
target_vib = "vibration_rms"

# Keep only rows without NaNs in these columns
keep_cols = feature_cols + [target_energy, target_eff, target_temp, target_vib]
per_cycle = per_cycle.dropna(subset=keep_cols).reset_index(drop=True)

X = per_cycle[feature_cols]
y_energy = per_cycle[target_energy]
y_eff = per_cycle[target_eff]
y_temp = per_cycle[target_temp]
y_vib = per_cycle[target_vib]

# ---------- 5. Split data (split once for all targets to keep alignment) ----------
X_train, X_test, y_energy_train, y_energy_test, y_eff_train, y_eff_test, y_temp_train, y_temp_test, y_vib_train, y_vib_test = train_test_split(
    X, y_energy, y_eff, y_temp, y_vib, test_size=0.2, random_state=42
)

# ---------- 6. Train models (one regressor per target) ----------
def train_and_report(X_tr, X_te, y_tr, y_te, name, save_path=None):
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    mse = mean_squared_error(y_te, preds)
    r2 = r2_score(y_te, preds)
    print(f"{name} -> MSE: {mse:.4f}, R2: {r2:.4f}")
    if save_path:
        joblib.dump(model, save_path)
        print(f"Saved {name} model -> {save_path}")
    return model

energy_model = train_and_report(X_train, X_test, y_energy_train, y_energy_test,
                                "energy_kwh", os.path.join(SAVE_DIR, "energy_model.joblib"))

eff_model = train_and_report(X_train, X_test, y_eff_train, y_eff_test,
                             "efficiency", os.path.join(SAVE_DIR, "eff_model.joblib"))

temp_model = train_and_report(X_train, X_test, y_temp_train, y_temp_test,
                              "temperature_C", os.path.join(SAVE_DIR, "temp_model.joblib"))

vib_model = train_and_report(X_train, X_test, y_vib_train, y_vib_test,
                             "vibration_rms", os.path.join(SAVE_DIR, "vib_model.joblib"))

# ---------- 7. Example: how to call predict (use DataFrame to avoid sklearn name warning) ----------
import pandas as pd
example = pd.DataFrame([[100, 50, 20, 2.0]], columns=feature_cols)  # sample candidate [pressure,flow,speed,cycle_time_s]
print("example features:\n", example)
print("pred energy (kWh):", energy_model.predict(example)[0])
print("pred eff:", eff_model.predict(example)[0])
print("pred temp:", temp_model.predict(example)[0])
print("pred vib:", vib_model.predict(example)[0])

# Done
print("Training complete. Models saved in:", SAVE_DIR)
