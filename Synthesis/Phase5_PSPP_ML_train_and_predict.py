import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# ======= FILE PATHS (adjust as needed) =======
REAL_DATA_CSV = r"E:\Python project_2026\Phase 4_SQL_Delivery\specimen_master_processed.csv"
SYNTHETIC_CSV = r"E:\Python project_2026\Phase 4_SQL_Delivery\task_synthetic_specimens_850.csv"
OUTPUT_CSV = r"E:\Python project_2026\Phase 4_SQL_Delivery\task_synthetic_specimens_850_with_predictions.csv"

print("Loading real specimen data...")
df_real = pd.read_csv(REAL_DATA_CSV)
print("Real data shape:", df_real.shape)

print("\nLoading synthetic specimen data...")
df_synth = pd.read_csv(SYNTHETIC_CSV)
print("Synthetic data shape:", df_synth.shape)

def ensure_derived_features(df):
    # d_inv_sqrt = 1 / sqrt(grain_size)
    if "grain_size_um" in df.columns:
        df["d_inv_sqrt"] = 1 / np.sqrt(df["grain_size_um"])

    # strength_ratio = YS / UTS
    if {"YS_MPa", "UTS_MPa"}.issubset(df.columns):
        df["strength_ratio"] = df["YS_MPa"] / df["UTS_MPa"]

    # fatigue_efficiency = log_Nf / YS_MPa (if log_Nf present)
    if {"log_Nf", "YS_MPa"}.issubset(df.columns):
        df["fatigue_efficiency"] = df["log_Nf"] / df["YS_MPa"]

    return df

df_real = ensure_derived_features(df_real)
df_synth = ensure_derived_features(df_synth)

# Target: log10 fatigue life
TARGET_COL = "log_Nf"

# PSPP-aware feature set
FEATURE_COLS = [
    "d_inv_sqrt",       # Structure (Hall-Petch)
    "Hardness_hv",      # Property
    "YS_MPa",           # Property
    "UTS_MPa",          # Property
    "strength_ratio",   # Derived property
    "PSA_mean",         # Fatigue loading
    "mean_stress_mean",          # Mean stress effect
]

print("\nChecking availability of features in real data:")
print([col for col in FEATURE_COLS if col in df_real.columns])

# Keep only rows where target and all features are present
cols_needed = FEATURE_COLS + [TARGET_COL]
df_real_clean = df_real.dropna(subset=cols_needed).copy()

print("\nCleaned real data shape:", df_real_clean.shape)

X_real = df_real_clean[FEATURE_COLS].values
y_real = df_real_clean[TARGET_COL].values

# ====== Model A: PSPP Linear Regression (baseline) ======
lin_model = Pipeline([
    ("scaler", StandardScaler()),
    ("linreg", LinearRegression())
])

# ====== Model B: PSPP Random Forest (complex model) ======
rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
# (RF doesn't need scaling, so no pipeline needed, but you could wrap it if you like)

print("\n=== Training PSPP Linear Regression model ===")
lin_model.fit(X_real, y_real)

y_pred_lin = lin_model.predict(X_real)
r2_lin = r2_score(y_real, y_pred_lin)
rmse_lin = np.sqrt(mean_squared_error(y_real, y_pred_lin))

print(f"Linear model R² (train): {r2_lin:.4f}")
print(f"Linear model RMSE (train): {rmse_lin:.4f}")

# Optional: 5-fold cross-validation for R²
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_lin = cross_val_score(lin_model, X_real, y_real, cv=kf, scoring="r2")
print("Linear model R² (5-fold CV): "
      f"mean={cv_scores_lin.mean():.4f}, std={cv_scores_lin.std():.4f}")

print("\n=== Training PSPP Random Forest model ===")
rf_model.fit(X_real, y_real)

y_pred_rf = rf_model.predict(X_real)
r2_rf = r2_score(y_real, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_real, y_pred_rf))

print(f"Random Forest R² (train): {r2_rf:.4f}")
print(f"Random Forest RMSE (train): {rmse_rf:.4f}")

cv_scores_rf = cross_val_score(rf_model, X_real, y_real, cv=kf, scoring="r2")
print("Random Forest R² (5-fold CV): "
      f"mean={cv_scores_rf.mean():.4f}, std={cv_scores_rf.std():.4f}")

print("\nRandom Forest feature importances:")
for col, imp in sorted(zip(FEATURE_COLS, rf_model.feature_importances_),
                       key=lambda x: x[1], reverse=True):
    print(f"{col:15s}: {imp:.4f}")

print("\nPreparing synthetic data for prediction...")

# Ensure required columns present
missing_in_synth = [c for c in FEATURE_COLS if c not in df_synth.columns]
if missing_in_synth:
    print("⚠ WARNING: synthetic data is missing columns:", missing_in_synth)

df_synth_pred = df_synth.copy()
X_synth = df_synth_pred[FEATURE_COLS].values

# Linear model predictions
log_Nf_lin_pred = lin_model.predict(X_synth)
Nf_lin_pred = np.power(10.0, log_Nf_lin_pred)

# Random Forest predictions
log_Nf_rf_pred = rf_model.predict(X_synth)
Nf_rf_pred = np.power(10.0, log_Nf_rf_pred)

df_synth_pred["log_Nf_lin_pred"] = log_Nf_lin_pred
df_synth_pred["Nf_lin_pred"] = Nf_lin_pred

df_synth_pred["log_Nf_rf_pred"] = log_Nf_rf_pred
df_synth_pred["Nf_rf_pred"] = Nf_rf_pred

print("\n=== Predicted fatigue summary by route (Random Forest) ===")
group_cols = ["route_id"]

summary_rf = (
    df_synth_pred
    .groupby(group_cols)["Nf_rf_pred"]
    .agg(["count", "mean", "min", "max"])
    .reset_index()
)

print(summary_rf)

df_synth_pred.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Synthetic dataset with predictions saved to:\n{OUTPUT_CSV}")