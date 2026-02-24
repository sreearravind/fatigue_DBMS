import numpy as np
import pandas as pd
# Update the path to your actual file
INPUT_CSV = r"E:\Python project_2026\Phase 4_SQL_Delivery\specimen_master_processed.csv"

df_real = pd.read_csv(INPUT_CSV)

print("Real data loaded.")
print("Shape:", df_real.shape)
print("Columns:", df_real.columns.tolist())
print("\nRoute counts in real data:")
print(df_real["route_id"].value_counts())

# 3 basic routes
route_target_counts = {
    "T5": 50,
    "T6A": 50,
    "T6W": 50,
    "ECAP90": 50,
    "ECAP120": 50,
}

# Add CT2, CT4, ..., CT24 → 12 routes
for h in range(2, 26, 2):  # 2,4,6,...,24
    route_name = f"CT{h}"
    route_target_counts[route_name] = 50

print("Target routes and sample counts:")
for r, n in route_target_counts.items():
    print(f"{r}: {n}")

# Columns where we apply small multiplicative noise
NOISE_COLS = [
    "grain_size_um",
    "Hardness_hv",
    "YS_MPa",
    "UTS_MPa",
    "PSA_mean",
    "mean_stress_mean",
]

# Noise level: 3% (you can tune to 2% or 5% later)
NOISE_STD = 0.03   # 3%

def generate_synthetic_for_route(df, route_id, n_samples,
                                 noise_cols=NOISE_COLS,
                                 noise_std=NOISE_STD,
                                 random_state=None):
    """
    df         : real specimen dataframe
    route_id   : e.g. 'T5', 'T6A', 'CT2', 'ECAP90'
    n_samples  : e.g. 50
    noise_cols : list of numeric columns to perturb
    noise_std  : relative std-dev for multiplicative noise (e.g. 0.03 = ±3%)
    """

    df_route = df[df["route_id"] == route_id].copy()

    if df_route.empty:
        print(f"⚠ WARNING: No real rows found for route_id = {route_id}. Skipping.")
        return pd.DataFrame()

    # If real data is fewer than n_samples, sample with replacement
    df_sampled = df_route.sample(
        n=n_samples,
        replace=True,            # repeat allowed
        random_state=random_state
    ).reset_index(drop=True)

    # Apply noise to each chosen numeric column
    for col in noise_cols:
        if col in df_sampled.columns:
            noise = 1.0 + noise_std * np.random.randn(len(df_sampled))
            df_sampled[col] = df_sampled[col] * noise
        else:
            print(f"  (info) Column '{col}' not found for route '{route_id}', skipping noise.")

    # Optionally recompute derived columns here (d_inv_sqrt, strength_ratio, etc.)
    if "grain_size_um" in df_sampled.columns:
        df_sampled["d_inv_sqrt"] = 1 / np.sqrt(df_sampled["grain_size_um"])

    if {"YS_MPa", "UTS_MPa"}.issubset(df_sampled.columns):
        df_sampled["strength_ratio"] = df_sampled["YS_MPa"] / df_sampled["UTS_MPa"]

    if {"log_Nf", "YS_MPa"}.issubset(df_sampled.columns):
        # If you already have logNf from real data
        df_sampled["fatigue_efficiency"] = df_sampled["log_Nf"] / df_sampled["YS_MPa"]

    return df_sampled

all_synth_list = []

# You can fix a base seed for reproducibility
BASE_RANDOM_STATE = 42

for i, (route, n_samp) in enumerate(route_target_counts.items()):
    print(f"\nGenerating synthetic samples for route: {route} (n = {n_samp})")

    rs = BASE_RANDOM_STATE + i  # different seed per route
    df_synth_route = generate_synthetic_for_route(
        df_real,
        route_id=route,
        n_samples=n_samp,
        random_state=rs
    )

    print(f"  -> Generated rows: {len(df_synth_route)}")

    all_synth_list.append(df_synth_route)

# Concatenate all routes
df_synth = pd.concat(all_synth_list, ignore_index=True)

print("\n=== Synthetic dataset summary ===")
print("Shape:", df_synth.shape)
print("\nRoute counts:")
print(df_synth["route_id"].value_counts())

print("\nSynthetic numeric summary:")
print(df_synth.describe())

# Optional: compare one route between real and synthetic
route_check = "ECAP90"
print(f"\nReal vs Synthetic quick compare for route: {route_check}")

real_route = df_real[df_real["route_id"] == route_check]
synth_route = df_synth[df_synth["route_id"] == route_check]

cols_check = ["grain_size_um", "Hardness_hv", "YS_MPa", "UTS_MPa"]

for col in cols_check:
    if col in real_route.columns and col in synth_route.columns:
        print(f"\nColumn: {col}")
        print("Real   mean:", real_route[col].mean())
        print("Synth  mean:", synth_route[col].mean())

OUTPUT_CSV = r"E:\Python project_2026\Phase 4_SQL_Delivery\task_synthetic_specimens_850.csv"

df_synth.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Synthetic dataset saved to:\n{OUTPUT_CSV}")