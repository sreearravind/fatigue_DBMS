import pandas as pd
from ml.torch_summary_engine import TorchSummaryEngine

# 1) Load the processed CSV
df = pd.read_csv("processed/task1_master_processed.csv")

# 2) Inspect columns and route IDs
print("Columns:", df.columns.tolist())
print("Unique route_ids:", df["route_id"].unique())

# 3) Pick one of the actual route IDs from the print above
test_route = df["route_id"].unique()[0]  # or manually set e.g. "ECAP90"

print(f"\nTesting TorchSummaryEngine on route: {test_route}")

# 4) Run the summary engine
engine = TorchSummaryEngine()
summary = engine.generate_summary(test_route, df)

print("\nSummary output:\n")
print(summary)