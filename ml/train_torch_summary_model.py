import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

DATA_PATH = Path("task1_master_processed.csv")
MODEL_DIR = Path("ml/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ------------- Simple MLP Model -------------
class SummaryMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)  # logits for classes
        )

    def forward(self, x):
        return self.net(x)

def main():
    df = pd.read_csv(DATA_PATH)

    # === 1) Construct features ===
    feature_cols_num = ["grain_size_um", "Hardness_hv", "YS_MPa", "UTS_MPa", "PSA_mean", "mean_stress_mean"]
    # simple route encoding
    route_dummies = pd.get_dummies(df["route_id"], prefix="route")
    X = pd.concat([df[feature_cols_num], route_dummies], axis=1)
    X = X.apply(pd.to_numeric, errors="coerce")
    # 2) Keep only numeric columns (this will drop any stubborn object columns)
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(0.0)
    # (optional, for one run only – to inspect)
    # print("Final X dtypes:\n", X.dtypes)
    # print("Any object columns? ->", any(X.dtypes == "object"))
    # Now convert to tensor

    X_tensor = torch.tensor(X.values, dtype=torch.float32)

    # === 2) Build labels: performance vs baseline ===
    baseline_route = "T6W"   # adjust name if different
    base_mean = df[df["route_id"] == baseline_route]["cycles_to_failure"].mean()

    ratios = df["cycles_to_failure"] / base_mean
    # 0 = low, 1 = medium, 2 = high
    y = np.where(ratios >= 1.5, 2,
        np.where(ratios >= 1.1, 1, 0)
    )

    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = SummaryMLP(in_dim=X.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # === 3) Train for a few epochs ===
    for epoch in range(50):
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d}: loss={running_loss/total:.4f}, "
                  f"acc={correct/total:.3f}")

    # === 4) Save model + feature metadata ===
    save_path = MODEL_DIR / "ml.torch_summary_engine.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_cols": list(X.columns),
        "baseline_route": baseline_route
    }, save_path)
    print(f"Saved model to {save_path}")
#sanity check - delete later
    print("X dtypes:\n", X.dtypes)
    print("Any object dtypes? ->", any(X.dtypes == "object"))
    print("X shape:", X.shape)

if __name__ == "__main__":
    main()