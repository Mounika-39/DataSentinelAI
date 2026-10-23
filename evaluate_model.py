
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

# Reproducible synthetic dataset
rng = np.random.default_rng(42)

n_normal = 500
n_anomalies = 25

normal = pd.DataFrame({
    "quantity": rng.integers(1, 10, n_normal),
    "unit_price": rng.normal(100, 15, n_normal).clip(1)
})

anomalies = pd.DataFrame({
    "quantity": rng.integers(1, 10, n_anomalies),
    "unit_price": rng.uniform(250, 500, n_anomalies)
})

df = pd.concat(
    [normal, anomalies],
    ignore_index=True
)

# Ground truth: 0 = normal, 1 = injected anomaly
y_true = np.array(
    [0] * n_normal + [1] * n_anomalies
)

# Fit Isolation Forest
model = IsolationForest(
    contamination=n_anomalies / len(df),
    random_state=42
)

predictions = model.fit_predict(
    df[["quantity", "unit_price"]]
)

# Convert model output: -1 = anomaly, 1 = normal
y_pred = (predictions == -1).astype(int)

print("=== DATASET ===")
print("Total records:", len(df))
print("Known anomalies:", int(y_true.sum()))

print("\n=== CLASSIFICATION REPORT ===")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Normal", "Anomaly"],
        zero_division=0
    )
)

print("=== CONFUSION MATRIX ===")
print(confusion_matrix(y_true, y_pred))

df["actual_anomaly"] = y_true
df["predicted_anomaly"] = y_pred

df.to_csv(
    "reports/model_evaluation.csv",
    index=False
)

print("\nResults saved to reports/model_evaluation.csv")
