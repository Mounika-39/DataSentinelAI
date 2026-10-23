
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

rng = np.random.default_rng(7)

n_normal = 1000
n_anomalies = 100

# Normal records have variation
normal = pd.DataFrame({
    "quantity": rng.integers(1, 10, n_normal),
    "unit_price": rng.normal(100, 25, n_normal).clip(5)
})

# Anomalies are unusual, but overlap with normal prices
anomalies = pd.DataFrame({
    "quantity": rng.integers(1, 10, n_anomalies),
    "unit_price": rng.uniform(125, 175, n_anomalies)
})

df = pd.concat([normal, anomalies], ignore_index=True)

y_true = np.array(
    [0] * n_normal + [1] * n_anomalies
)

features = ["quantity", "unit_price"]

model = IsolationForest(
    contamination=n_anomalies / len(df),
    random_state=42
)

predictions = model.fit_predict(df[features])
y_pred = (predictions == -1).astype(int)

print("=== ADVANCED EVALUATION ===")
print("Records:", len(df))
print("Known anomalies:", int(y_true.sum()))

print("\nPrecision:", round(precision_score(y_true, y_pred), 3))
print("Recall:", round(recall_score(y_true, y_pred), 3))
print("F1-score:", round(f1_score(y_true, y_pred), 3))

print("\nConfusion matrix:")
print(confusion_matrix(y_true, y_pred))

df["actual_anomaly"] = y_true
df["predicted_anomaly"] = y_pred

df.to_csv(
    "reports/advanced_model_evaluation.csv",
    index=False
)

print("\nSaved reports/advanced_model_evaluation.csv")
