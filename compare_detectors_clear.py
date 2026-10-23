
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

rng = np.random.default_rng(42)

# Normal records
normal = pd.DataFrame({
    "quantity": rng.integers(1, 10, 1000),
    "unit_price": rng.normal(100, 15, 1000).clip(5),
})

# Clearly abnormal prices
anomalies = pd.DataFrame({
    "quantity": rng.integers(1, 10, 100),
    "unit_price": rng.uniform(250, 500, 100),
})

df = pd.concat([normal, anomalies], ignore_index=True)
y_true = np.array([0] * 1000 + [1] * 100)

# Isolation Forest
model = IsolationForest(
    contamination=100 / len(df),
    random_state=42,
)
if_pred = (model.fit_predict(df) == -1).astype(int)

# IQR rule
q1 = df["unit_price"].quantile(0.25)
q3 = df["unit_price"].quantile(0.75)
iqr = q3 - q1

iqr_pred = (
    (df["unit_price"] < q1 - 1.5 * iqr)
    | (df["unit_price"] > q3 + 1.5 * iqr)
).astype(int)

# Combined detector
combined_pred = ((if_pred == 1) | (iqr_pred == 1)).astype(int)


def evaluate(name, predictions):
    print(f"\n=== {name} ===")
    print("Precision:", round(precision_score(y_true, predictions, zero_division=0), 2))
    print("Recall:", round(recall_score(y_true, predictions, zero_division=0), 2))
    print("F1-score:", round(f1_score(y_true, predictions, zero_division=0), 2))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, predictions))


evaluate("Isolation Forest", if_pred)
evaluate("IQR price rule", iqr_pred)
evaluate("Combined detector", combined_pred)

df["actual_anomaly"] = y_true
df["isolation_forest"] = if_pred
df["iqr_rule"] = iqr_pred
df["combined"] = combined_pred

df.to_csv("reports/clear_detector_comparison.csv", index=False)
print("\nSaved reports/clear_detector_comparison.csv")
