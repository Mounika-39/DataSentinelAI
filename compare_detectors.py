
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

normal = pd.DataFrame({
    "quantity": rng.integers(1, 10, n_normal),
    "unit_price": rng.normal(100, 25, n_normal).clip(5)
})

anomalies = pd.DataFrame({
    "quantity": rng.integers(1, 10, n_anomalies),
    "unit_price": rng.uniform(125, 175, n_anomalies)
})

df = pd.concat([normal, anomalies], ignore_index=True)
y_true = np.array([0] * n_normal + [1] * n_anomalies)

features = ["quantity", "unit_price"]

# Method 1: Isolation Forest
model = IsolationForest(
    contamination=100 / 1100,
    random_state=42
)

if_predictions = (
    model.fit_predict(df[features]) == -1
).astype(int)

# Method 2: IQR on unit price
q1 = df["unit_price"].quantile(0.25)
q3 = df["unit_price"].quantile(0.75)
iqr = q3 - q1

iqr_predictions = (
    (df["unit_price"] < q1 - 1.5 * iqr)
    | (df["unit_price"] > q3 + 1.5 * iqr)
).astype(int)

# Method 3: Flag if either method detects an anomaly
combined_predictions = (
    (if_predictions == 1) | (iqr_predictions == 1)
).astype(int)


def evaluate(name, predictions):
    print(f"\n=== {name} ===")
    print("Precision:", round(
        precision_score(y_true, predictions, zero_division=0), 3
    ))
    print("Recall:", round(
        recall_score(y_true, predictions, zero_division=0), 3
    ))
    print("F1-score:", round(
        f1_score(y_true, predictions, zero_division=0), 3
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, predictions))


evaluate("Isolation Forest", if_predictions)
evaluate("IQR price rule", iqr_predictions)
evaluate("Combined detector", combined_predictions)

df["actual_anomaly"] = y_true
df["isolation_forest"] = if_predictions
df["iqr_rule"] = iqr_predictions
df["combined_detector"] = combined_predictions

df.to_csv("reports/detector_comparison.csv", index=False)
print("\nSaved reports/detector_comparison.csv")
