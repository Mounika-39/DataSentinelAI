
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(df):
    results = df.copy()

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    results["anomaly_score"] = float("nan")
    results["is_anomaly"] = False

    if not numeric_columns or len(df) < 5:
        return results

    X = df[numeric_columns].copy()

    # Remove columns that contain no usable values
    X = X.dropna(axis=1, how="all")

    if X.empty:
        return results

    # Fill missing model inputs without modifying original data
    X = X.fillna(X.median()).fillna(0)

    if len(X) < 5:
        return results

    model = IsolationForest(
        contamination="auto",
        random_state=42
    )

    predictions = model.fit_predict(X)
    scores = model.decision_function(X)

    results["anomaly_score"] = scores
    results["is_anomaly"] = predictions == -1

    return results
