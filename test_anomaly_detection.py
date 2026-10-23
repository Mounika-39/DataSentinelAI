
import pandas as pd

from anomaly_detection import detect_anomalies


def test_anomaly_output_columns():
    df = pd.DataFrame({
        "quantity": [1, 2, 1, 3, 2, 1000],
        "unit_price": [100, 200, 150, 300, 250, 999999]
    })

    results = detect_anomalies(df)

    assert "is_anomaly" in results.columns
    assert "anomaly_score" in results.columns
    assert len(results) == len(df)
    assert results["is_anomaly"].dtype == bool


def test_small_dataset_skips_anomaly_detection():
    df = pd.DataFrame({
        "latitude": [17.39, 17.38, 17.40, 17.41],
        "longitude": [78.31, 78.32, 78.33, 78.34]
    })

    results = detect_anomalies(df)

    assert not results["is_anomaly"].any()
