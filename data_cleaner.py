
import pandas as pd


def clean_and_quarantine(df, ml_results=None):
    original = df.copy()

    # Remove exact duplicate rows from the clean output
    clean = original.drop_duplicates().copy()

    # Repair missing numeric values using column medians
    for column in ["quantity", "unit_price"]:
        if column in clean.columns:
            clean[column] = clean[column].fillna(
                clean[column].median()
            )

    # Identify records that violate business rules
    invalid_mask = pd.Series(False, index=clean.index)

    if "quantity" in clean.columns:
        invalid_mask |= clean["quantity"] < 0

    if "unit_price" in clean.columns:
        invalid_mask |= clean["unit_price"] < 0

    # ML anomalies need review, not automatic deletion
    if ml_results is not None and "is_anomaly" in ml_results.columns:
        anomaly_ids = set(
            ml_results.loc[ml_results["is_anomaly"], "record_id"]
        )
        invalid_mask |= clean["record_id"].isin(anomaly_ids)

    quarantine = clean.loc[invalid_mask].copy()
    safe_data = clean.loc[~invalid_mask].copy()

    return safe_data, quarantine
