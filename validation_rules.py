
import pandas as pd


def validate_dataset(df, rules=None):
    """
    Validate a DataFrame using optional column rules
    and common data-quality checks.
    """
    rules = rules or {}
    issues = []

    # Generic checks: missing values and duplicates
    for column in df.columns:
        missing_count = int(df[column].isna().sum())

        if missing_count:
            issues.append({
                "column": column,
                "issue": "missing_values",
                "count": missing_count,
                "details": f"{missing_count} missing value(s)"
            })

    duplicate_count = int(df.duplicated().sum())

    if duplicate_count:
        issues.append({
            "column": None,
            "issue": "duplicate_rows",
            "count": duplicate_count,
            "details": f"{duplicate_count} duplicate row(s)"
        })

    # Common semantic checks
    for column in df.columns:
        normalized = column.strip().lower()

        if normalized in {"quantity", "unit_price", "price", "amount"}:
            values = pd.to_numeric(df[column], errors="coerce")
            invalid_count = int((values < 0).sum())

            if invalid_count:
                issues.append({
                    "column": column,
                    "issue": "negative_value",
                    "count": invalid_count,
                    "details": "Values must be non-negative"
                })

        if normalized in {"latitude", "lat"}:
            values = pd.to_numeric(df[column], errors="coerce")
            invalid_count = int((~values.between(-90, 90)).sum())

            if invalid_count:
                issues.append({
                    "column": column,
                    "issue": "out_of_range",
                    "count": invalid_count,
                    "details": "Latitude must be between -90 and 90"
                })

        if normalized in {"longitude", "lon", "lng"}:
            values = pd.to_numeric(df[column], errors="coerce")
            invalid_count = int((~values.between(-180, 180)).sum())

            if invalid_count:
                issues.append({
                    "column": column,
                    "issue": "out_of_range",
                    "count": invalid_count,
                    "details": "Longitude must be between -180 and 180"
                })

    # User-supplied rules, e.g. {"risk_level": {"allowed": ["Low", "High"]}}
    for column, rule in rules.items():
        if column not in df.columns:
            issues.append({
                "column": column,
                "issue": "missing_column",
                "count": 1,
                "details": "Required column is absent"
            })
            continue

        if "allowed" in rule:
            invalid_mask = (
                df[column].notna()
                & ~df[column].isin(rule["allowed"])
            )
            count = int(invalid_mask.sum())

            if count:
                issues.append({
                    "column": column,
                    "issue": "invalid_category",
                    "count": count,
                    "details": f"Allowed values: {rule['allowed']}"
                })

        if "min" in rule or "max" in rule:
            values = pd.to_numeric(df[column], errors="coerce")
            invalid_mask = values.isna() & df[column].notna()

            if "min" in rule:
                invalid_mask |= values < rule["min"]

            if "max" in rule:
                invalid_mask |= values > rule["max"]

            count = int(invalid_mask.sum())

            if count:
                issues.append({
                    "column": column,
                    "issue": "out_of_range",
                    "count": count,
                    "details": (
                        f"Expected range: {rule.get('min', '-∞')} "
                        f"to {rule.get('max', '∞')}"
                    )
                })

    return pd.DataFrame(
        issues,
        columns=["column", "issue", "count", "details"]
    )
