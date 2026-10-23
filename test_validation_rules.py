
import pandas as pd

from validation_rules import validate_dataset


def test_detects_invalid_coordinates():
    df = pd.DataFrame({
        "Latitude": [17.4, 120],
        "Longitude": [78.5, 78.6]
    })

    issues = validate_dataset(df)

    assert "out_of_range" in issues["issue"].values


def test_custom_allowed_values():
    df = pd.DataFrame({
        "risk_level": ["Low", "High", "Unknown"]
    })

    issues = validate_dataset(
        df,
        rules={"risk_level": {"allowed": ["Low", "High"]}}
    )

    assert "invalid_category" in issues["issue"].values


def test_clean_dataset_has_no_issues():
    df = pd.DataFrame({
        "Latitude": [17.4, 17.5],
        "Longitude": [78.5, 78.6]
    })

    issues = validate_dataset(df)

    assert issues.empty
