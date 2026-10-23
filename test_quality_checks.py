
import pandas as pd

from quality_checks import run_quality_checks


def test_detects_missing_values():
    df = pd.DataFrame({
        "product": ["Laptop", None],
        "price": [1000, 2000]
    })

    report = run_quality_checks(df)

    assert report["missing_values"]["product"] == 1


def test_detects_duplicate_rows():
    df = pd.DataFrame({
        "product": ["Mouse", "Mouse"],
        "price": [500, 500]
    })

    report = run_quality_checks(df)

    assert report["duplicate_rows"] == 1


def test_detects_negative_values():
    df = pd.DataFrame({
        "quantity": [2, -1],
        "unit_price": [100, 200]
    })

    report = run_quality_checks(df)

    assert report["invalid_values"]["quantity"] == 1
    assert report["invalid_values"]["unit_price"] == 0


def test_accepts_clean_dataset():
    df = pd.DataFrame({
        "product": ["Laptop", "Mouse"],
        "price": [50000, 500]
    })

    report = run_quality_checks(df)

    assert report["missing_values"] == {}
    assert report["duplicate_rows"] == 0
    assert report["invalid_values"] == {}
