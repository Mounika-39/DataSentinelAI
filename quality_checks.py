
import pandas as pd


def run_quality_checks(df):
    return {
        "missing_values": df.isnull().sum()[
            df.isnull().sum() > 0
        ].to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "invalid_values": {
            column: int((df[column] < 0).sum())
            for column in ["quantity", "unit_price"]
            if column in df.columns
            and pd.api.types.is_numeric_dtype(df[column])
        },
        "schema": {
            "columns_found": df.columns.tolist(),
            "column_count": len(df.columns),
            "duplicate_column_names": bool(
                df.columns.duplicated().any()
            )
        }
    }
