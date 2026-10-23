
import pandas as pd
from pathlib import Path
from anomaly_detection import detect_anomalies
from quality_checks import run_quality_checks
from data_cleaner import clean_and_quarantine
from database import (
    initialize_database,
    save_scan,
    get_scan_history
)


data = {
    "record_id": [101, 102, 103, 104, 105, 106, 107, 108],
    "product": [
        "Laptop", "Mouse", "Keyboard", "Monitor",
        "Laptop", "Mouse", "Headphones", "Keyboard"
    ],
    "quantity": [1, 2, 1, 1, 3, 2, 1, 1],
    "unit_price": [55000, 700, 1500, 12000, 55000, 700, 2500, 1500],
    "region": [
        "Hyderabad", "Chennai", "Delhi", "Mumbai",
        "Hyderabad", "Chennai", "Pune", "Delhi"
    ]
}

df = pd.DataFrame(data)

# Introduce intentional data-quality problems
df.loc[2, "unit_price"] = None       # Missing price
df.loc[5, "quantity"] = -2           # Invalid quantity
df.loc[6, "unit_price"] = 999999     # Potential price anomaly

# Add an unexpected column
df["customer_note"] = [
    "Regular", "Regular", "Check", "Regular",
    "Regular", "Urgent", "Review", "Regular"
]

# Add a duplicate record
df = pd.concat([df, df.iloc[[1]]], ignore_index=True)

output_path = Path("data") / "sales.csv"
df.to_csv(output_path, index=False)



report = run_quality_checks(df)
print("\n===== ML ANOMALY DETECTION =====")

results = detect_anomalies(df)

anomalies = results[results["is_anomaly"]]

print("\nSuspicious records:")
print(
    anomalies[
        ["record_id", "product", "quantity",
         "unit_price", "anomaly_score"]
    ].to_string(index=False)
)

print("\nTotal records:", len(results))
print("Anomalies detected:", len(anomalies))


print("\n===== SAFE CLEANING =====")

safe_data, quarantine = clean_and_quarantine(df, results)

safe_data.to_csv("reports/cleaned_sales.csv", index=False)
quarantine.to_csv("reports/quarantined_sales.csv", index=False)

print("Original records:", len(df))
print("Cleaned records:", len(safe_data))
print("Quarantined records:", len(quarantine))

print("\nQuarantined records:")
print(quarantine.to_string(index=False))

print("\nOutput files saved in the reports folder.")

initialize_database()

save_scan(
    report=report,
    total_records=len(df),
    anomalies_detected=int(results["is_anomaly"].sum())
)

print("\n===== SCAN HISTORY =====")

for scan in get_scan_history():
    print(scan)