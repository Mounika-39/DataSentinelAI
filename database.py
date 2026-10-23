
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path("reports/datasentinel.db")


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scanned_at TEXT NOT NULL,
                total_records INTEGER,
                missing_values INTEGER,
                duplicate_rows INTEGER,
                invalid_values INTEGER,
                anomalies_detected INTEGER
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS scan_issues (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id INTEGER NOT NULL,
                column_name TEXT,
                issue_type TEXT NOT NULL,
                issue_count INTEGER NOT NULL,
                details TEXT,
                action_taken TEXT,
                FOREIGN KEY (scan_id) REFERENCES scans(id)
            )
        """)


def save_scan(report, total_records, anomalies_detected):
    missing = sum(report.get("missing_values", {}).values())
    duplicates = report.get("duplicate_rows", 0)
    invalid = sum(report.get("invalid_values", {}).values())

    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO scans (
                scanned_at, total_records, missing_values,
                duplicate_rows, invalid_values, anomalies_detected
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            total_records,
            int(missing),
            int(duplicates),
            int(invalid),
            int(anomalies_detected),
        ))
        return cursor.lastrowid


def save_scan_issues(scan_id, issues, action_taken="Flagged for review"):
    with get_connection() as conn:
        for _, issue in issues.iterrows():
            conn.execute("""
                INSERT INTO scan_issues (
                    scan_id, column_name, issue_type,
                    issue_count, details, action_taken
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                scan_id,
                issue.get("column"),
                issue.get("issue", "unknown"),
                int(issue.get("count", 0)),
                str(issue.get("details", "")),
                action_taken,
            ))


def get_scan_history(limit=20):
    with get_connection() as conn:
        return conn.execute("""
            SELECT id, scanned_at, total_records, missing_values,
                   duplicate_rows, invalid_values, anomalies_detected
            FROM scans
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()


def get_scan_issues(scan_id):
    with get_connection() as conn:
        return conn.execute("""
            SELECT column_name, issue_type, issue_count, details, action_taken
            FROM scan_issues
            WHERE scan_id = ?
            ORDER BY id
        """, (scan_id,)).fetchall()
