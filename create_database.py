#!/usr/bin/env python3
"""Create a SQLite database from Pilot_v1.csv."""

import csv
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CSV_PATH = REPO_ROOT / "Pilot_v1.csv"
DB_PATH = REPO_ROOT / "fatigue_data.db"
TABLE_NAME = "pilot_data"


def normalize_column_name(name: str) -> str:
    """Normalize CSV headers into SQL-friendly snake_case names."""
    normalized = name.strip().lower().replace(" ", "_").replace("-", "_")
    return "".join(char if char.isalnum() or char == "_" else "_" for char in normalized)


def infer_sqlite_type(value: str) -> str:
    """Infer SQLite type from a sample value."""
    if value is None or value == "":
        return "TEXT"

    try:
        int(value)
        return "INTEGER"
    except ValueError:
        pass

    try:
        float(value)
        return "REAL"
    except ValueError:
        return "TEXT"


def create_database(csv_path: Path, db_path: Path, table_name: str) -> None:
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            raise ValueError("CSV file has no header row.")

        original_headers = reader.fieldnames
        headers = [normalize_column_name(header) for header in original_headers]
        rows = list(reader)

    sample_row = rows[0] if rows else {}
    column_types = [infer_sqlite_type(sample_row.get(original, "")) for original in original_headers]

    quoted_headers = [f'"{header}"' for header in headers]
    schema_columns = [
        f'{quoted_headers[index]} {column_types[index]}'
        for index in range(len(headers))
    ]

    insert_placeholders = ", ".join(["?"] * len(headers))
    insert_columns = ", ".join(quoted_headers)

    db_path.unlink(missing_ok=True)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        cursor.execute(f'CREATE TABLE "{table_name}" ({", ".join(schema_columns)})')

        values = [
            tuple(row.get(original, None) for original in original_headers)
            for row in rows
        ]
        cursor.executemany(
            f'INSERT INTO "{table_name}" ({insert_columns}) VALUES ({insert_placeholders})',
            values,
        )
        conn.commit()


if __name__ == "__main__":
    create_database(CSV_PATH, DB_PATH, TABLE_NAME)
    print(f"Created {DB_PATH.name} with table '{TABLE_NAME}' from {CSV_PATH.name}")
