"""
src/validate_data.py
--------------------
Reproducible data-validation stage for the NYC Taxi Trip Duration project.

All heavy computation runs inside DuckDB — the full dataset is never loaded
into memory.  The Parquet files are not modified.

Run:
    python src/validate_data.py
"""

import sys
from pathlib import Path

# Allow `from src.db import ...` when run from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import get_connection, get_taxi_paths  # noqa: E402

# ── Helpers ───────────────────────────────────────────────────────────────────

SEP = "=" * 64
SUB = "-" * 64


def header(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def subheader(title: str) -> None:
    print(f"\n{SUB}")
    print(f"  {title}")
    print(SUB)


def fmt(value) -> str:
    """Format integers with commas; leave other types as-is."""
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.4f}"
    return str(value)


# ── Main validation ───────────────────────────────────────────────────────────

def main() -> None:
    print(SEP)
    print("  NYC Taxi Trip Duration — Data Validation Report")
    print(SEP)

    con = get_connection()
    paths = get_taxi_paths()
    p = str(paths)          # DuckDB accepts a Python list literal directly

    # ── 1. Total trips ────────────────────────────────────────────────────────
    header("1. TOTAL NUMBER OF TRIPS")
    total = con.execute(f"SELECT COUNT(*) FROM read_parquet({p})").fetchone()[0]
    print(f"  Total trips : {total:,}")

    # ── 2. Trips by year ──────────────────────────────────────────────────────
    header("2. TRIPS BY YEAR")
    rows = con.execute(f"""
        SELECT
            YEAR(tpep_pickup_datetime) AS year,
            COUNT(*)                   AS trips
        FROM read_parquet({p})
        GROUP BY year
        ORDER BY year
    """).fetchall()
    print(f"  {'Year':<8} {'Trips':>15}")
    print(f"  {'-'*7:<8} {'-'*15:>15}")
    for year, count in rows:
        print(f"  {year:<8} {count:>15,}")

    # ── 3. Pickup datetime range ──────────────────────────────────────────────
    header("3. PICKUP DATETIME RANGE")
    r = con.execute(f"""
        SELECT
            MIN(tpep_pickup_datetime) AS min_pickup,
            MAX(tpep_pickup_datetime) AS max_pickup
        FROM read_parquet({p})
    """).fetchone()
    print(f"  Min : {r[0]}")
    print(f"  Max : {r[1]}")

    # ── 4. Dropoff datetime range ─────────────────────────────────────────────
    header("4. DROPOFF DATETIME RANGE")
    r = con.execute(f"""
        SELECT
            MIN(tpep_dropoff_datetime) AS min_dropoff,
            MAX(tpep_dropoff_datetime) AS max_dropoff
        FROM read_parquet({p})
    """).fetchone()
    print(f"  Min : {r[0]}")
    print(f"  Max : {r[1]}")

    # ── 5. Column names and data types ────────────────────────────────────────
    header("5. COLUMN NAMES AND DATA TYPES")
    schema = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet({p}) LIMIT 0"
    ).fetchall()
    print(f"  {'#':<4} {'Column':<26} {'Type'}")
    print(f"  {'-'*3:<4} {'-'*25:<26} {'-'*12}")
    for i, row in enumerate(schema, 1):
        col, dtype = row[0], row[1]
        print(f"  {i:<4} {col:<26} {dtype}")

    # ── 6. Missing-value counts ───────────────────────────────────────────────
    header("6. MISSING VALUE COUNTS (NULL)")
    col_names = [row[0] for row in schema]
    null_exprs = ",\n        ".join(
        f"COUNT(*) FILTER (WHERE {c} IS NULL) AS \"{c}\""
        for c in col_names
    )
    null_row = con.execute(
        f"SELECT {null_exprs} FROM read_parquet({p})"
    ).fetchone()
    print(f"  {'Column':<26} {'Nulls':>12} {'% of total':>12}")
    print(f"  {'-'*25:<26} {'-'*12:>12} {'-'*12:>12}")
    any_nulls = False
    for col, nulls in zip(col_names, null_row):
        pct = (nulls / total * 100) if total else 0
        flag = " [!!]" if nulls > 0 else ""
        print(f"  {col:<26} {nulls:>12,} {pct:>11.2f}%{flag}")
        if nulls > 0:
            any_nulls = True
    if not any_nulls:
        print("\n  [OK] No missing values detected.")

    # ── 7. Trip duration statistics ───────────────────────────────────────────
    header("7. TRIP DURATION STATISTICS (minutes)")
    r = con.execute(f"""
        SELECT
            MIN(trip_duration_minutes)                          AS min,
            PERCENTILE_CONT(0.25) WITHIN GROUP
                (ORDER BY trip_duration_minutes)                AS p25,
            PERCENTILE_CONT(0.50) WITHIN GROUP
                (ORDER BY trip_duration_minutes)                AS median,
            PERCENTILE_CONT(0.75) WITHIN GROUP
                (ORDER BY trip_duration_minutes)                AS p75,
            PERCENTILE_CONT(0.95) WITHIN GROUP
                (ORDER BY trip_duration_minutes)                AS p95,
            PERCENTILE_CONT(0.99) WITHIN GROUP
                (ORDER BY trip_duration_minutes)                AS p99,
            MAX(trip_duration_minutes)                          AS max
        FROM read_parquet({p})
    """).fetchone()
    labels = ["Min", "25th pct", "Median", "75th pct", "95th pct", "99th pct", "Max"]
    for label, val in zip(labels, r):
        print(f"  {label:<12} : {val:>10.2f} min")

    # ── 8. Trip distance statistics ───────────────────────────────────────────
    header("8. TRIP DISTANCE STATISTICS (miles)")
    r = con.execute(f"""
        SELECT
            MIN(trip_distance)                                  AS min,
            PERCENTILE_CONT(0.50) WITHIN GROUP
                (ORDER BY trip_distance)                        AS median,
            AVG(trip_distance)                                  AS mean,
            MAX(trip_distance)                                  AS max
        FROM read_parquet({p})
    """).fetchone()
    labels = ["Min", "Median", "Mean", "Max"]
    for label, val in zip(labels, r):
        print(f"  {label:<12} : {val:>10.2f} miles")

    # ── 9. Unique pickup zones ────────────────────────────────────────────────
    header("9. UNIQUE PICKUP ZONES (PULocationID)")
    count = con.execute(
        f"SELECT COUNT(DISTINCT PULocationID) FROM read_parquet({p})"
    ).fetchone()[0]
    print(f"  Distinct pickup zones : {count:,}")

    # ── 10. Unique dropoff zones ───────────────────────────────────────────────
    header("10. UNIQUE DROPOFF ZONES (DOLocationID)")
    count = con.execute(
        f"SELECT COUNT(DISTINCT DOLocationID) FROM read_parquet({p})"
    ).fetchone()[0]
    print(f"  Distinct dropoff zones : {count:,}")

    # ── 11. Distinct VendorID values ───────────────────────────────────────────
    header("11. VENDOR ID VALUES")
    rows = con.execute(f"""
        SELECT VendorID, COUNT(*) AS trips
        FROM read_parquet({p})
        GROUP BY VendorID
        ORDER BY VendorID
    """).fetchall()
    distinct = len(rows)
    print(f"  Distinct VendorID values : {distinct}")
    print()
    print(f"  {'VendorID':<12} {'Trips':>15}")
    print(f"  {'-'*11:<12} {'-'*15:>15}")
    for vendor_id, count in rows:
        print(f"  {str(vendor_id):<12} {count:>15,}")

    # ── 12. Exact duplicate rows ───────────────────────────────────────────────
    header("12. EXACT DUPLICATE ROWS")
    dup_count = con.execute(f"""
        SELECT SUM(cnt - 1) AS duplicates
        FROM (
            SELECT COUNT(*) AS cnt
            FROM read_parquet({p})
            GROUP BY {', '.join(col_names)}
            HAVING COUNT(*) > 1
        )
    """).fetchone()[0]
    dup_count = dup_count if dup_count is not None else 0
    pct = (dup_count / total * 100) if total else 0
    flag = " [!!]" if dup_count > 0 else " [OK]"
    print(f"  Exact duplicate rows : {int(dup_count):,}  ({pct:.4f}% of total){flag}")

    # ── Summary footer ─────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Validation complete — no data was modified.")
    print(SEP)


if __name__ == "__main__":
    main()
