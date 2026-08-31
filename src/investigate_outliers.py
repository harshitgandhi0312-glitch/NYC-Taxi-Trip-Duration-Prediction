"""
src/investigate_outliers.py
---------------------------
Deep-dive investigation of six unusual-value groups identified during EDA.

No records are deleted, modified, or capped.
All computation runs in DuckDB; the Parquet files are untouched.

Run:
    python src/investigate_outliers.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.db import get_connection, get_taxi_paths  # noqa: E402

# ── Helpers ───────────────────────────────────────────────────────────────────
SEP  = "=" * 72
SUB  = "-" * 72
SUB2 = "  " + "-" * 68

def hdr(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")

def sub(title: str) -> None:
    print(f"\n  {title}")
    print(SUB2)

def pct(n: int, total: int) -> str:
    return f"{n / total * 100:.4f}%"

# ── Setup ─────────────────────────────────────────────────────────────────────
con   = get_connection()
paths = get_taxi_paths()
p     = str(paths)

TOTAL = con.execute(f"SELECT COUNT(*) FROM read_parquet({p})").fetchone()[0]

ALL_COLS = [
    "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "passenger_count", "trip_distance", "RatecodeID",
    "PULocationID", "DOLocationID", "trip_duration_minutes",
]

print(SEP)
print("  NYC Taxi — Outlier Investigation Report")
print(f"  Total records in dataset : {TOTAL:,}")
print(SEP)

# =============================================================================
def investigate(
    group_label: str,
    group_number: str,
    where_clause: str,
    show_raw: bool = True,
) -> None:
    """Run the full investigation for one group and print results."""

    hdr(f"GROUP {group_number}: {group_label}")

    # 1. Count & percentage ---------------------------------------------------
    n = con.execute(
        f"SELECT COUNT(*) FROM read_parquet({p}) WHERE {where_clause}"
    ).fetchone()[0]
    print(f"\n  Records : {n:,}  ({pct(n, TOTAL)} of dataset)")

    if n == 0:
        print("  [No records found — skipping further analysis]")
        return

    # 2. Duration stats -------------------------------------------------------
    sub("Trip Duration (minutes)")
    r = con.execute(f"""
        SELECT
            MIN(trip_duration_minutes)                                 AS min,
            PERCENTILE_CONT(0.5) WITHIN GROUP
                (ORDER BY trip_duration_minutes)                       AS median,
            MAX(trip_duration_minutes)                                 AS max,
            AVG(trip_duration_minutes)                                 AS mean
        FROM read_parquet({p})
        WHERE {where_clause}
    """).fetchone()
    print(f"  Min={r[0]:.3f}  Median={r[1]:.3f}  Mean={r[3]:.3f}  Max={r[2]:.3f}")

    # 3. Distance stats -------------------------------------------------------
    sub("Trip Distance (miles)")
    r = con.execute(f"""
        SELECT
            MIN(trip_distance)                                         AS min,
            PERCENTILE_CONT(0.5) WITHIN GROUP
                (ORDER BY trip_distance)                               AS median,
            MAX(trip_distance)                                         AS max,
            AVG(trip_distance)                                         AS mean
        FROM read_parquet({p})
        WHERE {where_clause}
    """).fetchone()
    print(f"  Min={r[0]:.3f}  Median={r[1]:.3f}  Mean={r[3]:.3f}  Max={r[2]:.3f}")

    # 4. Passenger count distribution -----------------------------------------
    sub("Passenger Count Distribution")
    rows = con.execute(f"""
        SELECT CAST(passenger_count AS INTEGER) AS pax, COUNT(*) AS cnt
        FROM read_parquet({p})
        WHERE {where_clause}
        GROUP BY pax ORDER BY pax
    """).fetchall()
    print(f"  {'Pax':>5}  {'Count':>10}  {'%':>8}")
    for pax_val, cnt in rows:
        print(f"  {pax_val:>5}  {cnt:>10,}  {cnt/n*100:>7.2f}%")

    # 5. Top pickup zones -----------------------------------------------------
    sub("Top 10 Pickup Zones (PULocationID)")
    rows = con.execute(f"""
        SELECT PULocationID, COUNT(*) AS cnt
        FROM read_parquet({p})
        WHERE {where_clause}
        GROUP BY PULocationID
        ORDER BY cnt DESC
        LIMIT 10
    """).fetchall()
    print(f"  {'Zone':>6}  {'Count':>10}  {'%':>8}")
    for zone, cnt in rows:
        print(f"  {zone:>6}  {cnt:>10,}  {cnt/n*100:>7.2f}%")

    # 6. Top dropoff zones ----------------------------------------------------
    sub("Top 10 Dropoff Zones (DOLocationID)")
    rows = con.execute(f"""
        SELECT DOLocationID, COUNT(*) AS cnt
        FROM read_parquet({p})
        WHERE {where_clause}
        GROUP BY DOLocationID
        ORDER BY cnt DESC
        LIMIT 10
    """).fetchall()
    print(f"  {'Zone':>6}  {'Count':>10}  {'%':>8}")
    for zone, cnt in rows:
        print(f"  {zone:>6}  {cnt:>10,}  {cnt/n*100:>7.2f}%")

    # 7. Year distribution ----------------------------------------------------
    sub("Year Distribution")
    rows = con.execute(f"""
        SELECT YEAR(tpep_pickup_datetime) AS yr, COUNT(*) AS cnt
        FROM read_parquet({p})
        WHERE {where_clause}
        GROUP BY yr ORDER BY yr
    """).fetchall()
    for yr, cnt in rows:
        print(f"  {yr}  :  {cnt:>10,}  ({cnt/n*100:.2f}%)")

    # 8. Vendor distribution --------------------------------------------------
    sub("VendorID Distribution")
    rows = con.execute(f"""
        SELECT VendorID, COUNT(*) AS cnt
        FROM read_parquet({p})
        WHERE {where_clause}
        GROUP BY VendorID ORDER BY VendorID
    """).fetchall()
    for vendor, cnt in rows:
        print(f"  Vendor {vendor}  :  {cnt:>10,}  ({cnt/n*100:.2f}%)")

    # 9. Representative raw records -------------------------------------------
    if show_raw:
        sub("10 Representative Raw Records (random sample)")
        sample = con.execute(f"""
            SELECT {', '.join(ALL_COLS)}
            FROM (
                SELECT {', '.join(ALL_COLS)},
                       random() AS _r
                FROM read_parquet({p})
                WHERE {where_clause}
            )
            ORDER BY _r
            LIMIT 10
        """).df()
        for i, (_, row) in enumerate(sample.iterrows(), 1):
            print(f"\n  -- Record {i} --")
            for col in ALL_COLS:
                val = row[col]
                if col in ("tpep_pickup_datetime", "tpep_dropoff_datetime"):
                    print(f"     {col:<28}: {val}")
                elif isinstance(val, float):
                    print(f"     {col:<28}: {val:.4f}")
                else:
                    print(f"     {col:<28}: {val}")


# =============================================================================
# GROUP 1: Duration < 1 minute
# =============================================================================
investigate(
    group_label  = "Trip Duration < 1 Minute",
    group_number = "1",
    where_clause = "trip_duration_minutes < 1",
)
print(f"""
  ASSESSMENT
  ----------
  Near-zero durations (< 1 min) are almost certainly measurement or
  recording artefacts:
    - A real taxi trip cannot meaningfully start and end in under 60 s.
    - The very short distance values observed confirm no genuine travel.
    - Common causes: meter started / stopped immediately, GPS glitch,
      trip cancellation recorded as completed.

  VERDICT : A — Clearly Invalid
  RECOMMENDATION : Remove before feature engineering (100k rows, 0.14%).
""")

# =============================================================================
# GROUP 2: Duration 1–2 minutes
# =============================================================================
investigate(
    group_label  = "Trip Duration 1 to 2 Minutes",
    group_number = "2",
    where_clause = "trip_duration_minutes >= 1 AND trip_duration_minutes < 2",
)
print(f"""
  ASSESSMENT
  ----------
  These are at the borderline. Some could be genuine very short moves
  (one block in Manhattan); others may still be recording errors.
    - Median distance will indicate if any real travel occurred.
    - If median distance < 0.2 mi, majority are likely artefacts.
    - If median distance ~ 0.5+ mi, some may be legitimately short trips.

  VERDICT : C — Uncertain (decision required)
  RECOMMENDATION : Review median distance. If < 0.2 mi, remove alongside
                   group 1. If > 0.2 mi, consider keeping with a lower-
                   bound flag feature. Revisit after feature engineering.
""")

# =============================================================================
# GROUP 3: Duration > 120 minutes
# =============================================================================
investigate(
    group_label  = "Trip Duration > 120 Minutes",
    group_number = "3",
    where_clause = "trip_duration_minutes > 120",
)
print(f"""
  ASSESSMENT
  ----------
  Trips over 2 hours require careful examination:
    - Long-haul airport trips (JFK, EWR, LGA) are genuine and common.
    - A trip from Manhattan to JFK legitimately takes 60-90 min in traffic.
    - Trips over 120 min could be: genuine long hauls, meter-left-running,
      or a driver who forgot to end the trip.
    - The zone distribution will reveal if these are airport-heavy.
    - If distance is large (10+ mi) AND duration is 120-180 min, likely real.
    - If distance is tiny but duration is 120+ min, meter-left-running.

  VERDICT : B — Potentially Valid but Rare (for long-distance trips)
             A — Clearly Invalid (if distance < 2 mi and duration > 120 min)
  RECOMMENDATION : Apply a conditional rule during feature engineering:
                   - Keep if distance >= 10 mi
                   - Remove/cap if distance < 2 mi
                   Approximately 23k records (0.033%).
""")

# =============================================================================
# GROUP 4: Distance < 0.1 miles
# =============================================================================
investigate(
    group_label  = "Trip Distance < 0.1 Miles",
    group_number = "4",
    where_clause = "trip_distance < 0.1",
)
print(f"""
  ASSESSMENT
  ----------
  Near-zero distances point to GPS or metering failures:
    - 0.1 miles = ~160 metres. No taxi trip would be this short
      (passengers would walk).
    - However, duration distribution matters: a long-duration / zero-
      distance trip = meter left running with vehicle parked.
    - Short duration / zero distance = genuine data error or cancelled trip.

  VERDICT : A — Clearly Invalid (the vast majority)
  RECOMMENDATION : Remove before feature engineering (112k rows, 0.16%).
                   Note: overlap with group 1 is likely substantial.
""")

# =============================================================================
# GROUP 5: Distance > 50 miles
# =============================================================================
investigate(
    group_label  = "Trip Distance > 50 Miles",
    group_number = "5",
    where_clause = "trip_distance > 50",
)
print(f"""
  ASSESSMENT
  ----------
  Very long distances (>50 mi) include genuine out-of-city trips:
    - Newark Airport (EWR) from Manhattan is ~15-18 mi.
    - JFK from Manhattan is ~15 mi.
    - 50+ mi is unusual but not impossible (Connecticut, Long Island).
    - Inspect whether duration is proportionally long (plausible speed).
    - Implausibly fast trips (>50 mi in <10 min) are GPS errors.

  VERDICT : B — Potentially Valid but Rare
  RECOMMENDATION : Keep but cap at a sensible percentile (e.g. 99.9th)
                   during feature engineering. Flag as "long haul" indicator.
                   Only 9,219 records (0.013%).
""")

# =============================================================================
# GROUP 6: Passenger Count > 6
# =============================================================================
investigate(
    group_label  = "Passenger Count > 6",
    group_number = "6",
    where_clause = "passenger_count > 6",
)
print(f"""
  ASSESSMENT
  ----------
  NYC taxi regulations cap standard taxis at 4-5 passengers; max
  is 6 for some large vehicles (e.g. vans). Counts > 6 are invalid
  under any NYC TLC vehicle classification.
    - These are almost certainly data-entry errors (e.g. 9 instead of 0).
    - Only 174 records — negligible.

  VERDICT : A — Clearly Invalid
  RECOMMENDATION : Remove or clip to 6 before feature engineering.
                   Negligible impact (174 records, 0.0002%).
""")

# =============================================================================
# FINAL SUMMARY TABLE
# =============================================================================
hdr("FINAL RECOMMENDATION SUMMARY")
print(f"""
  {'Group':<40} {'Count':>10}  {'%':>8}  Verdict
  {'-'*39:<40} {'-'*10:>10}  {'-'*8:>8}  {'-------'}""")

groups = [
    ("Duration < 1 min",          "trip_duration_minutes < 1",                                         "A - Remove"),
    ("Duration 1-2 min",          "trip_duration_minutes >= 1 AND trip_duration_minutes < 2",           "C - Decide"),
    ("Duration > 120 min",        "trip_duration_minutes > 120",                                        "B/A - Conditional"),
    ("Distance < 0.1 mi",         "trip_distance < 0.1",                                                "A - Remove"),
    ("Distance > 50 mi",          "trip_distance > 50",                                                 "B - Keep+Flag"),
    ("Passenger count > 6",       "passenger_count > 6",                                                "A - Remove"),
]

for label, clause, verdict in groups:
    cnt = con.execute(
        f"SELECT COUNT(*) FROM read_parquet({p}) WHERE {clause}"
    ).fetchone()[0]
    print(f"  {label:<40} {cnt:>10,}  {pct(cnt,TOTAL):>8}  {verdict}")

print(f"""
  Notes:
  - Groups 1 and 4 likely overlap significantly (short duration AND short distance).
  - After applying group 1 + group 4 + group 6 removals, an estimated
    ~200k rows (<0.3% of 71M) will be removed.
  - Group 2 (1-2 min) decision deferred to feature-engineering stage.
  - Group 3 (>120 min) will use a distance-conditional rule.
  - Group 5 (>50 mi) is kept with a long-haul flag.
""")

print(SEP)
print("  Investigation complete. No data was modified.")
print(SEP)
