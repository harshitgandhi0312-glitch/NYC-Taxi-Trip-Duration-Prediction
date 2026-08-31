"""
src/clean_data.py
-----------------
Reproducible cleaning and feature-engineering pipeline for the
NYC Taxi Trip Duration Prediction project.

Design principles
-----------------
* Input  : two existing final Parquet files (read-only via DuckDB).
* Output : data/processed/cleaned_taxi_2023_2024.parquet
* All heavy work runs inside DuckDB — no 71M-row Pandas DataFrames.
* Source files are never modified.
* Every cleaning rule is documented with its investigation rationale.
* Only features available at PICKUP TIME are created; no future
  information or target-derived features are included.

Run:
    python src/clean_data.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.db import get_connection, get_taxi_paths  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
OUT_PATH  = ROOT / "data" / "processed" / "cleaned_taxi_2023_2024.parquet"

SEP  = "=" * 72
SUB  = "-" * 72

def hdr(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")

def sub(title: str) -> None:
    print(f"\n  {title}")
    print("  " + "-" * 68)

# ── Connect ───────────────────────────────────────────────────────────────────
con   = get_connection()
paths = get_taxi_paths()
p     = str(paths)

print(SEP)
print("  NYC Taxi — Cleaning & Feature Engineering Pipeline")
print(SEP)
print(f"\n  Input  : {paths[0]}")
print(f"           {paths[1]}")
print(f"  Output : {OUT_PATH}")

# =============================================================================
# STEP 1: BASELINE COUNT
# =============================================================================
hdr("STEP 1: BASELINE ROW COUNT")
total_original = con.execute(
    f"SELECT COUNT(*) FROM read_parquet({p})"
).fetchone()[0]
print(f"\n  Original row count : {total_original:,}")

# =============================================================================
# STEP 2: COUNT ROWS REMOVED BY EACH RULE (INDEPENDENTLY)
# =============================================================================
hdr("STEP 2: CLEANING RULES — INDIVIDUAL IMPACT")

# ── Rule 1 ────────────────────────────────────────────────────────────────────
# RATIONALE: EDA found 100,551 trips with duration < 1 min. Median distance
# for this group was 0.09 mi — below 0.1 mi threshold. These are meter-start/
# stop artefacts or immediate cancellations, not real trips. No passenger
# could realistically be transported in under 60 seconds.
r1_count = con.execute(
    f"SELECT COUNT(*) FROM read_parquet({p}) WHERE trip_duration_minutes < 1"
).fetchone()[0]
print(f"\n  Rule 1  | duration < 1 min (artefacts / cancelled trips)")
print(f"           Rows removed : {r1_count:,}  ({r1_count/total_original*100:.4f}%)")

# ── Rule 2 ────────────────────────────────────────────────────────────────────
# RATIONALE: EDA found 112,126 trips with distance < 0.1 mi. Crucially,
# 100% of these belonged to Vendor 2 — confirming a systematic GPS/reporting
# defect in that vendor's telemetry, not random noise. Trips covering < 0.1 mi
# (~160 metres) are not plausible taxi trips; the long-duration tail (median
# 5.5 min with 0.01 mi distance) represents the meter running while parked.
r2_count = con.execute(
    f"SELECT COUNT(*) FROM read_parquet({p}) WHERE trip_distance < 0.1"
).fetchone()[0]
print(f"\n  Rule 2  | distance < 0.1 mi (Vendor 2 GPS defect; no real travel)")
print(f"           Rows removed : {r2_count:,}  ({r2_count/total_original*100:.4f}%)")

# ── Rule 3 ────────────────────────────────────────────────────────────────────
# RATIONALE: Outlier investigation showed that trips > 120 min split into two
# clear populations:
#   (a) Legitimate long-haul/airport trips: large distance (median 17.7 mi),
#       dominated by JFK pickups (zone 132, 29%). These are real trips.
#   (b) Meter-left-running artefacts: very long duration but tiny distance
#       (< 2 mi). A trip taking 2+ hours but covering < 2 miles is not a
#       plausible taxi journey in any NYC scenario.
# Only population (b) is removed; population (a) is retained.
r3_count = con.execute(f"""
    SELECT COUNT(*)
    FROM read_parquet({p})
    WHERE trip_duration_minutes > 120
      AND trip_distance < 2
""").fetchone()[0]
print(f"\n  Rule 3  | duration > 120 min AND distance < 2 mi (meter-left-running)")
print(f"           Rows removed : {r3_count:,}  ({r3_count/total_original*100:.4f}%)")

# ── Rule 4 ────────────────────────────────────────────────────────────────────
# RATIONALE: Outlier investigation confirmed that trips with distance > 50 mi
# are predominantly JFK pickups (zone 132: 64.4%) dropping at zone 265
# (out-of-city: 93.9%), with median duration of 86 min — fully consistent
# with highway speeds for the observed distances. These are rare (9,219 rows,
# 0.013%) but legitimate. They are kept and flagged with is_long_haul = 1.
# (No rows removed — documented for transparency.)
r4_kept = con.execute(
    f"SELECT COUNT(*) FROM read_parquet({p}) WHERE trip_distance > 50"
).fetchone()[0]
print(f"\n  Rule 4  | distance > 50 mi — KEPT (legitimate long-haul trips)")
print(f"           Rows kept    : {r4_kept:,}  ({r4_kept/total_original*100:.4f}%)")

# ── Rule 5 ────────────────────────────────────────────────────────────────────
# RATIONALE: 491,495 trips with duration 1–2 min were investigated. Their
# median distance was 0.35 mi — above the 0.1 mi GPS-defect threshold. These
# could be genuine very short trips in dense Manhattan zones (236/237/263).
# Because the distance criterion already filters the clear GPS artefacts,
# we do NOT apply an additional duration lower bound beyond 1 min. Removing
# this group would risk deleting ~490k real (if rare) trips.
r5_kept = con.execute(f"""
    SELECT COUNT(*)
    FROM read_parquet({p})
    WHERE trip_duration_minutes >= 1
      AND trip_duration_minutes < 2
      AND trip_distance >= 0.1
""").fetchone()[0]
print(f"\n  Rule 5  | duration 1-2 min with distance >= 0.1 mi — KEPT")
print(f"           Rows kept    : {r5_kept:,}  ({r5_kept/total_original*100:.4f}%)")

# ── Rule 6 ────────────────────────────────────────────────────────────────────
# RATIONALE: NYC TLC regulations cap yellow taxi passenger count at 4–5,
# with an absolute max of 6 for large vehicles. Counts > 6 are data-entry
# errors. Outlier investigation found only 174 such records (0.0002%), all
# with RatecodeID = 5 (negotiated/van fare). Removing these has negligible
# data-volume impact and avoids corrupting the passenger_count feature.
r6_count = con.execute(
    f"SELECT COUNT(*) FROM read_parquet({p}) WHERE passenger_count > 6"
).fetchone()[0]
print(f"\n  Rule 6  | passenger_count > 6 (exceeds NYC TLC maximum)")
print(f"           Rows removed : {r6_count:,}  ({r6_count/total_original*100:.4f}%)")

# ── Combined removals (union of all removal conditions) ───────────────────────
total_removed = con.execute(f"""
    SELECT COUNT(*)
    FROM read_parquet({p})
    WHERE trip_duration_minutes < 1
       OR trip_distance < 0.1
       OR (trip_duration_minutes > 120 AND trip_distance < 2)
       OR passenger_count > 6
""").fetchone()[0]

total_final = total_original - total_removed
pct_retained = total_final / total_original * 100

print(f"\n  {'-'*68}")
print(f"  Total rows removed (union) : {total_removed:,}  ({total_removed/total_original*100:.4f}%)")
print(f"  Final row count            : {total_final:,}")
print(f"  Percentage retained        : {pct_retained:.4f}%")

# =============================================================================
# STEP 3: CLEANING + FEATURE ENGINEERING QUERY
# =============================================================================
hdr("STEP 3: APPLYING CLEANING RULES & ENGINEERING FEATURES")

# DuckDB DAYOFWEEK: 0 = Sunday, 6 = Saturday
# is_weekend = 1 when DAYOFWEEK IN (0, 6)

CLEAN_SQL = f"""
COPY (
    SELECT
        -- ── TARGET (label) ────────────────────────────────────────────────
        -- trip_duration_minutes is the prediction target; it is included in
        -- the output so train/test splits can be made downstream, but it must
        -- NEVER be used as an input feature.
        trip_duration_minutes,

        -- ── TIME FEATURES (available at pickup time) ──────────────────────
        CAST(YEAR(tpep_pickup_datetime)      AS INTEGER) AS pickup_year,
        CAST(MONTH(tpep_pickup_datetime)     AS INTEGER) AS pickup_month,
        CAST(DAY(tpep_pickup_datetime)       AS INTEGER) AS pickup_day,

        -- DAYOFWEEK: 0 = Sunday … 6 = Saturday (DuckDB convention)
        CAST(DAYOFWEEK(tpep_pickup_datetime) AS INTEGER) AS pickup_day_of_week,

        CAST(HOUR(tpep_pickup_datetime)      AS INTEGER) AS pickup_hour,

        -- is_weekend: Saturday (6) or Sunday (0)
        CAST(
            CASE WHEN DAYOFWEEK(tpep_pickup_datetime) IN (0, 6) THEN 1 ELSE 0 END
        AS INTEGER) AS is_weekend,

        -- ── TRIP FEATURES (known at or before pickup) ─────────────────────
        trip_distance,

        -- is_long_haul: flagged from outlier investigation; distance > 50 mi
        -- are legitimate JFK/out-of-city trips needing separate treatment.
        CAST(
            CASE WHEN trip_distance > 50 THEN 1 ELSE 0 END
        AS INTEGER) AS is_long_haul,

        CAST(passenger_count AS INTEGER)    AS passenger_count,
        CAST(RatecodeID      AS INTEGER)    AS RatecodeID,
        CAST(VendorID        AS INTEGER)    AS VendorID,
        CAST(PULocationID    AS INTEGER)    AS PULocationID,
        CAST(DOLocationID    AS INTEGER)    AS DOLocationID

        -- EXCLUDED: tpep_pickup_datetime (raw; replaced by extracted features)
        -- EXCLUDED: tpep_dropoff_datetime (future information — data leakage)

    FROM read_parquet({p})

    WHERE
        -- Rule 1: Remove sub-1-min trips (meter artefacts / cancellations)
        trip_duration_minutes >= 1

        -- Rule 2: Remove near-zero distance trips (Vendor 2 GPS defect)
        AND trip_distance >= 0.1

        -- Rule 3: Remove suspicious long-duration / short-distance combos
        --         (meter left running; NOT legitimate airport trips)
        AND NOT (trip_duration_minutes > 120 AND trip_distance < 2)

        -- Rule 4+5: No filter on trip_distance > 50 or duration 1-2 min;
        --           these are retained as investigated.

        -- Rule 6: Remove invalid passenger counts (> NYC TLC maximum of 6)
        AND passenger_count <= 6

) TO '{OUT_PATH.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
"""

print(f"\n  Writing cleaned dataset to:\n  {OUT_PATH}\n")
t0 = time.time()
con.execute(CLEAN_SQL)
elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f}s")

# =============================================================================
# STEP 4: VALIDATION REPORT ON OUTPUT FILE
# =============================================================================
hdr("STEP 4: OUTPUT VALIDATION REPORT")

out = f"['{OUT_PATH.as_posix()}']"

# Row count
actual_rows = con.execute(
    f"SELECT COUNT(*) FROM read_parquet({out})"
).fetchone()[0]

sub("Row Counts")
print(f"  Original rows      : {total_original:,}")
print(f"  Rows removed       : {total_removed:,}  ({total_removed/total_original*100:.4f}%)")
print(f"    Rule 1 (dur<1)   : {r1_count:,}")
print(f"    Rule 2 (dist<0.1): {r2_count:,}")
print(f"    Rule 3 (dur>120 & dist<2): {r3_count:,}")
print(f"    Rule 6 (pax>6)   : {r6_count:,}")
print(f"    (overlap correction applied in union count)")
print(f"  Final rows         : {actual_rows:,}")
print(f"  Rows retained      : {actual_rows/total_original*100:.4f}%")

# Trip duration stats
sub("Trip Duration After Cleaning (minutes)")
dur = con.execute(f"""
    SELECT
        MIN(trip_duration_minutes)                                  AS min,
        PERCENTILE_CONT(0.5) WITHIN GROUP
            (ORDER BY trip_duration_minutes)                        AS median,
        AVG(trip_duration_minutes)                                  AS mean,
        MAX(trip_duration_minutes)                                  AS max
    FROM read_parquet({out})
""").fetchone()
print(f"  Min    : {dur[0]:.3f} min")
print(f"  Median : {dur[1]:.3f} min")
print(f"  Mean   : {dur[2]:.3f} min")
print(f"  Max    : {dur[3]:.3f} min")

# Trip distance stats
sub("Trip Distance After Cleaning (miles)")
dist = con.execute(f"""
    SELECT
        MIN(trip_distance)                                          AS min,
        PERCENTILE_CONT(0.5) WITHIN GROUP
            (ORDER BY trip_distance)                                AS median,
        AVG(trip_distance)                                          AS mean,
        MAX(trip_distance)                                          AS max
    FROM read_parquet({out})
""").fetchone()
print(f"  Min    : {dist[0]:.3f} mi")
print(f"  Median : {dist[1]:.3f} mi")
print(f"  Mean   : {dist[2]:.3f} mi")
print(f"  Max    : {dist[3]:.3f} mi")

# Column list and null counts
sub("Final Column List & Null Counts")
schema = con.execute(
    f"DESCRIBE SELECT * FROM read_parquet({out}) LIMIT 0"
).fetchall()
col_names = [r[0] for r in schema]

null_exprs = ", ".join(
    f'COUNT(*) FILTER (WHERE "{c}" IS NULL) AS "{c}"'
    for c in col_names
)
null_row = con.execute(
    f"SELECT {null_exprs} FROM read_parquet({out})"
).fetchone()

print(f"\n  {'#':<4} {'Column':<26} {'Type':<12} {'Nulls':>8}  Role")
print(f"  {'-'*3:<4} {'-'*25:<26} {'-'*11:<12} {'-'*8:>8}  {'----'}")
roles = {
    "trip_duration_minutes": "TARGET",
    "pickup_year":           "feature (time)",
    "pickup_month":          "feature (time)",
    "pickup_day":            "feature (time)",
    "pickup_day_of_week":    "feature (time)",
    "pickup_hour":           "feature (time)",
    "is_weekend":            "feature (time)",
    "trip_distance":         "feature",
    "is_long_haul":          "feature (flag)",
    "passenger_count":       "feature",
    "RatecodeID":            "feature",
    "VendorID":              "feature",
    "PULocationID":          "feature",
    "DOLocationID":          "feature",
}
for i, (row, nulls) in enumerate(zip(schema, null_row), 1):
    col, dtype = row[0], row[1]
    role = roles.get(col, "")
    flag = "  [!!]" if nulls > 0 else ""
    print(f"  {i:<4} {col:<26} {dtype:<12} {nulls:>8}{flag}  {role}")

# is_long_haul distribution
sub("is_long_haul Distribution")
lh = con.execute(f"""
    SELECT is_long_haul, COUNT(*) AS n
    FROM read_parquet({out})
    GROUP BY is_long_haul ORDER BY is_long_haul
""").fetchall()
for val, cnt in lh:
    label = "long haul" if val == 1 else "standard"
    print(f"  {val} ({label:<10}) : {cnt:>10,}  ({cnt/actual_rows*100:.4f}%)")

# File size
sub("Output File")
fsize_mb = OUT_PATH.stat().st_size / (1024 ** 2)
print(f"  Path : {OUT_PATH}")
print(f"  Size : {fsize_mb:.1f} MB")

print(f"\n{SEP}")
print("  Cleaning complete. Source files were NOT modified.")
print(f"  Output : {OUT_PATH.name}")
print(SEP)
