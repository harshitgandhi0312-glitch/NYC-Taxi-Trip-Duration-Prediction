"""
src/create_splits.py
--------------------
Temporal train/validation/test split for NYC Taxi Trip Duration Prediction.
Input: data/processed/cleaned_taxi_2023_2024.parquet
Outputs:
  - data/processed/train.parquet
  - data/processed/validation.parquet
  - data/processed/test.parquet

Run:
    python src/create_splits.py
"""

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.db import get_connection

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "data" / "processed" / "cleaned_taxi_2023_2024.parquet"
TRAIN_PATH = ROOT / "data" / "processed" / "train.parquet"
VAL_PATH = ROOT / "data" / "processed" / "validation.parquet"
TEST_PATH = ROOT / "data" / "processed" / "test.parquet"

def main():
    con = get_connection()
    p = f"['{IN_PATH.as_posix()}']"

    print("=" * 72)
    print("  NYC Taxi - Creating Temporal Splits")
    print("=" * 72)
    
    # 1. Total rows
    total_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet({p})").fetchone()[0]
    print(f"\nTotal rows in cleaned dataset: {total_rows:,}")
    
    # 2. Define splits
    # Train: 2023
    # Val: 2024 Jan-Jun
    # Test: 2024 Jul-Dec
    
    print("\nCreating train.parquet (2023)...")
    t0 = time.time()
    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet({p})
            WHERE pickup_year = 2023
        ) TO '{TRAIN_PATH.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print(f"Done in {time.time() - t0:.1f}s")
    
    print("Creating validation.parquet (2024 H1)...")
    t0 = time.time()
    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet({p})
            WHERE pickup_year = 2024 AND pickup_month <= 6
        ) TO '{VAL_PATH.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print(f"Done in {time.time() - t0:.1f}s")
    
    print("Creating test.parquet (2024 H2)...")
    t0 = time.time()
    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet({p})
            WHERE pickup_year = 2024 AND pickup_month >= 7
        ) TO '{TEST_PATH.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print(f"Done in {time.time() - t0:.1f}s")
    
    print("\nSplits created successfully. Beginning validation...")
    
    # 3. Validation
    train_c = f"['{TRAIN_PATH.as_posix()}']"
    val_c = f"['{VAL_PATH.as_posix()}']"
    test_c = f"['{TEST_PATH.as_posix()}']"
    
    def get_stats(dataset_name, path_str):
        row = con.execute(f"""
            SELECT 
                COUNT(*) as row_count,
                MIN(MAKE_TIMESTAMP(pickup_year, pickup_month, pickup_day, pickup_hour, 0, 0)) as min_dt,
                MAX(MAKE_TIMESTAMP(pickup_year, pickup_month, pickup_day, pickup_hour, 59, 59)) as max_dt,
                MIN(trip_duration_minutes) as min_dur,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY trip_duration_minutes) as med_dur,
                AVG(trip_duration_minutes) as mean_dur,
                MAX(trip_duration_minutes) as max_dur,
                MIN(trip_distance) as min_dist,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY trip_distance) as med_dist,
                AVG(trip_distance) as mean_dist,
                MAX(trip_distance) as max_dist
            FROM read_parquet({path_str})
        """).fetchone()
        
        # null count
        columns = [c[0] for c in con.execute(f"DESCRIBE SELECT * FROM read_parquet({path_str}) LIMIT 0").fetchall()]
        null_conds = " + ".join([f"SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END)" for c in columns])
        nulls = con.execute(f"SELECT {null_conds} FROM read_parquet({path_str})").fetchone()[0]
        
        # duplicate count (exact matches across all columns)
        cols_str = ", ".join(columns)
        dups = con.execute(f"""
            SELECT SUM(cnt - 1) FROM (
                SELECT {cols_str}, COUNT(*) as cnt
                FROM read_parquet({path_str})
                GROUP BY {cols_str}
                HAVING COUNT(*) > 1
            )
        """).fetchone()[0]
        
        if dups is None:
            dups = 0
            
        return {
            "name": dataset_name,
            "rows": row[0],
            "min_dt": row[1],
            "max_dt": row[2],
            "min_dur": row[3],
            "med_dur": row[4],
            "mean_dur": row[5],
            "max_dur": row[6],
            "min_dist": row[7],
            "med_dist": row[8],
            "mean_dist": row[9],
            "max_dist": row[10],
            "nulls": nulls,
            "dups": dups
        }

    stats_train = get_stats("Train", train_c)
    stats_val = get_stats("Validation", val_c)
    stats_test = get_stats("Test", test_c)
    
    # Print stats
    print("\n" + "=" * 72)
    print("  SPLIT VALIDATION REPORT")
    print("=" * 72)
    
    for s in [stats_train, stats_val, stats_test]:
        print(f"\n--- {s['name']} ---")
        print(f"Row count       : {s['rows']:,} ({(s['rows'] / total_rows * 100):.2f}%)")
        print(f"Min pickup dt   : {s['min_dt']}")
        print(f"Max pickup dt   : {s['max_dt']}")
        print(f"Duration (min)  : Min={s['min_dur']:.3f}, Median={s['med_dur']:.3f}, Mean={s['mean_dur']:.3f}, Max={s['max_dur']:.3f}")
        print(f"Distance (miles): Min={s['min_dist']:.3f}, Median={s['med_dist']:.3f}, Mean={s['mean_dist']:.3f}, Max={s['max_dist']:.3f}")
        print(f"Null count      : {s['nulls']}")
        print(f"Duplicate rows  : {s['dups']:,}")
        
    print("\n--- Overlap & Integrity Checks ---")
    
    # Assert row counts
    sum_rows = stats_train['rows'] + stats_val['rows'] + stats_test['rows']
    if sum_rows != total_rows:
        raise ValueError(f"CRITICAL ERROR: Sum of splits ({sum_rows:,}) does NOT equal total rows ({total_rows:,}).")
    print(f"[OK] Sum of split rows ({sum_rows:,}) matches total rows ({total_rows:,}).")
    
    # Assert temporal ordering
    if stats_train['max_dt'] >= stats_val['min_dt']:
        raise ValueError(f"CRITICAL ERROR: Train data max ({stats_train['max_dt']}) overlaps with Validation data min ({stats_val['min_dt']}).")
    print(f"[OK] Train data strictly occurs before Validation data.")
    
    if stats_val['max_dt'] >= stats_test['min_dt']:
        raise ValueError(f"CRITICAL ERROR: Validation data max ({stats_val['max_dt']}) overlaps with Test data min ({stats_test['min_dt']}).")
    print(f"[OK] Validation data strictly occurs before Test data.")
    
    print("\nAll validations passed successfully.")

if __name__ == "__main__":
    main()
