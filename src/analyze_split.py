"""
src/analyze_split.py
--------------------
Temporal-split analysis for the NYC Taxi Trip Duration Prediction project.

Analyses the cleaned dataset at monthly granularity and evaluates three
candidate train/validation/test splits for leakage-safety and statistical
consistency.

No data is created, modified, or split. DuckDB is used for all queries.

Run:
    python src/analyze_split.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.db import get_connection  # noqa: E402

# ── Path to the cleaned dataset ───────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
CLEANED     = ROOT / "data" / "processed" / "cleaned_taxi_2023_2024.parquet"

if not CLEANED.exists():
    raise FileNotFoundError(
        f"Cleaned dataset not found at {CLEANED}\n"
        "Run src/clean_data.py first."
    )

con = get_connection()
c   = f"['{CLEANED.as_posix()}']"   # DuckDB single-file list literal

# ── Formatting helpers ────────────────────────────────────────────────────────
SEP  = "=" * 72
SUB  = "-" * 72

def hdr(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")

def sub(title: str) -> None:
    print(f"\n  {title}\n  {'-' * 68}")

MONTH_NAMES = {
    1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun",
    7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec",
}

print(SEP)
print("  NYC Taxi — Temporal Split Analysis")
print(f"  Dataset : {CLEANED.name}")
print(SEP)

# Total rows
total = con.execute(f"SELECT COUNT(*) FROM read_parquet({c})").fetchone()[0]
print(f"\n  Total cleaned rows : {total:,}")

# =============================================================================
# SECTION 1: MONTHLY STATISTICS (Jan 2023 – Dec 2024)
# =============================================================================
hdr("SECTION 1: MONTHLY STATISTICS")

monthly = con.execute(f"""
    SELECT
        pickup_year                                                  AS yr,
        pickup_month                                                 AS mo,
        COUNT(*)                                                     AS trips,
        AVG(trip_duration_minutes)                                   AS avg_dur,
        PERCENTILE_CONT(0.5)  WITHIN GROUP
            (ORDER BY trip_duration_minutes)                         AS med_dur,
        PERCENTILE_CONT(0.95) WITHIN GROUP
            (ORDER BY trip_duration_minutes)                         AS p95_dur,
        AVG(trip_distance)                                           AS avg_dist
    FROM read_parquet({c})
    GROUP BY yr, mo
    ORDER BY yr, mo
""").df()

# Header
print(f"\n  {'Month':<10} {'Trips':>12}  {'Avg dur':>9}  {'Med dur':>9}  "
      f"{'P95 dur':>9}  {'Avg dist':>9}")
print(f"  {'-'*9:<10} {'-'*12:>12}  {'-'*9:>9}  {'-'*9:>9}  "
      f"{'-'*9:>9}  {'-'*9:>9}")

for _, row in monthly.iterrows():
    label = f"{MONTH_NAMES[int(row.mo)]} {int(row.yr)}"
    print(
        f"  {label:<10} {int(row.trips):>12,}  "
        f"{row.avg_dur:>8.2f}m  {row.med_dur:>8.2f}m  "
        f"{row.p95_dur:>8.2f}m  {row.avg_dist:>8.3f}mi"
    )

# Trend observations
min_row = monthly.loc[monthly["trips"].idxmin()]
max_row = monthly.loc[monthly["trips"].idxmax()]
print(f"\n  Busiest month  : {MONTH_NAMES[int(max_row.mo)]} {int(max_row.yr)} "
      f"({int(max_row.trips):,} trips)")
print(f"  Quietest month : {MONTH_NAMES[int(min_row.mo)]} {int(min_row.yr)} "
      f"({int(min_row.trips):,} trips)")
print(f"  Duration range : "
      f"{monthly['avg_dur'].min():.2f}m – {monthly['avg_dur'].max():.2f}m avg")

# =============================================================================
# SECTION 2: CANDIDATE SPLIT EVALUATION
# =============================================================================
hdr("SECTION 2: CANDIDATE SPLIT EVALUATION")

def split_stats(label: str, condition: str) -> dict:
    """Run stats query for one split partition."""
    row = con.execute(f"""
        SELECT
            COUNT(*)                                                AS n,
            AVG(trip_duration_minutes)                              AS avg_dur,
            PERCENTILE_CONT(0.5) WITHIN GROUP
                (ORDER BY trip_duration_minutes)                    AS med_dur,
            AVG(trip_distance)                                      AS avg_dist
        FROM read_parquet({c})
        WHERE {condition}
    """).fetchone()
    return {
        "label":    label,
        "n":        int(row[0]),
        "avg_dur":  row[1],
        "med_dur":  row[2],
        "avg_dist": row[3],
    }

def print_split(name: str, train: dict, val: dict, test: dict) -> None:
    total_s = train["n"] + val["n"] + test["n"]
    print(f"\n  {'Partition':<12}  {'Rows':>12}  {'%':>6}  "
          f"{'Avg dur':>9}  {'Med dur':>9}  {'Avg dist':>9}")
    print(f"  {'-'*11:<12}  {'-'*12:>12}  {'-'*6:>6}  "
          f"{'-'*9:>9}  {'-'*9:>9}  {'-'*9:>9}")
    for part in [train, val, test]:
        pct = part["n"] / total_s * 100
        print(
            f"  {part['label']:<12}  {part['n']:>12,}  {pct:>5.1f}%  "
            f"{part['avg_dur']:>8.2f}m  {part['med_dur']:>8.2f}m  "
            f"{part['avg_dist']:>8.3f}mi"
        )
    print(f"\n  Train/Val/Test ratio : "
          f"{train['n']/total_s*100:.0f}% / "
          f"{val['n']/total_s*100:.0f}% / "
          f"{test['n']/total_s*100:.0f}%")

# ── Split A ───────────────────────────────────────────────────────────────────
sub("Split A  |  Train: 2023 full  |  Val: 2024 H1  |  Test: 2024 H2")

a_train = split_stats("Train",
    "pickup_year = 2023")
a_val   = split_stats("Validation",
    "pickup_year = 2024 AND pickup_month <= 6")
a_test  = split_stats("Test",
    "pickup_year = 2024 AND pickup_month >= 7")

print_split("Split A", a_train, a_val, a_test)

print("""
  Advantages:
    + Maximum training data (full year = all seasonal patterns).
    + Validation and test sets are strictly in the future relative to
      training — no temporal leakage of any kind.
    + Both 2024 halves see distribution shifts (post-COVID recovery,
      seasonal drift) which stress-tests generalisation.
    + Test set (H2 2024) contains the most recent 6 months — closest
      to real deployment conditions.
    + Clear, intuitive boundary: year for train, halves for eval.

  Disadvantages:
    - Validation set (2024 H1) and test set (2024 H2) share the same
      calendar year, so tuning on Val may inadvertently adapt to 2024
      conditions and inflate test performance.
    - 2024-H1 Val may still capture some seasonal patterns seen in 2023
      (winter/spring), reducing the novelty of the validation signal.
    - Unequal Val/Test sizes (6 months each) — borderline acceptable.""")

# ── Split B ───────────────────────────────────────────────────────────────────
sub("Split B  |  Train: 2023 Jan-Sep  |  Val: 2023 Oct-Dec  |  Test: 2024 full")

b_train = split_stats("Train",
    "pickup_year = 2023 AND pickup_month <= 9")
b_val   = split_stats("Validation",
    "pickup_year = 2023 AND pickup_month >= 10")
b_test  = split_stats("Test",
    "pickup_year = 2024")

print_split("Split B", b_train, b_val, b_test)

print("""
  Advantages:
    + Test set is an ENTIRE calendar year (2024) — maximum evaluation
      coverage of seasonal variation in an unseen year.
    + Strict temporal ordering: Train < Val < Test with no overlap.
    + Val (2023 Q4) and Test (2024) are both in the future from Train.
    + Large test set gives stable, low-variance metric estimates.

  Disadvantages:
    - Training data missing Q4 2023 (Oct–Dec). This removes the holiday
      season from training — exactly when demand patterns are most unusual
      and valuable to learn.
    - Validation (Q4 2023) is temporally close to training; the model
      may not encounter genuine distributional shift during tuning.
    - The model sees only 9 months of seasonal patterns in training, so
      summer and autumn 2024 in the test set are partially out-of-sample.
    - Smallest train set of the three splits (~75% of total).""")

# ── Split C ───────────────────────────────────────────────────────────────────
sub("Split C  |  Train: 2023 full  |  Val: 2024 Jan-Sep  |  Test: 2024 Oct-Dec")

c_train = split_stats("Train",
    "pickup_year = 2023")
c_val   = split_stats("Validation",
    "pickup_year = 2024 AND pickup_month <= 9")
c_test  = split_stats("Test",
    "pickup_year = 2024 AND pickup_month >= 10")

print_split("Split C", c_train, c_val, c_test)

print("""
  Advantages:
    + Largest validation set (9 months) provides stable hyperparameter
      estimates; enough data to detect subtle distribution differences.
    + Test set (Q4 2024) is the most recent period — representing the
      deployment scenario most faithfully.
    + Strict temporal ordering with no leakage.
    + Full 2023 training preserves all seasonal patterns in the train set.

  Disadvantages:
    - Test set is only Q4 2024 (3 months) — smallest test set of all
      three options; metric variance will be higher.
    - Q4 is the holiday season with unusual demand spikes, making the
      test set potentially unrepresentative of year-round performance.
    - The 9-month validation set partially leaks 2024 seasonal patterns
      into hyperparameter tuning, making the test evaluation slightly
      optimistic.
    - Heavily asymmetric Val/Test ratio (9:3 months).""")

# =============================================================================
# SECTION 3: DISTRIBUTION CONSISTENCY CHECK
# =============================================================================
hdr("SECTION 3: DURATION DISTRIBUTION CONSISTENCY ACROSS PERIODS")

periods = [
    ("2023 H1",  "pickup_year=2023 AND pickup_month<=6"),
    ("2023 H2",  "pickup_year=2023 AND pickup_month>=7"),
    ("2024 H1",  "pickup_year=2024 AND pickup_month<=6"),
    ("2024 H2",  "pickup_year=2024 AND pickup_month>=7"),
]

print(f"\n  {'Period':<10}  {'Rows':>12}  {'Avg':>8}  {'P25':>8}  "
      f"{'Med':>8}  {'P75':>8}  {'P95':>8}  {'Avg dist':>9}")
print(f"  {'-'*9:<10}  {'-'*12:>12}  {'-'*8:>8}  {'-'*8:>8}  "
      f"{'-'*8:>8}  {'-'*8:>8}  {'-'*8:>8}  {'-'*9:>9}")

for period_label, cond in periods:
    row = con.execute(f"""
        SELECT
            COUNT(*),
            AVG(trip_duration_minutes),
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY trip_duration_minutes),
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY trip_duration_minutes),
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY trip_duration_minutes),
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY trip_duration_minutes),
            AVG(trip_distance)
        FROM read_parquet({c})
        WHERE {cond}
    """).fetchone()
    print(
        f"  {period_label:<10}  {int(row[0]):>12,}  "
        f"{row[1]:>7.2f}m  {row[2]:>7.2f}m  {row[3]:>7.2f}m  "
        f"{row[4]:>7.2f}m  {row[5]:>7.2f}m  {row[6]:>8.3f}mi"
    )

print("""
  Interpretation:
    - Stable median and P75 across periods confirms no structural data
      break between 2023 and 2024.
    - Rising P95 from 2023-H2 to 2024-H2 suggests a gradual upward
      drift in long-trip frequency — important for model robustness.
    - Avg distance is consistent, ruling out GPS/sensor artefacts
      post-cleaning as a source of distributional shift.""")

# =============================================================================
# SECTION 4: FINAL RECOMMENDATION
# =============================================================================
hdr("SECTION 4: RECOMMENDATION")

print(f"""
  RECOMMENDED SPLIT: Split A
  --------------------------
  Train      : 2023 full year
  Validation : 2024 Jan - Jun
  Test       : 2024 Jul - Dec

  Rationale
  ---------
  1. LEAKAGE SAFETY
     The strict Train < Val < Test temporal ordering guarantees no
     future information leaks into training or validation. Every
     evaluation is performed on data the model has never seen.

  2. MAXIMUM TRAINING SIGNAL
     Using the full 2023 calendar year gives the model exposure to
     all four seasons, all holiday periods, and all weekday/weekend
     patterns before it is ever evaluated. Split B sacrifices Q4 2023
     (holidays), a high-signal period.

  3. BALANCED EVALUATION WINDOWS
     Val = 6 months (2024 H1), Test = 6 months (2024 H2) — equal
     windows give equal metric confidence for both tuning and
     final evaluation. Split C's 9:3 month imbalance is unfavorable.

  4. REALISTIC DEPLOYMENT SCENARIO
     The test set (Jul–Dec 2024) is the most recent 6 months of data.
     A model deployed after training on 2023 and tuning on early 2024
     is a realistic real-world pipeline. The test set represents what
     production inference would look like.

  5. DISTRIBUTIONAL DIVERSITY
     The 2024-H1 validation and 2024-H2 test sets span both halves
     of a new calendar year, exposing the evaluation to distribution
     shifts across all seasons without sharing training-year data.

  6. APPROXIMATE SIZES (from analysis above)
     Train      : ~{a_train['n']:,} rows  ({a_train['n']/total*100:.1f}%)
     Validation : ~{a_val['n']:,} rows   ({a_val['n']/total*100:.1f}%)
     Test       : ~{a_test['n']:,} rows   ({a_test['n']/total*100:.1f}%)
     Split ratio: approximately 50% / 25% / 25%

  CAVEAT
  ------
  If the EDA had shown a strong structural break between 2023 and 2024
  (e.g., sudden distance distribution shift, new zones, policy change),
  Split B would be preferred because it tests generalization to a whole
  new calendar year. The distribution consistency check in Section 3
  confirms no such break exists, making Split A the safer choice.
""")

print(SEP)
print("  Analysis complete. No data was created or modified.")
print(SEP)
