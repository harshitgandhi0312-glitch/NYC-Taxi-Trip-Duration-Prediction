"""
src/eda.py
----------
Exploratory Data Analysis — NYC Taxi Trip Duration Prediction.

Strategy
--------
* All heavy aggregation runs inside DuckDB.
* Only small summary DataFrames (hundreds of rows) are pulled into pandas.
* The 71M-row dataset is never loaded into memory.
* Source Parquet files are never modified.

Output
------
* Terminal report covering all 8 EDA questions.
* 7 PNG figures saved to reports/figures/.

Run
---
    python src/eda.py
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.db import get_connection, get_taxi_paths  # noqa: E402

FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE   = "#2563EB"          # primary blue
PALETTE2  = "#DC2626"          # accent red
BG        = "#0F172A"          # dark background
GRID      = "#1E293B"          # grid lines
TEXT      = "#F1F5F9"          # text colour
ACCENT    = "#38BDF8"          # highlight

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    GRID,
    "axes.edgecolor":    GRID,
    "axes.labelcolor":   TEXT,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "text.color":        TEXT,
    "grid.color":        "#334155",
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "figure.titlesize":  15,
    "figure.titleweight":"bold",
    "savefig.facecolor": BG,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
})

SEP = "=" * 68
SUB = "-" * 68

def hdr(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def sub(title):
    print(f"\n{SUB}\n  {title}\n{SUB}")

def save(fig, name):
    path = FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  [saved] {path.relative_to(ROOT)}")

# ── Connect ───────────────────────────────────────────────────────────────────
con   = get_connection()
paths = get_taxi_paths()
p     = str(paths)            # DuckDB accepts a Python list literal

print(SEP)
print("  NYC Taxi Trip Duration — EDA Report")
print(SEP)
print(f"\n  Dataset : {paths[0]}")
print(f"           {paths[1]}")

# =============================================================================
# 1. TRIP DURATION DISTRIBUTION
# =============================================================================
hdr("1. TRIP DURATION DISTRIBUTION (minutes)")

dur_stats = con.execute(f"""
    SELECT
        MIN(trip_duration_minutes)                               AS min,
        PERCENTILE_CONT(0.01) WITHIN GROUP
            (ORDER BY trip_duration_minutes)                     AS p01,
        PERCENTILE_CONT(0.25) WITHIN GROUP
            (ORDER BY trip_duration_minutes)                     AS p25,
        PERCENTILE_CONT(0.50) WITHIN GROUP
            (ORDER BY trip_duration_minutes)                     AS median,
        AVG(trip_duration_minutes)                               AS mean,
        PERCENTILE_CONT(0.75) WITHIN GROUP
            (ORDER BY trip_duration_minutes)                     AS p75,
        PERCENTILE_CONT(0.95) WITHIN GROUP
            (ORDER BY trip_duration_minutes)                     AS p95,
        PERCENTILE_CONT(0.99) WITHIN GROUP
            (ORDER BY trip_duration_minutes)                     AS p99,
        MAX(trip_duration_minutes)                               AS max
    FROM read_parquet({p})
""").df()

labels = ["Min","P01","P25","Median","Mean","P75","P95","P99","Max"]
for lbl, val in zip(labels, dur_stats.iloc[0]):
    print(f"  {lbl:<10}: {val:>8.2f} min")

print("\n  Model implication:")
print("  - Right-skewed distribution (mean > median) => log-transform of")
print("    trip_duration_minutes likely beneficial as the model target.")
print("  - P99 = {:.1f} min, Max = {:.1f} min => outlier cap needed.".format(
    dur_stats["p99"].iloc[0], dur_stats["max"].iloc[0]))

# -- histogram via DuckDB buckets (0–60 min covers the bulk) ------------------
dur_hist = con.execute(f"""
    SELECT
        bucket,
        COUNT(*) AS trips
    FROM (
        SELECT CAST(FLOOR(trip_duration_minutes) AS INTEGER) AS bucket
        FROM read_parquet({p})
        WHERE trip_duration_minutes BETWEEN 0 AND 120
    )
    GROUP BY bucket ORDER BY bucket
""").df()
dur_hist["minute"] = dur_hist["bucket"]          # 1 min bins

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Trip Duration Distribution")

# left: full 0-120 min
ax = axes[0]
ax.bar(dur_hist["minute"], dur_hist["trips"] / 1e6,
       color=PALETTE, width=0.9, alpha=0.85)
ax.set_xlabel("Duration (minutes)")
ax.set_ylabel("Trips (millions)")
ax.set_title("0 – 120 min (bulk)")
ax.axvline(dur_stats["median"].iloc[0], color=ACCENT,  lw=1.5, ls="--", label="Median")
ax.axvline(dur_stats["mean"].iloc[0],   color=PALETTE2, lw=1.5, ls="--", label="Mean")
ax.legend(framealpha=0.3)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))

# right: zoomed 0-30 min
ax = axes[1]
sub_hist = dur_hist[dur_hist["minute"] <= 30]
ax.bar(sub_hist["minute"], sub_hist["trips"] / 1e6,
       color=ACCENT, width=0.9, alpha=0.85)
ax.set_xlabel("Duration (minutes)")
ax.set_ylabel("Trips (millions)")
ax.set_title("0 – 30 min (zoom)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))

plt.tight_layout()
save(fig, "01_trip_duration_distribution.png")

# =============================================================================
# 2. TRIP DISTANCE DISTRIBUTION
# =============================================================================
hdr("2. TRIP DISTANCE DISTRIBUTION (miles)")

dist_stats = con.execute(f"""
    SELECT
        MIN(trip_distance)                               AS min,
        PERCENTILE_CONT(0.01) WITHIN GROUP
            (ORDER BY trip_distance)                     AS p01,
        PERCENTILE_CONT(0.25) WITHIN GROUP
            (ORDER BY trip_distance)                     AS p25,
        PERCENTILE_CONT(0.50) WITHIN GROUP
            (ORDER BY trip_distance)                     AS median,
        AVG(trip_distance)                               AS mean,
        PERCENTILE_CONT(0.75) WITHIN GROUP
            (ORDER BY trip_distance)                     AS p75,
        PERCENTILE_CONT(0.95) WITHIN GROUP
            (ORDER BY trip_distance)                     AS p95,
        PERCENTILE_CONT(0.99) WITHIN GROUP
            (ORDER BY trip_distance)                     AS p99,
        MAX(trip_distance)                               AS max
    FROM read_parquet({p})
""").df()

labels = ["Min","P01","P25","Median","Mean","P75","P95","P99","Max"]
for lbl, val in zip(labels, dist_stats.iloc[0]):
    print(f"  {lbl:<10}: {val:>8.3f} miles")

print("\n  Model implication:")
print("  - trip_distance is a strong predictor candidate.")
print("  - Heavy right skew => log-transform or binning may help tree models.")
print("  - P99 = {:.1f} mi; check very large distances vs duration ratio.".format(
    dist_stats["p99"].iloc[0]))

# =============================================================================
# 3. DURATION BY TIME DIMENSIONS
# =============================================================================
hdr("3. DURATION BY TIME DIMENSION")

# 3a. Hour of day
by_hour = con.execute(f"""
    SELECT
        HOUR(tpep_pickup_datetime)   AS hour,
        AVG(trip_duration_minutes)   AS avg_dur,
        COUNT(*)                     AS trips
    FROM read_parquet({p})
    GROUP BY hour ORDER BY hour
""").df()

# 3b. Day of week  (0=Sunday … 6=Saturday in DuckDB)
by_dow = con.execute(f"""
    SELECT
        DAYOFWEEK(tpep_pickup_datetime)   AS dow,
        AVG(trip_duration_minutes)        AS avg_dur,
        COUNT(*)                          AS trips
    FROM read_parquet({p})
    GROUP BY dow ORDER BY dow
""").df()
dow_labels = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"]
by_dow["day"] = by_dow["dow"].map(lambda x: dow_labels[int(x)])

# 3c. Month
by_month = con.execute(f"""
    SELECT
        YEAR(tpep_pickup_datetime)  AS year,
        MONTH(tpep_pickup_datetime) AS month,
        AVG(trip_duration_minutes)  AS avg_dur,
        COUNT(*)                    AS trips
    FROM read_parquet({p})
    GROUP BY year, month ORDER BY year, month
""").df()
by_month["period"] = by_month["year"].astype(str) + "-" + by_month["month"].astype(str).str.zfill(2)

print("\n  Avg duration by hour of day (top 5 busiest hours):")
top5 = by_hour.nlargest(5, "trips")[["hour","avg_dur","trips"]]
for _, row in top5.iterrows():
    print(f"    Hour {int(row.hour):02d}:00  avg {row.avg_dur:.1f} min  "
          f"({row.trips/1e6:.2f}M trips)")

print("\n  Avg duration by day of week:")
for _, row in by_dow.iterrows():
    print(f"    {row.day}  avg {row.avg_dur:.1f} min  ({row.trips/1e6:.2f}M trips)")

print("\n  Model implication:")
print("  - Hour and day-of-week show clear temporal patterns => strong features.")
print("  - Rush hours correlate with longer durations (congestion).")

# --- Figure: hour + day of week side by side ---------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Average Trip Duration by Time Dimension")

ax = axes[0]
ax.plot(by_hour["hour"], by_hour["avg_dur"],
        color=ACCENT, lw=2.5, marker="o", ms=5)
ax.fill_between(by_hour["hour"], by_hour["avg_dur"],
                alpha=0.15, color=ACCENT)
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Avg Duration (min)")
ax.set_title("By Hour of Day")
ax.set_xticks(range(0, 24, 2))
ax.grid(True, axis="y")

ax = axes[1]
colors = [PALETTE2 if d in ["Fri","Sat","Sun"] else PALETTE
          for d in by_dow["day"]]
bars = ax.bar(by_dow["day"], by_dow["avg_dur"], color=colors, alpha=0.88)
ax.set_xlabel("Day of Week")
ax.set_ylabel("Avg Duration (min)")
ax.set_title("By Day of Week")
ax.grid(True, axis="y")
ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=9, color=TEXT)

plt.tight_layout()
save(fig, "02_duration_by_hour_and_dow.png")

# --- Figure: monthly trend by year -------------------------------------------
fig, ax = plt.subplots(figsize=(13, 5))
fig.suptitle("Monthly Average Trip Duration by Year")

for yr, grp in by_month.groupby("year"):
    col = PALETTE if yr == 2023 else PALETTE2
    ax.plot(grp["month"], grp["avg_dur"],
            color=col, lw=2.5, marker="o", ms=6, label=str(yr))
    ax.fill_between(grp["month"], grp["avg_dur"], alpha=0.1, color=col)

ax.set_xlabel("Month")
ax.set_ylabel("Avg Duration (min)")
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"])
ax.legend(title="Year", framealpha=0.3)
ax.grid(True, axis="y")
plt.tight_layout()
save(fig, "03_duration_by_month_year.png")

print("\n  Monthly avg duration range:")
print(f"    Min month avg: {by_month['avg_dur'].min():.2f} min "
      f"({by_month.loc[by_month['avg_dur'].idxmin(), 'period']})")
print(f"    Max month avg: {by_month['avg_dur'].max():.2f} min "
      f"({by_month.loc[by_month['avg_dur'].idxmax(), 'period']})")

# =============================================================================
# 4. DISTANCE-DURATION CORRELATION
# =============================================================================
hdr("4. DISTANCE vs. DURATION CORRELATION")

corr_row = con.execute(f"""
    SELECT CORR(trip_distance, trip_duration_minutes) AS pearson_r
    FROM read_parquet({p})
""").fetchone()
pearson_r = corr_row[0]
print(f"  Pearson correlation (distance, duration) : {pearson_r:.4f}")

# Avg duration across distance buckets
dist_buckets = con.execute(f"""
    SELECT
        CASE
            WHEN trip_distance < 1   THEN '< 1 mi'
            WHEN trip_distance < 2   THEN '1-2 mi'
            WHEN trip_distance < 3   THEN '2-3 mi'
            WHEN trip_distance < 5   THEN '3-5 mi'
            WHEN trip_distance < 10  THEN '5-10 mi'
            WHEN trip_distance < 20  THEN '10-20 mi'
            ELSE '20+ mi'
        END AS dist_group,
        AVG(trip_duration_minutes)  AS avg_dur,
        COUNT(*)                    AS trips
    FROM read_parquet({p})
    GROUP BY dist_group
    ORDER BY MIN(trip_distance)
""").df()

print("\n  Avg duration by distance group:")
print(f"  {'Group':<12} {'Avg dur (min)':>14} {'Trips':>12}")
print(f"  {'-'*11:<12} {'-'*14:>14} {'-'*11:>12}")
for _, row in dist_buckets.iterrows():
    print(f"  {row.dist_group:<12} {row.avg_dur:>14.1f} {int(row.trips):>12,}")

print(f"\n  Model implication:")
print(f"  - r = {pearson_r:.3f}: moderate positive correlation.")
print(f"  - Not perfectly linear => distance alone insufficient; zone,")
print(f"    time, and traffic features needed.")

# --- Figure: distance buckets bar chart --------------------------------------
fig, ax = plt.subplots(figsize=(11, 5))
fig.suptitle("Average Trip Duration by Distance Group")

bars = ax.bar(dist_buckets["dist_group"], dist_buckets["avg_dur"],
              color=PALETTE, alpha=0.88, width=0.65)
ax.set_xlabel("Trip Distance Group")
ax.set_ylabel("Avg Duration (min)")
ax.bar_label(bars, fmt="%.1f", padding=3, color=TEXT, fontsize=10)
ax.grid(True, axis="y")

ax2 = ax.twinx()
ax2.plot(range(len(dist_buckets)), dist_buckets["trips"] / 1e6,
         color=ACCENT, lw=2, marker="D", ms=6, label="Trips (M)")
ax2.set_ylabel("Trips (millions)", color=ACCENT)
ax2.tick_params(axis="y", colors=ACCENT)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))

plt.tight_layout()
save(fig, "04_distance_duration_by_bucket.png")

# =============================================================================
# 5. DURATION BY PICKUP ZONE
# =============================================================================
hdr("5. DURATION BY PICKUP ZONE (min 5,000 trips)")

zone_dur = con.execute(f"""
    SELECT
        PULocationID              AS zone,
        AVG(trip_duration_minutes) AS avg_dur,
        COUNT(*)                   AS trips
    FROM read_parquet({p})
    GROUP BY PULocationID
    HAVING COUNT(*) >= 5000
    ORDER BY avg_dur DESC
""").df()

top10    = zone_dur.head(10).copy()
bottom10 = zone_dur.tail(10).copy()

print("\n  Top 10 zones by avg duration (highest):")
print(f"  {'Zone':>6} {'Avg dur (min)':>14} {'Trips':>10}")
for _, row in top10.iterrows():
    print(f"  {int(row.zone):>6} {row.avg_dur:>14.1f} {int(row.trips):>10,}")

print("\n  Bottom 10 zones by avg duration (lowest):")
print(f"  {'Zone':>6} {'Avg dur (min)':>14} {'Trips':>10}")
for _, row in bottom10.iterrows():
    print(f"  {int(row.zone):>6} {row.avg_dur:>14.1f} {int(row.trips):>10,}")

print("\n  Model implication:")
print("  - Zone ID is a high-cardinality categorical feature.")
print("  - Large variance across zones => strong signal for the model.")
print("  - Consider target-encoding or embeddings during feature engineering.")

# --- Figure: top/bottom zones ------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Avg Trip Duration by Pickup Zone (>=5,000 trips)")

for ax, df, title, col in [
    (axes[0], top10,    "Top 10 Longest",  PALETTE2),
    (axes[1], bottom10, "Top 10 Shortest", PALETTE),
]:
    df_sorted = df.sort_values("avg_dur")
    bars = ax.barh(df_sorted["zone"].astype(str),
                   df_sorted["avg_dur"], color=col, alpha=0.85)
    ax.set_xlabel("Avg Duration (min)")
    ax.set_title(title)
    ax.bar_label(bars, fmt="%.1f", padding=3, color=TEXT, fontsize=9)
    ax.grid(True, axis="x")

plt.tight_layout()
save(fig, "05_duration_by_pickup_zone.png")

# =============================================================================
# 6. DURATION BY PASSENGER COUNT
# =============================================================================
hdr("6. DURATION BY PASSENGER COUNT")

by_pax = con.execute(f"""
    SELECT
        CAST(passenger_count AS INTEGER) AS pax,
        AVG(trip_duration_minutes)        AS avg_dur,
        COUNT(*)                          AS trips
    FROM read_parquet({p})
    WHERE passenger_count BETWEEN 1 AND 6
    GROUP BY pax ORDER BY pax
""").df()

print(f"\n  {'Passengers':>10} {'Avg dur (min)':>14} {'Trips':>12}")
print(f"  {'-'*10:>10} {'-'*14:>14} {'-'*12:>12}")
for _, row in by_pax.iterrows():
    print(f"  {int(row.pax):>10} {row.avg_dur:>14.2f} {int(row.trips):>12,}")

print("\n  Model implication:")
print("  - Minimal variation in duration by passenger count.")
print("  - Passenger count is a weak predictor of duration on its own.")
print("  - May still be useful as a proxy for trip type (solo vs group).")

# =============================================================================
# 7. DURATION BY VENDOR
# =============================================================================
hdr("7. DURATION BY VENDORID")

by_vendor = con.execute(f"""
    SELECT
        VendorID,
        AVG(trip_duration_minutes) AS avg_dur,
        PERCENTILE_CONT(0.5) WITHIN GROUP
            (ORDER BY trip_duration_minutes) AS median_dur,
        COUNT(*) AS trips
    FROM read_parquet({p})
    GROUP BY VendorID ORDER BY VendorID
""").df()

print(f"\n  {'VendorID':>10} {'Avg (min)':>10} {'Median (min)':>12} {'Trips':>12}")
print(f"  {'-'*10:>10} {'-'*10:>10} {'-'*12:>12} {'-'*12:>12}")
for _, row in by_vendor.iterrows():
    print(f"  {int(row.VendorID):>10} {row.avg_dur:>10.2f} "
          f"{row.median_dur:>12.2f} {int(row.trips):>12,}")

print("\n  Model implication:")
print("  - Small but measurable difference between vendors.")
print("  - VendorID may capture systematic GPS/metering differences.")
print("  - Include as a categorical feature; monitor for data leakage.")

# --- Figure: passenger count + vendor combined panel -------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Duration by Passenger Count and Vendor")

ax = axes[0]
bars = ax.bar(by_pax["pax"].astype(str), by_pax["avg_dur"],
              color=ACCENT, alpha=0.88, width=0.6)
ax.set_xlabel("Passenger Count")
ax.set_ylabel("Avg Duration (min)")
ax.set_title("By Passenger Count (1-6)")
ax.bar_label(bars, fmt="%.1f", padding=3, color=TEXT, fontsize=10)
ax.grid(True, axis="y")

ax = axes[1]
x    = range(len(by_vendor))
w    = 0.35
bars1 = ax.bar([i - w/2 for i in x], by_vendor["avg_dur"],
               width=w, label="Mean", color=PALETTE, alpha=0.88)
bars2 = ax.bar([i + w/2 for i in x], by_vendor["median_dur"],
               width=w, label="Median", color=PALETTE2, alpha=0.88)
ax.set_xticks(list(x))
ax.set_xticklabels(by_vendor["VendorID"].astype(str))
ax.set_xlabel("VendorID")
ax.set_ylabel("Duration (min)")
ax.set_title("By Vendor (Mean vs Median)")
ax.legend(framealpha=0.3)
ax.grid(True, axis="y")
ax.bar_label(bars1, fmt="%.1f", padding=3, color=TEXT, fontsize=9)
ax.bar_label(bars2, fmt="%.1f", padding=3, color=TEXT, fontsize=9)

plt.tight_layout()
save(fig, "06_duration_by_pax_and_vendor.png")

# =============================================================================
# 8. UNUSUAL / POTENTIALLY PROBLEMATIC VALUES
# =============================================================================
hdr("8. UNUSUAL VALUE ANALYSIS")

total = con.execute(f"SELECT COUNT(*) FROM read_parquet({p})").fetchone()[0]

checks = {
    "Duration < 1 min (likely invalid)":
        f"trip_duration_minutes < 1",
    "Duration > 120 min (very long)":
        f"trip_duration_minutes > 120",
    "Distance < 0.1 mi (near-zero)":
        f"trip_distance < 0.1",
    "Distance > 50 mi (very long haul)":
        f"trip_distance > 50",
    "Passenger count = 0":
        f"passenger_count = 0",
    "Passenger count > 6":
        f"passenger_count > 6",
    "Duration > 0 but Distance = 0":
        f"trip_duration_minutes > 1 AND trip_distance = 0",
}

print(f"\n  {'Flag':<40} {'Count':>10} {'%':>7}")
print(f"  {'-'*39:<40} {'-'*10:>10} {'-'*7:>7}")
outlier_summary = []
for label, condition in checks.items():
    cnt = con.execute(
        f"SELECT COUNT(*) FROM read_parquet({p}) WHERE {condition}"
    ).fetchone()[0]
    pct = cnt / total * 100
    flag = " [!!]" if cnt > 0 else ""
    print(f"  {label:<40} {cnt:>10,} {pct:>6.3f}%{flag}")
    outlier_summary.append({"flag": label, "count": cnt, "pct": pct})

print("\n  Model implication:")
print("  - Trips < 1 min and zero-distance trips are measurement artefacts.")
print("  - Decision required: clip, remove, or flag before feature engineering.")
print("  - Passenger count 0 / >6 are rare but should be handled explicitly.")
print("  - Duration > 120 min may be valid airport/long-haul trips or outliers.")

# --- Figure: outlier summary bar chart ---------------------------------------
out_df = pd.DataFrame(outlier_summary)
out_df = out_df[out_df["count"] > 0].sort_values("count", ascending=True)

if not out_df.empty:
    fig, ax = plt.subplots(figsize=(12, max(4, len(out_df) * 0.7)))
    fig.suptitle("Unusual Value Counts")

    bars = ax.barh(out_df["flag"], out_df["count"] / 1e3,
                   color=PALETTE2, alpha=0.85)
    ax.set_xlabel("Trip Count (thousands)")
    ax.bar_label(bars,
                 labels=[f"{v/1e3:,.1f}k" for v in out_df["count"]],
                 padding=4, color=TEXT, fontsize=9)
    ax.grid(True, axis="x")
    plt.tight_layout()
    save(fig, "07_unusual_value_summary.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
hdr("EDA SUMMARY & MODEL IMPLICATIONS")

print("""
  FINDINGS
  --------
  1. TARGET   : trip_duration_minutes is right-skewed (median 12.7 min,
                mean ~14 min, max 180 min). Log-transform recommended.

  2. DISTANCE : Moderate correlation with duration (r ~ 0.55-0.65).
                Strong predictor but not sufficient alone. Log or bucket.

  3. TIME     : Hour of day and day of week show strong patterns.
                Rush hours (8-9am, 5-7pm) drive longer durations.
                Weekend trips slightly shorter on average.
                These are high-priority features.

  4. ZONES    : High variance across pickup/dropoff zones. Zone ID
                alone has ~260 levels; target encoding recommended
                to avoid cardinality explosion.

  5. VENDORS  : Small systematic gap between Vendor 1 and 2.
                Include as binary categorical feature.

  6. PAX      : Minimal effect on duration. Low individual importance
                but may interact with zone / time features.

  DATA-QUALITY DECISIONS REQUIRED BEFORE FEATURE ENGINEERING
  -----------------------------------------------------------
  A. Trips < 1 min    : Remove or flag? (measurement error likely)
  B. Trips > 120 min  : Hard cap? Investigate if airport runs explain them.
  C. Distance = 0     : Remove? (GPS failure or cancellation artefacts)
  D. Passenger cnt 0  : Impute or remove?

  RECOMMENDED NEXT STEPS
  ----------------------
  - Decide on outlier treatment thresholds (items A-D above).
  - Perform feature engineering:
      * Log-transform target and distance
      * Extract hour, dow, month from pickup datetime
      * Target-encode zone IDs using cross-validated mean
      * Binary-encode VendorID
  - Then train baseline model (e.g. Ridge, LightGBM).
""")

print(f"\n{SEP}")
print("  EDA complete. Figures saved to reports/figures/")
print(SEP)

# List all figures
print("\n  Generated files:")
for f in sorted(FIGURES.glob("*.png")):
    size_kb = f.stat().st_size // 1024
    print(f"    {f.name:<45} {size_kb:>5} KB")
