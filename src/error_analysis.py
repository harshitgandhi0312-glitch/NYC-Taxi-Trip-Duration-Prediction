"""
src/error_analysis.py
---------------------
Lightweight error analysis using test predictions from evaluate_final.py.

Reads pre-saved numpy arrays from reports/:
  - test_y_true.npy
  - test_y_pred.npy
  - test_trip_distance.npy
  - test_pickup_hour.npy

Produces 2 figures in reports/figures/:
  1. mae_by_distance_bucket.png  -- MAE broken down by trip distance bucket
  2. mae_by_pickup_hour.png      -- MAE broken down by pickup hour

Run AFTER evaluate_final.py:
    python src/error_analysis.py
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend -- safe for script mode
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT        = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"
FIG_DIR     = REPORTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Load saved arrays ─────────────────────────────────────────────────────────
print("Loading saved test arrays ...")
y_true    = np.load(REPORTS_DIR / "test_y_true.npy")
y_pred    = np.load(REPORTS_DIR / "test_y_pred.npy")
distances = np.load(REPORTS_DIR / "test_trip_distance.npy")
hours     = np.load(REPORTS_DIR / "test_pickup_hour.npy")

abs_errors = np.abs(y_true - y_pred)
print(f"  Loaded {len(y_true):,} test predictions.")
print(f"  Overall test MAE: {abs_errors.mean():.4f} min")

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE   = "#2563EB"   # blue
ACCENT    = "#DC2626"   # red for reference line
BG        = "#F8FAFC"
GRID_COL  = "#E2E8F0"

def style_ax(ax):
    ax.set_facecolor(BG)
    ax.grid(axis="y", color=GRID_COL, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CBD5E1")
    ax.tick_params(colors="#475569")
    ax.xaxis.label.set_color("#1E293B")
    ax.yaxis.label.set_color("#1E293B")
    ax.title.set_color("#1E293B")

# ── Figure 1: MAE by distance bucket ─────────────────────────────────────────
print("\n[1/2] Generating MAE by distance bucket ...")

edges  = [0, 1, 2, 3, 5, 8, 12, 20, 50, 200]
labels = ["0–1", "1–2", "2–3", "3–5", "5–8", "8–12", "12–20", "20–50", "50+"]
bucket_idx = np.digitize(distances, bins=edges[1:-1])   # 0..len(labels)-1

bucket_mae   = []
bucket_count = []
for i in range(len(labels)):
    mask = bucket_idx == i
    if mask.sum() > 0:
        bucket_mae.append(abs_errors[mask].mean())
        bucket_count.append(mask.sum())
    else:
        bucket_mae.append(np.nan)
        bucket_count.append(0)

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(BG)
bars = ax.bar(labels, bucket_mae, color=PALETTE, width=0.65, zorder=3)
ax.axhline(abs_errors.mean(), color=ACCENT, linestyle="--", linewidth=1.4,
           label=f"Overall MAE ({abs_errors.mean():.2f} min)", zorder=4)

# Annotate counts
for bar, count in zip(bars, bucket_count):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"n={count/1000:.0f}k", ha="center", va="bottom",
            fontsize=7.5, color="#475569")

style_ax(ax)
ax.set_xlabel("Trip Distance (miles)", fontsize=11, labelpad=8)
ax.set_ylabel("Mean Absolute Error (minutes)", fontsize=11, labelpad=8)
ax.set_title("Test Set: MAE by Trip Distance Bucket", fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=9, framealpha=0.8)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
plt.tight_layout()
out1 = FIG_DIR / "mae_by_distance_bucket.png"
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out1}")

# ── Figure 2: MAE by pickup hour ──────────────────────────────────────────────
print("\n[2/2] Generating MAE by pickup hour ...")

hour_mae   = []
hour_count = []
hour_list  = list(range(24))
for h in hour_list:
    mask = hours == h
    if mask.sum() > 0:
        hour_mae.append(abs_errors[mask].mean())
        hour_count.append(mask.sum())
    else:
        hour_mae.append(np.nan)
        hour_count.append(0)

fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor(BG)

# Color bars by time-of-day band
colors = []
for h in hour_list:
    if 0 <= h < 6:
        colors.append("#7C3AED")    # night -- purple
    elif 6 <= h < 10:
        colors.append("#D97706")    # morning rush -- amber
    elif 10 <= h < 16:
        colors.append(PALETTE)      # midday -- blue
    elif 16 <= h < 20:
        colors.append("#DC2626")    # evening rush -- red
    else:
        colors.append("#059669")    # evening -- green

ax.bar(hour_list, hour_mae, color=colors, width=0.75, zorder=3)
ax.axhline(abs_errors.mean(), color="#94A3B8", linestyle="--", linewidth=1.4,
           label=f"Overall MAE ({abs_errors.mean():.2f} min)", zorder=4)

style_ax(ax)
ax.set_xlabel("Pickup Hour (0 = midnight)", fontsize=11, labelpad=8)
ax.set_ylabel("Mean Absolute Error (minutes)", fontsize=11, labelpad=8)
ax.set_title("Test Set: MAE by Pickup Hour", fontsize=13, fontweight="bold", pad=12)
ax.set_xticks(hour_list)
ax.set_xticklabels([str(h) for h in hour_list], fontsize=8)

# Legend for time bands
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#7C3AED", label="Night (0–5)"),
    Patch(facecolor="#D97706", label="Morning rush (6–9)"),
    Patch(facecolor=PALETTE,   label="Midday (10–15)"),
    Patch(facecolor="#DC2626", label="Evening rush (16–19)"),
    Patch(facecolor="#059669", label="Evening (20–23)"),
]
ax.legend(handles=legend_elements, fontsize=8.5, framealpha=0.85, ncol=3)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
plt.tight_layout()
out2 = FIG_DIR / "mae_by_pickup_hour.png"
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out2}")

print("\nError analysis complete.")
print(f"  {out1.name}")
print(f"  {out2.name}")
