"""
src/train_hgb_v2.py
-------------------
Model 2b: HistGradientBoostingRegressor - controlled experiment.

Changes from Model 2 (train_hgb.py)
-------------------------------------
  REMOVED: pickup_year
    Reason: training set contains only 2023; the column has cardinality=1
    and therefore provides zero useful training variation. It would only
    add noise and could hurt generalisation to 2024 validation data.

Everything else is identical to Model 2:
  - Same 10 remaining features
  - Same hyperparameters (max_iter=300, lr=0.1, etc.)
  - PULocationID / DOLocationID remain numerical (cardinality 262 > max_bins 255)
  - Native categorical handling for pickup_day_of_week, pickup_month,
    RatecodeID, VendorID  (all cardinality <= 12)
  - No OHE, no test set, no broad hyperparameter search

Model 2 reference (12 features, including pickup_year)
-------------------------------------------------------
  MAE  : 3.3063 min
  RMSE : 5.5786 min
  R2   : 0.829329

Run
---
    python src/train_hgb_v2.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -- Project root on path -----------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.db import get_connection

# -- Paths --------------------------------------------------------------------
TRAIN_PATH = ROOT / "data" / "processed" / "train.parquet"
VAL_PATH   = ROOT / "data" / "processed" / "validation.parquet"
MODEL_DIR  = ROOT / "models"
MODEL_PATH = MODEL_DIR / "hist_gradient_boosting_v2.joblib"

# -- Feature config -----------------------------------------------------------
# PULocationID / DOLocationID: cardinality 262 > max_bins ceiling of 255.
# Must remain numerical (HGBR hard constraint -- cannot be changed).
NUMERICAL_FEATURES = [
    "trip_distance",     # 0
    "passenger_count",   # 1
    "pickup_hour",       # 2
    "is_weekend",        # 3
    "is_long_haul",      # 4
    "PULocationID",      # 5  -- numerical (cardinality 262 > 255 limit)
    "DOLocationID",      # 6  -- numerical (cardinality 262 > 255 limit)
]
CATEGORICAL_FEATURES = [
    # Cardinality all <= 12 -- well within max_bins=255 limit.
    # pickup_year REMOVED: cardinality=1 in training (2023 only) -- no signal.
    "pickup_day_of_week",   # 7  -- cardinality 7
    "pickup_month",         # 8  -- cardinality 12
    "RatecodeID",           # 9  -- cardinality 7
    "VendorID",             # 10 -- cardinality 2
]
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET = "trip_duration_minutes"

# Indices of categorical columns within the feature matrix
CAT_INDICES = list(range(len(NUMERICAL_FEATURES), len(ALL_FEATURES)))
# = [7, 8, 9, 10]

# -- Reference metrics --------------------------------------------------------
MODEL2 = {"MAE": 3.3063, "RMSE": 5.5786, "R2": 0.829329}

# -- HGBR config (identical to Model 2) --------------------------------------
HGBR_PARAMS = dict(
    max_iter             = 300,
    max_leaf_nodes       = 31,
    min_samples_leaf     = 20,
    learning_rate        = 0.1,
    max_bins             = 255,
    l2_regularization    = 0.0,
    early_stopping       = False,
    categorical_features = CAT_INDICES,
    random_state         = 42,
)

COLS_SQL = ", ".join(ALL_FEATURES + [TARGET])


def load_split(con, path, label):
    """
    Project only the required columns from Parquet via DuckDB.
    float32 for numericals, int32 for categoricals -- minimal memory footprint.
    """
    posix = path.as_posix()
    print(f"  Loading {label} from: {posix}")
    t0 = time.time()

    result = con.execute(
        f"SELECT {COLS_SQL} FROM read_parquet('{posix}')"
    ).fetchnumpy()

    data = {}
    for col in NUMERICAL_FEATURES:
        data[col] = result[col].astype(np.float32)
    for col in CATEGORICAL_FEATURES:
        data[col] = result[col].astype(np.int32)
    data[TARGET] = result[TARGET].astype(np.float32)

    df = pd.DataFrame(data)
    elapsed = time.time() - t0
    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    print(f"  Loaded {len(df):,} rows in {elapsed:.2f}s  "
          f"| {len(ALL_FEATURES)} features | RAM: {mem_mb:.0f} MB")
    return df


def main():
    print("=" * 72)
    print("  NYC Taxi Trip Duration -- HGBR v2 (pickup_year removed)")
    print("=" * 72)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    con = get_connection()

    # 1. Load
    print(f"\n[1/5] Loading data")
    print(f"      Features ({len(ALL_FEATURES)})  : {ALL_FEATURES}")
    print(f"      Numerical ({len(NUMERICAL_FEATURES)}) : {NUMERICAL_FEATURES}")
    print(f"      Categorical ({len(CATEGORICAL_FEATURES)}): {CATEGORICAL_FEATURES}")
    print(f"      Categorical indices : {CAT_INDICES}")
    print(f"      Removed vs Model 2  : ['pickup_year'] (cardinality=1 in train)")

    train_df = load_split(con, TRAIN_PATH, "Train")
    val_df   = load_split(con, VAL_PATH,   "Validation")

    X_train = train_df[ALL_FEATURES].values
    y_train = train_df[TARGET].values.astype(np.float32)
    X_val   = val_df[ALL_FEATURES].values
    y_val   = val_df[TARGET].values.astype(np.float32)
    del train_df, val_df

    print(f"\n  Training rows   : {len(y_train):,}")
    print(f"  Validation rows : {len(y_val):,}")
    print(f"  Target          : {TARGET}")

    # 2. Validation target stats
    print("\n[2/5] Validation target statistics")
    val_mean   = float(np.mean(y_val))
    val_median = float(np.median(y_val))
    print(f"  Target mean   : {val_mean:.4f} min")
    print(f"  Target median : {val_median:.4f} min")

    # 3. Config
    print("\n[3/5] Model configuration")
    print(f"  Estimator       : HistGradientBoostingRegressor")
    for k, v in HGBR_PARAMS.items():
        print(f"  {k:<22}: {v}")

    # 4. Train
    print("\n[4/5] Fitting HistGradientBoostingRegressor ...")
    model = HistGradientBoostingRegressor(**HGBR_PARAMS)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    actual_iters = model.n_iter_
    print(f"  Training time    : {train_time:.2f}s")
    print(f"  Actual iterations: {actual_iters}")

    # 5. Evaluate
    print("\n[5/5] Evaluating on validation set ...")
    y_pred = model.predict(X_val)

    mae  = mean_absolute_error(y_val, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    r2   = r2_score(y_val, y_pred)

    joblib.dump(model, MODEL_PATH)
    print(f"  Model saved to: {MODEL_PATH}")

    # Final report
    print()
    print("=" * 72)
    print("  HGBR v2 REPORT (pickup_year removed)")
    print("=" * 72)
    print(f"  Estimator         : HistGradientBoostingRegressor (scikit-learn)")
    print(f"  Training rows     : {len(y_train):,}")
    print(f"  Validation rows   : {len(y_val):,}")
    print(f"  Features (num)    : {NUMERICAL_FEATURES}")
    print(f"  Features (cat)    : {CATEGORICAL_FEATURES}")
    print(f"  Removed features  : ['pickup_year']")
    print(f"  Target            : {TARGET}")
    print(f"  Training time     : {train_time:.2f}s")
    print(f"  Iterations used   : {actual_iters}")
    print()
    print("  -- Validation Target Statistics --")
    print(f"  Target mean       : {val_mean:.4f} min")
    print(f"  Target median     : {val_median:.4f} min")
    print()
    print(f"  {'Metric':<6}  {'Model 2 (HGBR)':<18} {'v2 (no yr)':<18} {'Abs Delta':<14} {'% Change'}")
    print(f"  {'-'*70}")

    def pct(new, old): return (new - old) / abs(old) * 100
    print(f"  {'MAE':<6}  {MODEL2['MAE']:<18.4f} {mae:<18.4f} {mae-MODEL2['MAE']:+.4f} min   {pct(mae,MODEL2['MAE']):+.2f}%")
    print(f"  {'RMSE':<6}  {MODEL2['RMSE']:<18.4f} {rmse:<18.4f} {rmse-MODEL2['RMSE']:+.4f} min   {pct(rmse,MODEL2['RMSE']):+.2f}%")
    print(f"  {'R2':<6}  {MODEL2['R2']:<18.6f} {r2:<18.6f} {r2-MODEL2['R2']:+.6f}       {pct(r2,MODEL2['R2']):+.2f}%")
    print("=" * 72)


if __name__ == "__main__":
    main()
