"""
src/train_hgb.py
----------------
Model 2: HistGradientBoostingRegressor for NYC Taxi Trip Duration Prediction.

Strategy
--------
- DuckDB projects only the 13 required columns (no full-schema load).
- Numerical columns loaded as float32 to halve memory vs float64.
- Categorical columns passed directly to HGBR via categorical_features --
  no OrdinalEncoder needed (sklearn >= 1.0 native support).
- PULocationID and DOLocationID are kept as integers (not one-hot encoded).
- Full 35M-row training set is used (memory pre-checked: feasible).

HGBR Initial Configuration (no hyperparameter tuning)
------------------------------------------------------
  max_iter          = 300    -- enough iterations for the model to converge
  max_leaf_nodes    = 31     -- sklearn default; controls tree complexity
  min_samples_leaf  = 20     -- slight regularisation for large dataset
  learning_rate     = 0.1    -- standard starting rate
  max_bins          = 255    -- maximum histogram resolution (sklearn default)
  l2_regularization = 0.0    -- no extra L2 (rely on leaf/depth constraints)
  early_stopping    = False  -- use full validation separately, not inline
  random_state      = 42

Memory pre-check (run during environment inspection)
----------------------------------------------------
  Available RAM : 7.1 GB
  Peak estimate : ~3.7 GB  -->  FEASIBLE

Baseline reference (LinearRegression on 5 features)
----------------------------------------------------
  MAE  : 5.2725 min
  RMSE : 8.1799 min
  R2   : 0.633056

Run
---
    python src/train_hgb.py
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
MODEL_PATH = MODEL_DIR / "hist_gradient_boosting.joblib"

# -- Feature config -----------------------------------------------------------
# NOTE: PULocationID and DOLocationID have 262 distinct values each.
# sklearn HGBR enforces cardinality <= max_bins (hard ceiling: 255) for any
# column declared as categorical. Exceeding this raises a ValueError.
# Solution: treat them as numerical -- HGBR still learns nonlinear location
# relationships via histogram splitting without the categorical OHE treatment.
NUMERICAL_FEATURES = [
    "trip_distance",
    "passenger_count",
    "pickup_hour",
    "is_weekend",
    "is_long_haul",
    "PULocationID",    # cardinality 262 > max_bins(255): must be numerical
    "DOLocationID",    # cardinality 262 > max_bins(255): must be numerical
]
CATEGORICAL_FEATURES = [
    # All have cardinality <= 12, well within the 255-bin limit.
    "pickup_day_of_week",   # 7
    "pickup_month",         # 12
    "pickup_year",          # 1  (2023 only in train)
    "RatecodeID",           # 7
    "VendorID",             # 2
]
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET = "trip_duration_minutes"

# Indices of categorical columns in the feature matrix (for HGBR)
CAT_INDICES = list(range(len(NUMERICAL_FEATURES), len(ALL_FEATURES)))

# -- Baseline reference -------------------------------------------------------
BASELINE = {"MAE": 5.2725, "RMSE": 8.1799, "R2": 0.633056}

# -- HGBR config --------------------------------------------------------------
HGBR_PARAMS = dict(
    max_iter             = 300,
    max_leaf_nodes       = 31,
    min_samples_leaf     = 20,
    learning_rate        = 0.1,
    max_bins             = 255,
    l2_regularization    = 0.0,
    early_stopping       = False,   # use full max_iter; evaluate separately
    categorical_features = CAT_INDICES,
    random_state         = 42,
    # n_iter_no_change omitted: only valid when early_stopping=True (sklearn >= 1.8)
)

COLS_SQL = ", ".join(ALL_FEATURES + [TARGET])


def load_split(con, path, label):
    """
    Project only the required columns from Parquet via DuckDB.
    Returns a pandas DataFrame with dtypes:
      - float32 for numerical columns (halves memory vs float64)
      - int32   for categorical columns (HGBR requires numeric input)
    """
    posix = path.as_posix()
    print(f"  Loading {label} from: {posix}")
    t0 = time.time()

    result = con.execute(
        f"SELECT {COLS_SQL} FROM read_parquet('{posix}')"
    ).fetchnumpy()

    # Build DataFrame with minimal dtypes
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
    print("  NYC Taxi Trip Duration -- Model 2: HistGradientBoostingRegressor")
    print("=" * 72)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    con = get_connection()

    # 1. Load data
    print(f"\n[1/5] Loading data")
    print(f"      Columns projected : {ALL_FEATURES}")
    print(f"      Numerical          : {NUMERICAL_FEATURES}")
    print(f"      Categorical        : {CATEGORICAL_FEATURES}")
    print(f"      Categorical indices: {CAT_INDICES}")
    train_df = load_split(con, TRAIN_PATH, "Train")
    val_df   = load_split(con, VAL_PATH,   "Validation")

    X_train = train_df[ALL_FEATURES].values
    y_train = train_df[TARGET].values.astype(np.float32)
    X_val   = val_df[ALL_FEATURES].values
    y_val   = val_df[TARGET].values.astype(np.float32)

    # Free DataFrames now that we have arrays
    del train_df, val_df

    print(f"\n  Training rows   : {len(y_train):,}")
    print(f"  Validation rows : {len(y_val):,}")
    print(f"  Target          : {TARGET}")

    # 2. Validation target statistics
    print("\n[2/5] Validation target statistics")
    val_mean   = float(np.mean(y_val))
    val_median = float(np.median(y_val))
    print(f"  Validation target mean   : {val_mean:.4f} min")
    print(f"  Validation target median : {val_median:.4f} min")

    # 3. Model config
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
    print(f"  Training time   : {train_time:.2f}s")
    print(f"  Actual iterations: {actual_iters}")

    # 5. Evaluate
    print("\n[5/5] Evaluating on validation set ...")
    y_pred = model.predict(X_val)

    mae  = mean_absolute_error(y_val, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    r2   = r2_score(y_val, y_pred)

    # Save
    joblib.dump(model, MODEL_PATH)
    print(f"  Model saved to: {MODEL_PATH}")

    # Final report
    print()
    print("=" * 72)
    print("  MODEL 2 REPORT -- HistGradientBoostingRegressor")
    print("=" * 72)
    print(f"  Estimator         : HistGradientBoostingRegressor (scikit-learn)")
    print(f"  Training rows     : {len(y_train):,}")
    print(f"  Validation rows   : {len(y_val):,}")
    print(f"  Features (num)    : {NUMERICAL_FEATURES}")
    print(f"  Features (cat)    : {CATEGORICAL_FEATURES}")
    print(f"  Target            : {TARGET}")
    print(f"  Training time     : {train_time:.2f}s")
    print(f"  Iterations used   : {actual_iters}")
    print()
    print("  -- Validation Target Statistics --")
    print(f"  Target mean       : {val_mean:.4f} min")
    print(f"  Target median     : {val_median:.4f} min")
    print()
    print(f"  {'Metric':<8}  {'Baseline (LR)':<18} {'Model 2 (HGBR)':<18} {'Delta'}")
    print(f"  {'-'*60}")
    print(f"  {'MAE':<8}  {BASELINE['MAE']:<18.4f} {mae:<18.4f} {mae - BASELINE['MAE']:+.4f} min")
    print(f"  {'RMSE':<8}  {BASELINE['RMSE']:<18.4f} {rmse:<18.4f} {rmse - BASELINE['RMSE']:+.4f} min")
    print(f"  {'R2':<8}  {BASELINE['R2']:<18.6f} {r2:<18.6f} {r2 - BASELINE['R2']:+.6f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
