"""
src/train_baseline.py
---------------------
Baseline Linear Regression model for NYC Taxi Trip Duration Prediction.

Strategy
--------
- Use DuckDB to read ONLY the 5 required feature columns + target from Parquet.
  This avoids loading the full wide schema into memory -- critical for 35M+ rows.
- Fit sklearn LinearRegression on the training set.
- Evaluate on the validation set (MAE, RMSE, R2).
- Save the trained model to models/baseline_linear_regression.joblib.

Run
---
    python src/train_baseline.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -- Project root on path -----------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.db import get_connection

# -- Paths --------------------------------------------------------------------
TRAIN_PATH  = ROOT / "data" / "processed" / "train.parquet"
VAL_PATH    = ROOT / "data" / "processed" / "validation.parquet"
MODEL_DIR   = ROOT / "models"
MODEL_PATH  = MODEL_DIR / "baseline_linear_regression.joblib"

# -- Config -------------------------------------------------------------------
FEATURES = [
    "trip_distance",
    "passenger_count",
    "pickup_hour",
    "pickup_day_of_week",
    "pickup_month",
]
TARGET = "trip_duration_minutes"

COLS_SQL = ", ".join(FEATURES + [TARGET])


def load_split(con, path, label):
    """
    Use DuckDB to project only the required columns from a Parquet file.
    Returns (X, y) as NumPy arrays -- avoids loading the full schema.
    """
    posix = path.as_posix()
    print(f"  Loading {label} from: {posix}")
    t0 = time.time()

    result = con.execute(
        f"SELECT {COLS_SQL} FROM read_parquet('{posix}')"
    ).fetchnumpy()

    X = np.column_stack([result[f] for f in FEATURES]).astype(np.float64)
    y = result[TARGET].astype(np.float64)

    elapsed = time.time() - t0
    print(f"  Loaded {len(y):,} rows in {elapsed:.2f}s  "
          f"| X shape: {X.shape} | y shape: {y.shape}")
    return X, y


def main():
    print("=" * 70)
    print("  NYC Taxi Trip Duration -- Baseline Linear Regression")
    print("=" * 70)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    con = get_connection()

    # 1. Load data
    print("\n[1/4] Loading data (DuckDB column projection -- 5 features + target only)")
    X_train, y_train = load_split(con, TRAIN_PATH, "Train")
    X_val,   y_val   = load_split(con, VAL_PATH,   "Validation")

    print(f"\n  Training rows   : {len(y_train):,}")
    print(f"  Validation rows : {len(y_val):,}")
    print(f"  Features used   : {FEATURES}")
    print(f"  Target          : {TARGET}")

    # 2. Validation target statistics
    print("\n[2/4] Validation target statistics")
    val_mean   = float(np.mean(y_val))
    val_median = float(np.median(y_val))
    print(f"  Validation target mean   : {val_mean:.4f} min")
    print(f"  Validation target median : {val_median:.4f} min")

    # 3. Train
    print("\n[3/4] Fitting LinearRegression ...")
    model = LinearRegression(n_jobs=-1)
    t_train_start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t_train_start
    print(f"  Training time: {train_time:.2f}s")

    # 4. Evaluate on validation
    print("\n[4/4] Evaluating on validation set ...")
    y_pred = model.predict(X_val)

    mae  = mean_absolute_error(y_val, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    r2   = r2_score(y_val, y_pred)

    # 5. Save model
    joblib.dump(model, MODEL_PATH)
    print(f"\n  Model saved to: {MODEL_PATH}")

    # 6. Final report
    print()
    print("=" * 70)
    print("  BASELINE MODEL REPORT")
    print("=" * 70)
    print(f"  Model             : LinearRegression (scikit-learn)")
    print(f"  Training rows     : {len(y_train):,}")
    print(f"  Validation rows   : {len(y_val):,}")
    print(f"  Features used     : {FEATURES}")
    print(f"  Target            : {TARGET}")
    print(f"  Training time     : {train_time:.2f}s")
    print()
    print("  -- Validation Target Statistics --")
    print(f"  Target mean       : {val_mean:.4f} min")
    print(f"  Target median     : {val_median:.4f} min")
    print()
    print("  -- Validation Metrics --")
    print(f"  MAE               : {mae:.4f} min")
    print(f"  RMSE              : {rmse:.4f} min")
    print(f"  R2                : {r2:.6f}")
    print()
    print("  -- Model Coefficients --")
    for feat, coef in zip(FEATURES, model.coef_):
        print(f"  {feat:<22}: {coef:+.6f}")
    print(f"  {'intercept':<22}: {model.intercept_:+.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
