"""
src/evaluate_final.py
---------------------
Final one-time evaluation of the canonical HGBR v2 model on the held-out test set.

This is the first and only evaluation of this model on test data.
Do NOT retrain or tune after seeing these results.

Model : models/hist_gradient_boosting_v2.joblib
Test  : data/processed/test.parquet (2024 H2 -- never seen during training)

Validation reference (for context):
  MAE  : 3.3063 min
  RMSE : 5.5786 min
  R2   : 0.829329

Run
---
    python src/evaluate_final.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.db import get_connection

MODEL_PATH = ROOT / "models" / "hist_gradient_boosting_v2.joblib"
TEST_PATH  = ROOT / "data" / "processed" / "test.parquet"

NUMERICAL_FEATURES = [
    "trip_distance", "passenger_count", "pickup_hour",
    "is_weekend", "is_long_haul", "PULocationID", "DOLocationID",
]
CATEGORICAL_FEATURES = [
    "pickup_day_of_week", "pickup_month", "RatecodeID", "VendorID",
]
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET = "trip_duration_minutes"
COLS_SQL = ", ".join(ALL_FEATURES + [TARGET])

VAL_REF = {"MAE": 3.3063, "RMSE": 5.5786, "R2": 0.829329}


def main():
    print("=" * 70)
    print("  NYC Taxi Trip Duration -- FINAL TEST EVALUATION")
    print("  Model: hist_gradient_boosting_v2.joblib")
    print("=" * 70)

    # Load model
    print("\n[1/3] Loading model ...")
    model = joblib.load(MODEL_PATH)
    print(f"  Model loaded: {MODEL_PATH.name}")

    # Load test data (column projection only)
    print("\n[2/3] Loading test data ...")
    con = get_connection()
    posix = TEST_PATH.as_posix()
    t0 = time.time()
    result = con.execute(f"SELECT {COLS_SQL} FROM read_parquet('{posix}')").fetchnumpy()
    data = {}
    for col in NUMERICAL_FEATURES:
        data[col] = result[col].astype(np.float32)
    for col in CATEGORICAL_FEATURES:
        data[col] = result[col].astype(np.int32)
    data[TARGET] = result[TARGET].astype(np.float32)
    df = pd.DataFrame(data)
    load_time = time.time() - t0
    print(f"  Loaded {len(df):,} rows in {load_time:.2f}s")

    X_test = df[ALL_FEATURES].values
    y_test = df[TARGET].values.astype(np.float32)
    del df

    # Evaluate
    print("\n[3/3] Evaluating ...")
    t0 = time.time()
    y_pred = model.predict(X_test)
    eval_time = time.time() - t0

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = r2_score(y_test, y_pred)
    test_mean   = float(np.mean(y_test))
    test_median = float(np.median(y_test))

    # Save predictions for error analysis
    np.save(ROOT / "reports" / "test_y_true.npy", y_test)
    np.save(ROOT / "reports" / "test_y_pred.npy", y_pred)
    # Also save X_test columns needed for error analysis
    np.save(ROOT / "reports" / "test_trip_distance.npy",
            X_test[:, ALL_FEATURES.index("trip_distance")].astype(np.float32))
    np.save(ROOT / "reports" / "test_pickup_hour.npy",
            X_test[:, ALL_FEATURES.index("pickup_hour")].astype(np.int32))
    print("  Predictions saved to reports/ for error analysis.")

    def pct(new, old): return (new - old) / abs(old) * 100

    print()
    print("=" * 70)
    print("  FINAL TEST RESULTS")
    print("=" * 70)
    print(f"  Model             : HistGradientBoostingRegressor v2 (11 features)")
    print(f"  Test set          : data/processed/test.parquet (2024 H2)")
    print(f"  Test rows         : {len(y_test):,}")
    print(f"  Evaluation time   : {eval_time:.2f}s")
    print()
    print(f"  Test target mean  : {test_mean:.4f} min")
    print(f"  Test target median: {test_median:.4f} min")
    print()
    print(f"  {'Metric':<6}  {'Validation':<16} {'Test':<16} {'Delta':<12} {'% Change'}")
    print(f"  {'-'*62}")
    print(f"  {'MAE':<6}  {VAL_REF['MAE']:<16.4f} {mae:<16.4f} {mae-VAL_REF['MAE']:+.4f} min   {pct(mae,VAL_REF['MAE']):+.2f}%")
    print(f"  {'RMSE':<6}  {VAL_REF['RMSE']:<16.4f} {rmse:<16.4f} {rmse-VAL_REF['RMSE']:+.4f} min   {pct(rmse,VAL_REF['RMSE']):+.2f}%")
    print(f"  {'R2':<6}  {VAL_REF['R2']:<16.6f} {r2:<16.6f} {r2-VAL_REF['R2']:+.6f}       {pct(r2,VAL_REF['R2']):+.2f}%")
    print("=" * 70)
    print()
    print("  IMPORTANT: Do not retrain or tune after seeing these results.")


if __name__ == "__main__":
    main()
