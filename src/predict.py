"""
src/predict.py
--------------
Inference pipeline for the canonical HGBR v2 model.

Loads the model once at module import time.
Provides predict_duration() as the single public API.

Derives automatically:
  - is_weekend   : 1 if pickup_day_of_week in {5, 6} (Saturday=5, Sunday=6)
  - is_long_haul : 1 if trip_distance > 50.0 miles  (matches clean_data.py rule)

Feature order matches training exactly:
  [trip_distance, passenger_count, pickup_hour, is_weekend, is_long_haul,
   PULocationID, DOLocationID, pickup_day_of_week, pickup_month,
   RatecodeID, VendorID]

Usage
-----
    from src.predict import predict_duration
    minutes = predict_duration(
        trip_distance=3.5,
        passenger_count=2,
        pickup_hour=14,
        pickup_day_of_week=2,
        pickup_month=6,
        PULocationID=161,
        DOLocationID=234,
        RatecodeID=1,
        VendorID=2,
    )
"""

from pathlib import Path
import numpy as np
import joblib

# ---------------------------------------------------------------------------
# Constants matching clean_data.py feature engineering
# ---------------------------------------------------------------------------
LONG_HAUL_THRESHOLD_MILES = 50.0
WEEKEND_DAYS = {5, 6}   # Monday=0 ... Saturday=5, Sunday=6

# Valid ranges / sets
VALID_PASSENGER_COUNT = range(1, 7)         # 1–6 inclusive
VALID_HOUR            = range(0, 24)        # 0–23
VALID_MONTH           = range(1, 13)        # 1–12
VALID_DOW             = range(0, 7)         # 0=Mon … 6=Sun
VALID_VENDOR_IDS      = {1, 2}
VALID_RATECODEID      = {1, 2, 3, 4, 5, 6, 99}
VALID_LOCATION_RANGE  = range(1, 266)       # TLC zones 1–265

# Feature order MUST match training exactly
FEATURE_ORDER = [
    "trip_distance",
    "passenger_count",
    "pickup_hour",
    "is_weekend",
    "is_long_haul",
    "PULocationID",
    "DOLocationID",
    "pickup_day_of_week",
    "pickup_month",
    "RatecodeID",
    "VendorID",
]

# ---------------------------------------------------------------------------
# Load model once at import time
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
_MODEL_PATH = _ROOT / "models" / "hist_gradient_boosting_v2.joblib"

_model = joblib.load(_MODEL_PATH)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def predict_duration(
    trip_distance: float,
    passenger_count: int,
    pickup_hour: int,
    pickup_day_of_week: int,
    pickup_month: int,
    PULocationID: int,
    DOLocationID: int,
    RatecodeID: int = 1,
    VendorID: int = 1,
) -> float:
    """
    Predict taxi trip duration in minutes.

    Parameters
    ----------
    trip_distance      : float  -- trip distance in miles (>= 0)
    passenger_count    : int    -- number of passengers (1–6)
    pickup_hour        : int    -- hour of pickup (0–23)
    pickup_day_of_week : int    -- 0=Monday … 6=Sunday
    pickup_month       : int    -- 1–12
    PULocationID       : int    -- TLC pickup zone ID (1–265)
    DOLocationID       : int    -- TLC dropoff zone ID (1–265)
    RatecodeID         : int    -- 1=Standard, 2=JFK, 3=Newark, 4=Nassau/Westchester,
                                   5=Negotiated, 6=Group ride, 99=Unknown
    VendorID           : int    -- 1=Creative Mobile Technologies, 2=VeriFone Inc.

    Returns
    -------
    float : predicted trip duration in minutes

    Raises
    ------
    ValueError : on any invalid input
    """
    # --- Validate ---
    if trip_distance < 0:
        raise ValueError(f"trip_distance must be >= 0, got {trip_distance}")
    if passenger_count not in VALID_PASSENGER_COUNT:
        raise ValueError(f"passenger_count must be 1–6, got {passenger_count}")
    if pickup_hour not in VALID_HOUR:
        raise ValueError(f"pickup_hour must be 0–23, got {pickup_hour}")
    if pickup_day_of_week not in VALID_DOW:
        raise ValueError(f"pickup_day_of_week must be 0–6, got {pickup_day_of_week}")
    if pickup_month not in VALID_MONTH:
        raise ValueError(f"pickup_month must be 1–12, got {pickup_month}")
    if PULocationID not in VALID_LOCATION_RANGE:
        raise ValueError(f"PULocationID must be 1–265, got {PULocationID}")
    if DOLocationID not in VALID_LOCATION_RANGE:
        raise ValueError(f"DOLocationID must be 1–265, got {DOLocationID}")
    if RatecodeID not in VALID_RATECODEID:
        raise ValueError(f"RatecodeID must be one of {sorted(VALID_RATECODEID)}, got {RatecodeID}")
    if VendorID not in VALID_VENDOR_IDS:
        raise ValueError(f"VendorID must be 1 or 2, got {VendorID}")

    # --- Derive engineered features ---
    is_weekend   = int(pickup_day_of_week in WEEKEND_DAYS)
    is_long_haul = int(trip_distance > LONG_HAUL_THRESHOLD_MILES)

    # --- Build feature vector in exact training order ---
    feature_vector = np.array([[
        float(trip_distance),
        float(passenger_count),
        float(pickup_hour),
        float(is_weekend),
        float(is_long_haul),
        float(PULocationID),
        float(DOLocationID),
        float(pickup_day_of_week),
        float(pickup_month),
        float(RatecodeID),
        float(VendorID),
    ]], dtype=np.float32)

    prediction = float(_model.predict(feature_vector)[0])
    return max(prediction, 0.0)   # clamp: never return negative duration
