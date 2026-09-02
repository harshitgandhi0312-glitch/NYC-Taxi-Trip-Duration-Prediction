# NYC Taxi Trip Duration Prediction

A machine learning portfolio project that predicts the total ride duration of NYC Yellow Taxi trips using the [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

## 🚀 Live Demo

**[▶ Try the Live Application](https://nyc-taxi-trip-duration-prediction-fhkne5ea89xukodxeysrij.streamlit.app/)**

## Project Overview

This project builds a complete end-to-end machine learning pipeline:

- Data ingestion from 2023-2024 NYC TLC Parquet files
- Large-scale data cleaning and validation on 70M+ records using DuckDB
- Feature engineering (time-of-day flags, long-haul indicator)
- Chronological train/validation/test split
- Baseline and advanced ML models
- Final evaluation on a strictly held-out test set
- Interactive Streamlit web application

---

## Problem Statement

Given a taxi trip's distance, pickup time, location, and fare details, predict the trip duration in minutes.

This is a **regression** problem. The target variable is `trip_duration_minutes`.

---

## Dataset

| Property | Details |
|---|---|
| Source | [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) |
| Years | 2023 and 2024 (H1 + H2) |
| Raw records | ~74M+ rows |
| After cleaning | **70,996,150** trips |
| Format | Parquet (ZSTD-compressed) |

Data files are excluded from version control (see `.gitignore`).

---

## Data Cleaning

Key cleaning steps applied via DuckDB SQL pipeline (`src/clean_data.py`):

- Removed trips with zero or negative duration
- Removed trips with zero or negative distance
- Removed extreme duration outliers (> 3 hours)
- Removed extreme distance outliers (> 200 miles for standard trips)
- Filtered to valid passenger counts (1-6)
- Removed invalid rate codes and vendor IDs
- Computed `trip_duration_minutes` from pickup and dropoff timestamps

---

## Feature Engineering

| Feature | Type | Description |
|---|---|---|
| `trip_distance` | Numerical | Distance in miles |
| `passenger_count` | Numerical | Number of passengers (1-6) |
| `pickup_hour` | Numerical | Hour of pickup (0-23) |
| `is_weekend` | Numerical (binary) | 1 if Saturday or Sunday |
| `is_long_haul` | Numerical (binary) | 1 if trip_distance > 50 miles |
| `PULocationID` | Numerical* | TLC pickup zone ID |
| `DOLocationID` | Numerical* | TLC dropoff zone ID |
| `pickup_day_of_week` | Categorical | 0=Monday...6=Sunday |
| `pickup_month` | Categorical | 1-12 |
| `RatecodeID` | Categorical | Fare rate type |
| `VendorID` | Categorical | Meter system vendor |

*PULocationID and DOLocationID have 262 distinct values, exceeding scikit-learn HGBR's hard categorical cardinality limit of `max_bins=255`. They are treated as numerical features.

---

## Temporal Data Split

| Split | Period | Rows |
|---|---|---|
| **Train** | All of 2023 | 35,477,486 |
| **Validation** | 2024 January - June | 17,612,604 |
| **Test** | 2024 July - December | 17,906,060 |
| **Total** | 2023 - 2024 | **70,996,150** |

Splits are strictly chronological. No future data leaks into training.

---

## Modeling

### Model 1: Linear Regression (Baseline)

- 5 numerical features only
- No hyperparameter tuning
- Scikit-learn `LinearRegression`

### Model 2: HistGradientBoostingRegressor (Canonical)

- 11 features (7 numerical + 4 categorical)
- Scikit-learn native categorical feature support
- `pickup_year` removed: cardinality=1 in training set (all 2023), provides no signal
- Default configuration, no hyperparameter tuning

| Parameter | Value |
|---|---|
| max_iter | 300 |
| max_leaf_nodes | 31 |
| min_samples_leaf | 20 |
| learning_rate | 0.1 |
| max_bins | 255 |

---

## Model Comparison

### Validation Set Results (2024 H1)

| Metric | Linear Regression | HGBR v2 | Improvement |
|---|---|---|---|
| **MAE** | 5.2725 min | **3.3063 min** | -37.3% |
| **RMSE** | 8.1799 min | **5.5786 min** | -31.8% |
| **R²** | 0.633056 | **0.829329** | +31.0% |

---

## Final Test Results

Test set: **2024 H2 (July-December)** — held out until final evaluation.
Evaluated exactly once. No retraining or tuning after seeing results.

| Metric | Validation | Test | Delta |
|---|---|---|---|
| **MAE** | 3.3063 min | **3.7831 min** | +14.4% |
| **RMSE** | 5.5786 min | **6.1953 min** | +11.1% |
| **R²** | 0.829329 | **0.819075** | -1.2% |

The small degradation from validation to test is expected and acceptable.
The model generalises well to unseen future data.

---

## Error Analysis

Error analysis was performed on test set predictions.

Key findings:
- **Short trips (0-1 mile)**: higher relative MAE due to variability in congestion and stops
- **Long-haul trips (50+ miles)**: lower absolute error; duration is dominated by distance
- **Rush hours (8-9am, 5-7pm)**: elevated MAE due to traffic unpredictability
- **Night hours (1-5am)**: lowest MAE; fewer trips, more consistent conditions

Figures: `reports/figures/`
- `mae_by_distance_bucket.png`
- `mae_by_pickup_hour.png`

---

## Streamlit Application

A professional web application is provided for interactive predictions.

```bash
streamlit run app.py
```

Features:
- Trip distance, passenger count, pickup time, location, rate code inputs
- Automatic derivation of `is_weekend` and `is_long_haul`
- Input validation with clear error messages
- Predicted duration displayed prominently in minutes
- Model performance section with validation and test metrics
- Baseline vs HGBR comparison
- Project insights and limitations

---

## Project Structure

```
NYC-Taxi-Trip-Duration-Prediction/
|
+-- data/
|   +-- raw/              # Raw Parquet files (not committed)
|   +-- processed/        # Cleaned splits: train/validation/test (not committed)
|
+-- models/               # Trained model files (not committed)
|   +-- baseline_linear_regression.joblib
|   +-- hist_gradient_boosting.joblib
|   +-- hist_gradient_boosting_v2.joblib    # Canonical final model
|
+-- reports/
|   +-- figures/          # Error analysis charts
|   +-- test_y_true.npy   # Saved test predictions (not committed)
|   +-- test_y_pred.npy
|
+-- src/
|   +-- db.py                  # DuckDB connection helper
|   +-- clean_data.py          # Data cleaning pipeline
|   +-- create_splits.py       # Temporal train/val/test splits
|   +-- validate_data.py       # Data validation checks
|   +-- eda.py                 # Exploratory data analysis
|   +-- train_baseline.py      # Model 1: Linear Regression
|   +-- train_hgb.py           # Model 2: HGBR (12 features)
|   +-- train_hgb_v2.py        # Model 2b: HGBR (11 features, canonical)
|   +-- evaluate_final.py      # Final test set evaluation
|   +-- predict.py             # Inference pipeline
|   +-- error_analysis.py      # Error analysis figures
|
+-- app.py                # Streamlit web application
+-- requirements.txt
+-- config.ini            # Dataset path template
+-- config.local.ini      # Local paths (not committed)
+-- .gitignore
+-- README.md
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure dataset paths

```bash
cp config.ini config.local.ini
# Edit config.local.ini with your local Parquet file paths
```

### 3. Data pipeline (if running from scratch)

```bash
python src/clean_data.py
python src/create_splits.py
```

### 4. Train models

```bash
python src/train_baseline.py
python src/train_hgb_v2.py
```

### 5. Evaluate on test set

```bash
python src/evaluate_final.py
python src/error_analysis.py
```

### 6. Run the application

```bash
streamlit run app.py
```

---

## Limitations

- The model does not use real-time traffic or weather data
- Predictions are statistical estimates based on historical patterns
- PULocationID and DOLocationID represent TLC taxi zones (1-265); treated as numerical due to scikit-learn HGBR cardinality constraints
- The model was trained on Yellow Taxi data only (not green taxi or rideshare)
- Performance may degrade for dates beyond the training period without retraining

---

## Future Improvements

- Add weather features (precipitation, temperature)
- Integrate with NYC traffic API for real-time adjustments
- Try XGBoost or LightGBM once environment compatibility is resolved
- Add hyperparameter tuning (Optuna or scikit-learn GridSearchCV)
- Map TLC zone IDs to borough names for a friendlier UI
- Add prediction intervals using quantile regression
- Schedule periodic retraining as new TLC data is released

---

## Tech Stack

| Area | Tool |
|---|---|
| Data access | DuckDB 1.5.5 |
| Data manipulation | pandas 2.3.3, numpy 2.4.2 |
| Machine learning | scikit-learn 1.8.0 |
| Model serialisation | joblib 1.5.3 |
| Visualisation | matplotlib |
| Web application | Streamlit 1.54.0 |
| Python | 3.14.3 |

---

## License

MIT License
