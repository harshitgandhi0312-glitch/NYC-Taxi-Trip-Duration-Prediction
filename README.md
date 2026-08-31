# 🚕 NYC Taxi Trip Duration Prediction

A machine learning portfolio project that predicts the total ride duration of taxi trips taken in New York City, using the [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

---

## 📁 Project Structure

```
NYC-Taxi-Trip-Duration-Prediction/
│
├── data/
│   ├── raw/          # Original, immutable data (not committed to Git)
│   └── processed/    # Cleaned & feature-engineered data (not committed to Git)
│
├── src/              # Reusable Python modules (feature engineering, modelling, etc.)
├── reports/
│   └── figures/      # Generated charts and visualisations
├── models/           # Serialised trained models (not committed to Git)
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🎯 Project Goals

- Perform exploratory data analysis (EDA) on NYC Yellow Taxi trip data.
- Engineer meaningful features (time of day, distance, borough zones, etc.).
- Train and evaluate regression models to predict trip duration.
- Document findings clearly for a professional portfolio.

---

## 🚀 Getting Started

```bash
# 1. Clone the repository
git clone <repo-url>
cd NYC-Taxi-Trip-Duration-Prediction

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download data
# Place raw Parquet files in data/raw/ (see Data section below)
```

---

## 📊 Data

Download the NYC TLC Yellow Taxi trip records (Parquet format) from:  
👉 https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Place the downloaded files inside `data/raw/`. These files are excluded from version control via `.gitignore`.

---

## 🛠️ Tech Stack

| Area | Libraries |
|------|-----------|
| Data manipulation | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| Machine learning | `scikit-learn`, `xgboost`, `lightgbm` |
| Notebooks | `jupyter` |
| Data I/O | `pyarrow`, `fastparquet` |

---

## 📄 License

This project is released under the MIT License.
