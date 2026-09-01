"""
app.py
------
NYC Taxi Trip Duration Predictor - Streamlit Application.
"""

import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Taxi Trip Duration Predictor",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #0F172A; }
[data-testid="stAppViewContainer"] { background: #0F172A; }
[data-testid="stHeader"] { background: #0F172A; }

.hero {
    background: linear-gradient(135deg, #1E3A5F 0%, #0F172A 60%, #1A1040 100%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    border: 1px solid #1E3A5F;
    text-align: center;
}
.hero h1 { color: #F8FAFC; font-size: 2.4rem; font-weight: 700; margin: 0 0 0.5rem 0; }
.hero p  { color: #94A3B8; font-size: 1.05rem; margin: 0; }

.card {
    background: #1E293B;
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid #334155;
    margin-bottom: 1.2rem;
}
.card h3 { color: #F1F5F9; font-size: 1.05rem; font-weight: 600; margin: 0 0 1rem 0; }

.result-box {
    background: linear-gradient(135deg, #1E3A8A 0%, #312E81 100%);
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    border: 1px solid #3B82F6;
    margin: 1.5rem 0;
}
.result-box .duration { color: #DBEAFE; font-size: 3.8rem; font-weight: 700; line-height: 1; }
.result-box .unit     { color: #93C5FD; font-size: 1.1rem; margin-top: 0.4rem; }
.result-box .note     { color: #64748B; font-size: 0.82rem; margin-top: 0.8rem; }

.metric-row {
    display: flex;
    gap: 0.8rem;
    margin-bottom: 0.5rem;
}
.metric-chip {
    background: #0F172A;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 0.5rem 0.9rem;
    flex: 1;
    text-align: center;
}
.metric-chip .label { color: #64748B; font-size: 0.75rem; }
.metric-chip .value { color: #F1F5F9; font-size: 1.1rem; font-weight: 600; }

.improvement-badge {
    background: #064E3B;
    color: #6EE7B7;
    border-radius: 6px;
    padding: 0.15rem 0.5rem;
    font-size: 0.78rem;
    font-weight: 600;
    display: inline-block;
    margin-left: 0.4rem;
}
.section-title {
    color: #F1F5F9;
    font-size: 1.2rem;
    font-weight: 600;
    margin: 0.5rem 0 1rem 0;
    border-bottom: 1px solid #334155;
    padding-bottom: 0.5rem;
}
.insight-item {
    display: flex;
    align-items: flex-start;
    gap: 0.7rem;
    padding: 0.5rem 0;
    color: #CBD5E1;
    font-size: 0.9rem;
    border-bottom: 1px solid #1E293B;
}
.tag {
    background: #1E3A5F;
    color: #93C5FD;
    font-size: 0.72rem;
    border-radius: 4px;
    padding: 0.1rem 0.45rem;
    white-space: nowrap;
    margin-top: 0.15rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_predictor():
    from src.predict import predict_duration
    return predict_duration

predict_duration = load_predictor()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🚕 NYC Taxi Trip Duration Predictor</h1>
  <p>Predict taxi trip duration using a machine learning model trained on 70M+ NYC taxi trips.</p>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────
col_inputs, col_results = st.columns([1.1, 0.9], gap="large")

# ── INPUT PANEL ───────────────────────────────────────────────────────────────
with col_inputs:
    st.markdown('<p class="section-title">Trip Details</p>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="card"><h3>Distance & Passengers</h3>', unsafe_allow_html=True)
        trip_distance    = st.number_input("Trip Distance (miles)", min_value=0.0, max_value=200.0,
                                           value=3.5, step=0.1, format="%.1f")
        passenger_count  = st.slider("Passenger Count", min_value=1, max_value=6, value=1)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="card"><h3>Pickup Time</h3>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            pickup_hour = st.selectbox(
                "Pickup Hour",
                options=list(range(24)),
                index=9,
                format_func=lambda h: f"{h:02d}:00 {'AM' if h < 12 else 'PM'}"
            )
        with c2:
            DOW_LABELS = ["Monday", "Tuesday", "Wednesday", "Thursday",
                          "Friday", "Saturday", "Sunday"]
            pickup_day_of_week = st.selectbox("Day of Week", options=list(range(7)),
                                              index=0, format_func=lambda d: DOW_LABELS[d])

        MONTH_LABELS = ["January","February","March","April","May","June",
                        "July","August","September","October","November","December"]
        pickup_month = st.selectbox("Pickup Month", options=list(range(1, 13)),
                                    index=5, format_func=lambda m: MONTH_LABELS[m - 1])
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="card"><h3>Route & Fare Type</h3>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            PULocationID = st.number_input("Pickup Zone ID (1–265)", min_value=1, max_value=265,
                                           value=161, step=1)
        with c4:
            DOLocationID = st.number_input("Dropoff Zone ID (1–265)", min_value=1, max_value=265,
                                           value=234, step=1)

        RATECODE_LABELS = {
            1: "1 - Standard Rate",
            2: "2 - JFK Airport",
            3: "3 - Newark Airport",
            4: "4 - Nassau / Westchester",
            5: "5 - Negotiated Fare",
            6: "6 - Group Ride",
            99: "99 - Unknown",
        }
        VENDOR_LABELS = {1: "1 - Creative Mobile Technologies", 2: "2 - VeriFone Inc."}

        RatecodeID = st.selectbox("Rate Code", options=[1, 2, 3, 4, 5, 6, 99],
                                  format_func=lambda r: RATECODE_LABELS[r])
        VendorID   = st.selectbox("Vendor", options=[1, 2],
                                  format_func=lambda v: VENDOR_LABELS[v])
        st.markdown("</div>", unsafe_allow_html=True)

    # Derived features info
    is_weekend   = int(pickup_day_of_week in {5, 6})
    is_long_haul = int(trip_distance > 50.0)
    st.markdown(
        f"**Auto-derived:** is_weekend = `{is_weekend}` &nbsp;|&nbsp; is_long_haul = `{is_long_haul}` "
        f"*(distance {'> 50 mi' if is_long_haul else '<= 50 mi'})*",
        unsafe_allow_html=True
    )

    predict_btn = st.button("🔮 Predict Trip Duration", type="primary", use_container_width=True)

# ── RESULTS PANEL ─────────────────────────────────────────────────────────────
with col_results:
    st.markdown('<p class="section-title">Prediction</p>', unsafe_allow_html=True)

    if predict_btn:
        try:
            minutes = predict_duration(
                trip_distance=trip_distance,
                passenger_count=passenger_count,
                pickup_hour=pickup_hour,
                pickup_day_of_week=pickup_day_of_week,
                pickup_month=pickup_month,
                PULocationID=PULocationID,
                DOLocationID=DOLocationID,
                RatecodeID=RatecodeID,
                VendorID=VendorID,
            )
            mins_int  = int(minutes)
            secs      = int((minutes - mins_int) * 60)
            st.markdown(f"""
<div class="result-box">
  <div class="duration">{minutes:.1f}</div>
  <div class="unit">minutes estimated</div>
  <div style="color:#93C5FD;font-size:0.95rem;margin-top:0.5rem;">
    (~{mins_int}m {secs}s)
  </div>
  <div class="note">Based on historical NYC taxi trip patterns.<br>
  This is a statistical estimate, not a real-time calculation.</div>
</div>
""", unsafe_allow_html=True)
        except ValueError as e:
            st.error(f"Invalid input: {e}")
    else:
        st.markdown("""
<div class="result-box" style="opacity:0.5;">
  <div class="duration">--.-</div>
  <div class="unit">minutes</div>
  <div class="note">Fill in trip details and click Predict</div>
</div>
""", unsafe_allow_html=True)

    # ── Model Performance ─────────────────────────────────────────────────────
    st.markdown('<p class="section-title">Model Performance</p>', unsafe_allow_html=True)
    st.markdown("""
<div class="card">
<h3>Validation Set (2024 H1)</h3>
<div class="metric-row">
  <div class="metric-chip"><div class="label">MAE</div><div class="value">3.31 min</div></div>
  <div class="metric-chip"><div class="label">RMSE</div><div class="value">5.58 min</div></div>
  <div class="metric-chip"><div class="label">R&sup2;</div><div class="value">0.829</div></div>
</div>
<h3 style="margin-top:1rem;">Final Test Set (2024 H2 — held out)</h3>
<div class="metric-row">
  <div class="metric-chip"><div class="label">MAE</div><div class="value">3.78 min</div></div>
  <div class="metric-chip"><div class="label">RMSE</div><div class="value">6.20 min</div></div>
  <div class="metric-chip"><div class="label">R&sup2;</div><div class="value">0.819</div></div>
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="card">
<h3>vs. Linear Regression Baseline</h3>
<table style="width:100%;border-collapse:collapse;font-size:0.88rem;">
<thead>
<tr style="color:#64748B;">
  <th style="text-align:left;padding:4px 0;">Metric</th>
  <th style="text-align:right;">Baseline (LR)</th>
  <th style="text-align:right;">HGBR (ours)</th>
  <th style="text-align:right;">Improvement</th>
</tr>
</thead>
<tbody style="color:#CBD5E1;">
<tr>
  <td style="padding:5px 0;border-top:1px solid #334155;">MAE</td>
  <td style="text-align:right;">5.27 min</td>
  <td style="text-align:right;">3.31 min</td>
  <td style="text-align:right;"><span class="improvement-badge">-37%</span></td>
</tr>
<tr>
  <td style="padding:5px 0;border-top:1px solid #334155;">RMSE</td>
  <td style="text-align:right;">8.18 min</td>
  <td style="text-align:right;">5.58 min</td>
  <td style="text-align:right;"><span class="improvement-badge">-32%</span></td>
</tr>
<tr>
  <td style="padding:5px 0;border-top:1px solid #334155;">R&sup2;</td>
  <td style="text-align:right;">0.633</td>
  <td style="text-align:right;">0.829</td>
  <td style="text-align:right;"><span class="improvement-badge">+31%</span></td>
</tr>
</tbody>
</table>
</div>
""", unsafe_allow_html=True)

    # ── Project Insights ──────────────────────────────────────────────────────
    st.markdown('<p class="section-title">Project Insights</p>', unsafe_allow_html=True)
    insights = [
        ("Dataset", "70,996,150 cleaned trips from NYC Yellow Taxi (2023-2024)"),
        ("Split", "Chronological: Train 2023 | Val 2024 H1 | Test 2024 H2"),
        ("Key predictor", "Trip distance is the strongest single predictor of duration"),
        ("Time effects", "Rush hours (8-9am, 5-7pm) add significant duration"),
        ("Location", "Pickup/dropoff TLC zones encode neighbourhood-level traffic patterns"),
        ("Nonlinearity", "HGBR captures complex interactions that linear models miss"),
    ]
    for tag, text in insights:
        st.markdown(
            f'<div class="insight-item"><span class="tag">{tag}</span>{text}</div>',
            unsafe_allow_html=True
        )

    # ── Limitations ───────────────────────────────────────────────────────────
    with st.expander("About & Limitations"):
        st.markdown("""
**What this model does**
- Predicts NYC Yellow Taxi trip duration based on historical patterns.
- Trained on 35M+ trips from 2023 using a HistGradientBoostingRegressor.

**Limitations**
- Does not use live traffic data or real-time conditions.
- Does not account for weather, accidents, or events.
- Location IDs (PULocationID, DOLocationID) represent TLC taxi zones.
  Due to a scikit-learn HGBR categorical cardinality limit (max 255),
  the 262-zone location IDs are treated as numerical features.
- Predictions are statistical estimates, not guarantees.
- The model was trained on 2023 data and may drift over time.

**Validation methodology**
- Strict chronological split: the model never saw future data during training.
- The test set (2024 H2) was held out and evaluated exactly once.
- No hyperparameter tuning was performed after seeing test results.
        """)
