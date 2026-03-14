"""
Rossmann Sales Forecasting — Streamlit App
Deploy: streamlit run app.py
Required files in same directory: xgb_model.pkl, feature_columns.pkl, model_stats.pkl
Optional:                          actual_vs_predicted.png, feature_importance.png
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rossmann Sales Forecast",
    page_icon="🛒",
    layout="wide",
)

# ── Load artefacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load("xgb_model.pkl")
    features = joblib.load("feature_columns.pkl")
    stats    = joblib.load("model_stats.pkl")
    return model, features, stats

try:
    model, feature_columns, stats = load_model()
    model_loaded = True
except FileNotFoundError as e:
    model_loaded = False
    missing = str(e)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🛒 Rossmann Store Sales Forecasting")
st.markdown("Powered by **XGBoost** — predict daily store sales from store & date features.")
st.divider()

if not model_loaded:
    st.error(f"⚠️ Model file not found: `{missing}`\n\nPlease run the Colab notebook first and place the `.pkl` files in this directory.")
    st.stop()

# ── Sidebar — Model metrics ────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Model Performance")
    st.metric("R² (actual sales)",  f"{stats['r2_actual']:.4f}")
    st.metric("RMSE (actual sales)", f"{stats['rmse_actual']:,.0f} €")
    st.metric("MAE (actual sales)",  f"{stats['mae_actual']:,.0f} €")
    st.metric("MAPE",               f"{stats['mape']/100000000:.2f} %")
    st.metric("Log-scale R²",       f"{stats['r2_log']:.4f}")
    st.caption("Metrics on held-out 20 % test set")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📈 Model Insights", "ℹ️ About"])

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 1 — PREDICT                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab1:
    st.subheader("Input Store & Date Details")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Store Info**")
        store_id      = st.number_input("Store ID",           min_value=1, max_value=1115, value=1)
        store_type    = st.selectbox("Store Type",            ["Micro Scale Enterprise", "Small Scale Enterprise", "Medium Scale Enterprise", "Large Scale Enterprise"])
        assortment    = st.selectbox("Assortment",            ["basic", "extra", "extended"])
        comp_distance = st.number_input("Competition Distance (m)", min_value=0, max_value=100000, value=1000)

    with col2:
        st.markdown("**Date & Holiday Info**")
        import datetime
        date_input   = st.date_input("Date", value=datetime.date.today())
        state_holiday   = st.selectbox("State Holiday",       ["No Holiday", "Public Holiday", "Easter", "Christmas"])
        school_holiday  = st.selectbox("School Holiday",      [0, 1], format_func=lambda x: "Yes" if x else "No")
        comp_open_month = st.number_input("Competition Open Since Month", min_value=0, max_value=12, value=0)
        comp_open_year  = st.number_input("Competition Open Since Year",  min_value=0, max_value=2025, value=0)

    st.divider()

    if st.button("🔮 Predict Sales", type="primary", use_container_width=True):

        # ── Derive time features ───────────────────────────────────────────────
        dt         = pd.Timestamp(date_input)
        year       = dt.year
        month      = dt.month
        day        = dt.day
        week       = dt.isocalendar()[1]
        day_of_week= dt.isoweekday()            # 1=Mon … 7=Sun
        is_weekend = int(day_of_week in [6, 7])

        day_map = {1:"Monday",2:"Tuesday",3:"Wednesday",4:"Thursday",
                   5:"Friday",6:"Saturday",7:"Sunday"}
        day_name = day_map[day_of_week]

        # ── Build raw row (mirroring training pipeline) ────────────────────────
        raw = {
            "Store"                    : store_id,
            "DayOfWeek"                : day_of_week,
            "Customers"                : 0,           # unknown at predict time
            "Open"                     : 1,
            "SchoolHoliday"            : school_holiday,
            "StoreType"                : store_type,
            "Assortment"               : assortment,
            "CompetitionDistance"      : comp_distance,
            "CompetitionOpenSinceMonth": comp_open_month,
            "CompetitionOpenSinceYear" : comp_open_year,
            "Year"                     : year,
            "Month"                    : month,
            "Day"                      : day,
            "WeekOfYear"               : week,
            "DayName"                  : day_name,
            "IsWeekend"                : is_weekend,
            "StateHoliday"             : state_holiday,
        }

        df_input = pd.DataFrame([raw])

        # One-hot encode
        cat_cols = ["StoreType", "Assortment", "StateHoliday", "DayName"]
        df_enc   = pd.get_dummies(df_input, columns=cat_cols, drop_first=True)

        # Align to training feature columns
        for col in feature_columns:
            if col not in df_enc.columns:
                df_enc[col] = 0
        df_enc = df_enc[feature_columns]

        # ── Predict ────────────────────────────────────────────────────────────
        log_pred     = model.predict(df_enc)[0]
        sales_pred   = max(np.expm1(log_pred), 0)

        # ── Display result ─────────────────────────────────────────────────────
        st.success(f"### 💰 Predicted Sales: **€ {sales_pred:,.0f}**")

        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Store",   f"#{store_id}")
        res_col2.metric("Date",    str(date_input))
        res_col3.metric("Day",     day_name)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 2 — MODEL INSIGHTS                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab2:
    st.subheader("Actual vs Predicted Sales")

    if os.path.exists("actual_vs_predicted.png"):
        st.image("actual_vs_predicted.png", use_column_width=True)
    else:
        st.info("Run the Colab notebook to generate `actual_vs_predicted.png` and place it here.")

    st.divider()
    st.subheader("Top 20 Feature Importances")

    if os.path.exists("feature_importance.png"):
        st.image("feature_importance.png", use_column_width=True)
    else:
        st.info("Run the Colab notebook to generate `feature_importance.png` and place it here.")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 3 — ABOUT                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab3:
    st.subheader("About this project")
    st.markdown("""
**Dataset**: Rossmann Store Sales (Kaggle)  
**Target**: Daily store sales (log-transformed during training)  
**Model**: XGBoost Regressor  

### Pipeline summary
1. Merge `train.csv` + `store.csv` on `Store`
2. Parse `Date` → Year / Month / Day / WeekOfYear / DayName
3. Drop closed-store rows (`Open == 0`)
4. Fill missing competition fields
5. One-hot encode: StoreType, Assortment, StateHoliday, DayName
6. Log-transform target (`log1p`)
7. Train XGBoost (500 trees, lr=0.05, depth=6)
8. Evaluate on 20 % hold-out set

### Files needed for deployment
| File | Purpose |
|------|---------|
| `app.py` | This Streamlit app |
| `xgb_model.pkl` | Trained XGBoost model |
| `feature_columns.pkl` | Ordered feature list (alignment) |
| `model_stats.pkl` | Evaluation metrics |
| `actual_vs_predicted.png` | Evaluation plot (optional) |
| `feature_importance.png` | Importance plot (optional) |

### Run locally
```bash
pip install streamlit xgboost pandas numpy joblib
streamlit run app.py
```
""")
