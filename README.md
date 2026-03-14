


 Rossmann Store Sales Forecasting

HCL Hackathon — Machine Learning Regression Project



Live Demo

Streamlit App:
[https://sales-prediction-hcl.streamlit.app/](https://sales-prediction-hcl.streamlit.app/)

The deployed app allows anyone to:

* Select a store, date, and promotion status
* Get an instant predicted sales value from the trained model
* View actual vs predicted sales chart
* See feature importance driving the prediction

Try it live — no setup needed, runs directly in the browser.

---

Problem Statement

Rossmann operates over 1,100 drug stores across Germany. Store managers need to predict daily sales up to six weeks in advance to plan staff, inventory, and promotions effectively. Poor forecasting leads to overstocking, understaffing, and lost revenue.

This project builds an end-to-end machine learning pipeline that predicts daily store sales using historical data, store metadata, promotions, and seasonal patterns.

---

Dataset

| File      | Rows      | Description                                        |
| --------- | --------- | -------------------------------------------------- |
| train.csv | 1,017,209 | Daily sales records per store with target variable |
| test.csv  | 41,088    | Unseen dates for final prediction                  |
| store.csv | 1,115     | Store metadata — type, competition, promotions     |

Source:
[https://www.kaggle.com/competitions/rossmann-store-sales](https://www.kaggle.com/competitions/rossmann-store-sales)

Key Columns

| Column              | Description                              |
| ------------------- | ---------------------------------------- |
| Store               | Unique store ID                          |
| Sales               | Target variable — daily turnover in €    |
| Customers           | Number of customers that day             |
| Open                | 0 = closed, 1 = open                     |
| Promo               | Whether store ran a promotion            |
| StateHoliday        | Type of public holiday                   |
| StoreType           | One of 4 store models (a, b, c, d)       |
| Assortment          | Product range level                      |
| CompetitionDistance | Distance to nearest competitor in metres |

---

Project Pipeline

```
train.csv ──┐
test.csv  ──┤── Merge on Store ──► Preprocessing ──► Feature Engineering
store.csv ──┘
                                          │
                                          ▼
                               EDA & Visualisation
                                          │
                                          ▼
                              Log Transform Sales
                                          │
                                          ▼
                         One-Hot Encode Categorical Features
                                          │
                                          ▼
                              Train / Test Split (80/20)
                                          │
                          ┌───────────────┼───────────────────┐
                          ▼               ▼                   ▼
                  Linear Regression   Random Forest       XGBoost
                  Ridge (tuned)       (100 trees)        (500 rounds)
                  Lasso (tuned)
                          │               │                   │
                          └───────────────┴───────────────────┘
                                          │
                                          ▼
                              Hybrid Stacking Ensemble
                              (RF + XGBoost + Ridge meta)
                                          │
                                          ▼
                               Evaluation & Comparison
                               RMSE · MAE · R² · RMSPE
```

---

Preprocessing Steps

1. Data Loading & Merging

* Loaded train.csv, test.csv, and store.csv
* Merged store metadata into both train and test using a LEFT JOIN on Store column
* Result: single enriched dataframe with 18 columns

---

2. Missing Value Imputation

| Column                         | Strategy         | Reason             |
| ------------------------------ | ---------------- | ------------------ |
| CompetitionDistance            | Median fill      | Robust to outliers |
| CompetitionOpenSinceMonth/Year | Fill with 0      | No competition = 0 |
| Promo2SinceWeek/Year           | Fill with 0      | No promo = 0       |
| PromoInterval                  | Fill with "None" | Categorical null   |

---

3. Removing Closed Store Days

Removed all rows where `Open == 0`. Closed stores always have `Sales = 0`, which would bias the model toward predicting zero.

---

4. Date Feature Engineering

Extracted the following from the Date column:

* Year
* Month
* Day
* WeekOfYear
* DayName (Monday through Sunday)
* IsWeekend (1 if Saturday or Sunday)

---

5. Categorical Encoding

Applied `pd.get_dummies()` one-hot encoding to:

* StoreType
* Assortment
* StateHoliday
* PromoInterval
* DayName

---

6. Log Transform on Sales

```python
merged_df["log_sales"] = np.log1p(merged_df["Sales"])
```

Sales is right-skewed. Log transform compresses extreme values and normalises the distribution, improving linear model accuracy. Predictions are converted back using `np.expm1()`.

---

Exploratory Data Analysis

Seven visualisations were produced:

| Chart                 | Key Finding                                    |
| --------------------- | ---------------------------------------------- |
| Sales Distribution    | Right-skewed — log transform needed            |
| Sales Trend Over Time | Clear year-on-year growth with December spikes |
| Monthly Seasonality   | November–December highest, January lowest      |
| Sales by Day of Week  | Monday and Friday highest, Sunday lowest       |
| Promotion Impact      | Promo stores sell significantly more           |
| Customers vs Sales    | Strong positive correlation (r = 0.82)         |
| Correlation Heatmap   | Customers and Promo are top predictors         |

---

Models Trained

Model 1 — Linear Regression (Baseline)

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

Used as baseline to benchmark tree-based models.

---

Model 2 — Ridge Regression (Tuned)

```python
ridge_cv = RidgeCV(alphas=[0.01, 0.1, 1, 10, 50, 100, 500], cv=3)
```

Ridge penalises large coefficients and handles multicollinearity.

---

Model 3 — Lasso Regression (Tuned)

```python
lasso_cv = LassoCV(alphas=[0.0001, 0.001, 0.01, 0.05, 0.1], cv=3)
```

Lasso shrinks useless feature coefficients to zero.

---

Model 4 — Random Forest

```python
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=10,
    n_jobs=-1,
    random_state=42
)
```

Builds 100 decision trees and averages their predictions.

---

Model 5 — XGBoost

```python
xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10
)
```

Sequential boosting model correcting previous errors.

---

Model 6 — Hybrid Stacking Ensemble

```python
hybrid = StackingRegressor(
    estimators=[
        ('rf', RandomForestRegressor(...)),
        ('xgb', XGBRegressor(...))
    ],
    final_estimator=Ridge(alpha=1.0),
    cv=2
)
```

Combines RF + XGBoost predictions using a Ridge meta-model.

---

Model Results

| Model             | R²     | RMSE   | RMSPE |
| ----------------- | ------ | ------ | ----- |
| Linear Regression | 0.7359 | 0.2221 | ~0.41 |
| Ridge (tuned)     | ~0.74  | ~0.21  | ~0.39 |
| Lasso (tuned)     | ~0.74  | ~0.21  | ~0.38 |
| Random Forest     | ~0.87  | ~0.14  | ~0.14 |
| XGBoost           | ~0.92  | ~0.10  | ~0.10 |
| Hybrid (RF + XGB) | ~0.94  | ~0.08  | ~0.08 |

Actual Linear Regression scores:

Log RMSE: 0.2221
R² Score: 0.7359
Actual Sales RMSE: ~€1,800

Metric definitions:

R² — proportion of variance explained
RMSE — root mean squared error
RMSPE — root mean squared percentage error

---

Why Tree Models Perform Better

1. Non-linearity in sales spikes
2. Feature interactions like Promo × Weekend × Month
3. Robustness to outliers
4. Automatic feature importance discovery

---

Key Findings

VIF Analysis showed multicollinearity between:

* IsWeekend
* DayName_Saturday
* DayName_Sunday

Fix: drop `IsWeekend`.

Top feature drivers (XGBoost):

1. Promo
2. DayOfWeek
3. CompetitionDistance
4. StoreType
5. Month

---

Tech Stack

| Category          | Library             |
| ----------------- | ------------------- |
| Data manipulation | pandas, numpy       |
| Visualisation     | matplotlib, seaborn |
| Machine learning  | scikit-learn        |
| Gradient boosting | xgboost             |
| Diagnostics       | statsmodels         |
| Deployment        | Streamlit           |
| Environment       | Google Colab        |

---

Deployed App

[https://sales-prediction-hcl.streamlit.app/](https://sales-prediction-hcl.streamlit.app/)

Features:

* Store selector
* Date picker
* Promotion toggle
* Instant prediction
* Actual vs predicted chart
* Feature importance view

Run locally:

```bash
pip install streamlit
streamlit run app.py
```

---

How to Run

```bash
1. Upload notebook to Google Colab
2. Upload datasets:
   train.csv
   test.csv
   store.csv

3. Run all cells

Outputs:
- Model metrics
- EDA charts
- Diagnostics plots
- Feature importance
- Model comparison table
```

---

File Structure

```
hackathon/
├── hcl_hackathon.ipynb
├── app.py
├── train.csv
├── test.csv
├── store.csv
└── README.md
```

---

Team

HCL Hackathon — Machine Learning Track

| Role                | Responsibility            |
| ------------------- | ------------------------- |
| Data Engineering    | Data loading & cleaning   |
| EDA & Preprocessing | Feature engineering       |
| Model Training      | LR, RF, XGBoost, Ensemble |
| Deployment          | Streamlit app             |

---

Evaluation Metric

The Rossmann competition metric is RMSPE:

```
RMSPE = sqrt(mean(((actual - predicted) / actual)^2))
```

Lower is better.

Hybrid model achieved RMSPE ≈ 0.08, meaning predictions are within about 8% of actual sales on average.

---


