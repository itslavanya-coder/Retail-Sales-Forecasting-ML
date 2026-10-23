import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Advanced Retail Sales Forecasting",
    layout="wide"
)

st.title("Advanced Retail Sales Forecasting")

st.write(
    "Machine Learning | Time-Series Forecasting | "
    "Hyperparameter Tuning | Walk-Forward Backtesting"
)

# Load saved model
model = joblib.load("advanced_sales_model.pkl")
features = joblib.load("advanced_feature_columns.pkl")

st.success("Trained ML Model Loaded Successfully!")

# Model comparison
st.header("1. Model Performance")

results = pd.read_csv("advanced_model_comparison.csv")

st.dataframe(results, use_container_width=True)

best = results.sort_values("MAE").iloc[0]

col1, col2, col3 = st.columns(3)

col1.metric("Best Model", best["Model"])
col2.metric("MAE", f'{best["MAE"]:.2f}')
col3.metric("R²", f'{best["R2"]:.4f}')

# Walk-forward backtesting
st.header("2. Walk-Forward Backtesting")

backtest = pd.read_csv("walk_forward_results.csv")

st.dataframe(backtest, use_container_width=True)

st.image(
    "walk_forward_backtesting.png",
    use_container_width=True
)

# Future forecast
st.header("3. Next 7 Days Sales Forecast")

future = pd.read_csv(
    "future_7_day_forecast.csv",
    parse_dates=["date"]
)

st.dataframe(future, use_container_width=True)

st.line_chart(
    future.set_index("date")["predicted_sales"]
)

st.image(
    "future_7_day_forecast.png",
    use_container_width=True
)

st.info(
    "Forecasts represent the seven days following "
    "the final date in the historical dataset, "
    "not the current calendar week."
)

st.caption(
    "Forecasting scope: daily aggregated sales quantity. "
    "Walk-forward results evaluate one-day-ahead predictions; "
    "seven-day recursive forecast accuracy has not yet been "
    "independently validated."
)

# STEP 10B: Seven-Day Forecast Backtesting Results

st.header("4. Seven-Day Forecast Backtesting")

seven_day_results = pd.read_csv(
    "seven_day_backtest_results.csv"
)

st.dataframe(
    seven_day_results,
    use_container_width=True
)

model_mae = seven_day_results["Model_MAE"].mean()
baseline_mae = seven_day_results["Baseline_MAE"].mean()

col1, col2 = st.columns(2)

col1.metric(
    "Average ML Model MAE",
    f"{model_mae:,.2f}"
)

col2.metric(
    "Average Baseline MAE",
    f"{baseline_mae:,.2f}"
)

st.image(
    "seven_day_backtest.png",
    use_container_width=True
)

if model_mae < baseline_mae:
    st.success(
        "ML Model outperformed the baseline "
        "on average across the tested periods."
    )
else:
    st.warning(
        "Baseline performed better on average "
        "across the tested periods."
    )

    # Display 7-Day Backtesting Average Results

st.header("7-Day Forecast Backtesting Results")

backtest_results = pd.read_csv(
    "seven_day_backtest_results.csv"
)

st.write("Backtesting Results:")
st.dataframe(backtest_results)

average_model_mae = backtest_results["Model_MAE"].mean()
average_baseline_mae = backtest_results["Baseline_MAE"].mean()

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Average ML Model MAE",
        f"{average_model_mae:,.2f}"
    )

with col2:
    st.metric(
        "Average Baseline MAE",
        f"{average_baseline_mae:,.2f}"
    )

st.image(
    "seven_day_backtest.png",
    use_container_width=True
)