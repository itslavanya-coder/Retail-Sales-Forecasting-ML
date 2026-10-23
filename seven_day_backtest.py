import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
import joblib

print("7-Day Forecast Backtesting Started...")

# Load daily historical data
df = pd.read_csv(
    "data/daily_sales.csv",
    parse_dates=["date"]
).sort_values("date").reset_index(drop=True)

feature_cols = joblib.load("advanced_feature_columns.pkl")
saved_model = joblib.load("advanced_sales_model.pkl")

# Create the same training features
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["day_of_year"] = df["date"].dt.dayofyear
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

for lag in [1, 7, 14, 30]:
    df[f"lag_{lag}"] = df["sales"].shift(lag)

for window in [7, 14, 30]:
    df[f"rolling_mean_{window}"] = (
        df["sales"].shift(1).rolling(window).mean()
    )

df = df.dropna().reset_index(drop=True)

# Use multiple historical 7-day periods
test_starts = [
    len(df) - 84,
    len(df) - 56,
    len(df) - 28
]

all_results = []

for fold, start in enumerate(test_starts, start=1):

    # Only data before forecast period
    history_df = df.iloc[:start].copy()

    model = clone(saved_model)

    model.fit(
        history_df[feature_cols],
        history_df["sales"]
    )

    sales_history = history_df["sales"].tolist()
    last_date = history_df["date"].iloc[-1]

    actual = df.iloc[start:start + 7]["sales"].to_numpy()

    predictions = []
    baseline_predictions = []

    for day in range(1, 8):

        future_date = last_date + pd.Timedelta(days=day)

        # Seasonal naive: repeat the previous week's pattern
        baseline_predictions.append(
    history_df["sales"].iloc[-7 + (day - 1)]
)

        features = pd.DataFrame([{
            "day_of_week": future_date.dayofweek,
            "month": future_date.month,
            "day_of_year": future_date.dayofyear,
            "is_weekend": int(future_date.dayofweek >= 5),
            "lag_1": sales_history[-1],
            "lag_7": sales_history[-7],
            "lag_14": sales_history[-14],
            "lag_30": sales_history[-30],
            "rolling_mean_7": np.mean(sales_history[-7:]),
            "rolling_mean_14": np.mean(sales_history[-14:]),
            "rolling_mean_30": np.mean(sales_history[-30:])
        }])

        prediction = max(
            0,
            float(model.predict(features[feature_cols])[0])
        )

        predictions.append(prediction)
        sales_history.append(prediction)

    model_mae = mean_absolute_error(actual, predictions)
    baseline_mae = mean_absolute_error(
        actual, baseline_predictions
    )

    all_results.append({
        "Fold": fold,
        "Model_MAE": model_mae,
        "Baseline_MAE": baseline_mae
    })

    print(
        f"Fold {fold}: Model MAE = {model_mae:.2f}, "
        f"Baseline MAE = {baseline_mae:.2f}"
    )

results = pd.DataFrame(all_results)

results.to_csv(
    "seven_day_backtest_results.csv",
    index=False
)

print("\nAverage Model MAE:", results["Model_MAE"].mean())
print("Average Baseline MAE:", results["Baseline_MAE"].mean())

plt.figure(figsize=(9, 5))

plt.plot(
    results["Fold"],
    results["Model_MAE"],
    marker="o",
    label="ML Model"
)

plt.plot(
    results["Fold"],
    results["Baseline_MAE"],
    marker="o",
    label="Baseline"
)

plt.title("7-Day Forecast Backtesting")
plt.xlabel("Fold")
plt.ylabel("MAE")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("seven_day_backtest.png")
plt.close()

print("\nSTEP 10A Completed Successfully!")