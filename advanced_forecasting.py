import pandas as pd
import numpy as np

# STEP 2: Load Full Dataset

df = pd.read_csv(
    "data/train.csv",
    parse_dates=["date"]
)

print("Dataset Loaded Successfully!")

print("\nTotal Rows and Columns:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDate Range:")
print("Start Date:", df["date"].min())
print("End Date:", df["date"].max())

# Aggregate sales by date
daily_sales = (
    df.groupby("date", as_index=False)
    .agg(
        sales=("sales", "sum"),
        onpromotion=("onpromotion", "sum")
    )
    .sort_values("date")
)

print("\nDaily Sales Dataset:")
print(daily_sales.head())

print("\nTotal Days:", len(daily_sales))

daily_sales.to_csv(
    "data/daily_sales.csv",
    index=False
)

print("\nDaily Sales Dataset Saved Successfully!")

# STEP 3: Advanced Feature Engineering

# Calendar Features
daily_sales["day_of_week"] = daily_sales["date"].dt.dayofweek
daily_sales["month"] = daily_sales["date"].dt.month
daily_sales["day_of_year"] = daily_sales["date"].dt.dayofyear
daily_sales["is_weekend"] = (
    daily_sales["day_of_week"] >= 5
).astype(int)

# Lag Features - Previous Sales
daily_sales["lag_1"] = daily_sales["sales"].shift(1)
daily_sales["lag_7"] = daily_sales["sales"].shift(7)
daily_sales["lag_14"] = daily_sales["sales"].shift(14)
daily_sales["lag_30"] = daily_sales["sales"].shift(30)

# Rolling Average Features
previous_sales = daily_sales["sales"].shift(1)

daily_sales["rolling_mean_7"] = (
    previous_sales.rolling(window=7).mean()
)

daily_sales["rolling_mean_14"] = (
    previous_sales.rolling(window=14).mean()
)

daily_sales["rolling_mean_30"] = (
    previous_sales.rolling(window=30).mean()
)

# Remove rows with missing lag values
daily_sales = daily_sales.dropna().copy()

print("\nAdvanced Feature Engineering Completed!")

print("\nNew Features:")
print(daily_sales.columns.tolist())

print("\nFinal Dataset Shape:")
print(daily_sales.shape)

# Save engineered dataset
daily_sales.to_csv(
    "data/advanced_sales_features.csv",
    index=False
)

print("\nAdvanced Features Dataset Saved!")

# STEP 4: Time-Series Split and Baseline Model

from sklearn.metrics import mean_absolute_error
import numpy as np

# Sort data by date
daily_sales = daily_sales.sort_values("date").reset_index(drop=True)

# 80% Training and 20% Testing
split_index = int(len(daily_sales) * 0.8)

train_data = daily_sales.iloc[:split_index].copy()
test_data = daily_sales.iloc[split_index:].copy()

print("\nTime-Series Split Completed!")

print("Training Rows:", len(train_data))
print("Testing Rows:", len(test_data))

print("\nTraining Date Range:")
print(train_data["date"].min(), "to", train_data["date"].max())

print("\nTesting Date Range:")
print(test_data["date"].min(), "to", test_data["date"].max())

# Seasonal Naive Baseline
# Predict today's sales using sales from 7 days ago

baseline_predictions = test_data["lag_7"]

baseline_mae = mean_absolute_error(
    test_data["sales"],
    baseline_predictions
)

print("\nBaseline Model MAE:", round(baseline_mae, 2))

print("\nSTEP 4 Completed Successfully!")

# STEP 5: Advanced ML Model Training

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Input Features
feature_cols = [
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_30"
]

# Training and Testing Data
X_train = train_data[feature_cols]
y_train = train_data["sales"]

X_test = test_data[feature_cols]
y_test = test_data["sales"]

# Machine Learning Models
models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ),

    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        random_state=42
    )
}

results = []

# Add Baseline Results
results.append({
    "Model": "Seasonal Naive Baseline",
    "MAE": baseline_mae,
    "RMSE": np.sqrt(
        mean_squared_error(y_test, test_data["lag_7"])
    ),
    "R2": r2_score(y_test, test_data["lag_7"])
})

# Train and Evaluate Models
for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print(f"{name} MAE: {mae:.2f}")
    print(f"{name} RMSE: {rmse:.2f}")
    print(f"{name} R2: {r2:.4f}")

# Compare Results
results_df = pd.DataFrame(results)

results_df = results_df.sort_values("MAE")

print("\nMODEL COMPARISON RESULTS:")
print(results_df.to_string(index=False))

# Save Results
results_df.to_csv(
    "advanced_model_comparison.csv",
    index=False
)

print("\nSTEP 5 Completed Successfully!")

# STEP 6: Hyperparameter Tuning

from sklearn.model_selection import (
    RandomizedSearchCV,
    TimeSeriesSplit
)

print("\nSTEP 6: Hyperparameter Tuning Started...")

# Time-Series Cross Validation
tscv = TimeSeriesSplit(n_splits=3)

# Random Forest Parameters
param_grid = {
    "n_estimators": [100, 150, 200],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# Hyperparameter Search
rf_tuning = RandomizedSearchCV(
    estimator=RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    ),
    param_distributions=param_grid,
    n_iter=10,
    cv=tscv,
    scoring="neg_mean_absolute_error",
    random_state=42,
    n_jobs=-1
)

# Train on Training Data Only
rf_tuning.fit(X_train, y_train)

# Best Model
best_rf = rf_tuning.best_estimator_

print("\nBest Parameters:")
print(rf_tuning.best_params_)

# Test Best Model
tuned_predictions = best_rf.predict(X_test)

tuned_mae = mean_absolute_error(
    y_test,
    tuned_predictions
)

tuned_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        tuned_predictions
    )
)

tuned_r2 = r2_score(
    y_test,
    tuned_predictions
)

print("\nTuned Random Forest Results:")
print("MAE:", round(tuned_mae, 2))
print("RMSE:", round(tuned_rmse, 2))
print("R2:", round(tuned_r2, 4))

# Save Tuned Model Results
tuned_result = pd.DataFrame([{
    "Model": "Tuned Random Forest",
    "MAE": tuned_mae,
    "RMSE": tuned_rmse,
    "R2": tuned_r2
}])

results_df = pd.concat(
    [results_df, tuned_result],
    ignore_index=True
)

results_df = results_df.sort_values("MAE")

results_df.to_csv(
    "advanced_model_comparison.csv",
    index=False
)

print("\nUpdated Model Comparison:")
print(results_df.to_string(index=False))

print("\nSTEP 6 Completed Successfully!")

# STEP 7: Walk-Forward Backtesting

from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

print("\nSTEP 7: Walk-Forward Backtesting Started...")

# Use training data only for backtesting
walk_forward = TimeSeriesSplit(n_splits=5)

backtest_results = []

for fold, (train_idx, val_idx) in enumerate(
    walk_forward.split(X_train), start=1
):

    X_fold_train = X_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]

    y_fold_train = y_train.iloc[train_idx]
    y_fold_val = y_train.iloc[val_idx]

    # Fresh model for every fold
    model = clone(best_rf)

    model.fit(X_fold_train, y_fold_train)

    fold_predictions = model.predict(X_fold_val)

    fold_mae = mean_absolute_error(
        y_fold_val,
        fold_predictions
    )

    fold_rmse = np.sqrt(
        mean_squared_error(
            y_fold_val,
            fold_predictions
        )
    )

    fold_r2 = r2_score(
        y_fold_val,
        fold_predictions
    )

    backtest_results.append({
        "Fold": fold,
        "MAE": fold_mae,
        "RMSE": fold_rmse,
        "R2": fold_r2
    })

    print(
        f"Fold {fold} | "
        f"MAE: {fold_mae:.2f} | "
        f"RMSE: {fold_rmse:.2f} | "
        f"R2: {fold_r2:.4f}"
    )

# Save Backtesting Results
backtest_df = pd.DataFrame(backtest_results)

backtest_df.to_csv(
    "walk_forward_results.csv",
    index=False
)

print("\nAverage Backtesting Results:")

print("Average MAE:", round(backtest_df["MAE"].mean(), 2))
print("Average RMSE:", round(backtest_df["RMSE"].mean(), 2))
print("Average R2:", round(backtest_df["R2"].mean(), 4))

# Plot Backtesting Results
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

plt.plot(
    backtest_df["Fold"],
    backtest_df["MAE"],
    marker="o"
)

plt.title("Walk-Forward Backtesting - MAE")
plt.xlabel("Fold")
plt.ylabel("MAE")
plt.xticks(backtest_df["Fold"])
plt.grid(True)

plt.tight_layout()
plt.savefig("walk_forward_backtesting.png")
plt.close()

print("\nWalk-Forward Backtesting Graph Saved!")
print("STEP 7 Completed Successfully!")

# STEP 8: Future 7-Day Sales Forecasting

print("\nSTEP 8: Future Forecasting Started...")

# Train final model using all available historical data
final_model = clone(best_rf)
X = daily_sales[feature_cols]
y = daily_sales["sales"]

final_model.fit(X, y)

# Keep historical daily sales
sales_history = daily_sales["sales"].tolist()

# Last available date
last_date = daily_sales["date"].max()

future_results = []

for day in range(1, 8):

    future_date = last_date + pd.Timedelta(days=day)

    # Create features using historical/predicted sales only
    future_features = pd.DataFrame([{
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

    prediction = float(
        final_model.predict(future_features[feature_cols])[0]
    )

    # Sales quantity cannot be negative
    prediction = max(0, prediction)

    future_results.append({
        "date": future_date,
        "predicted_sales": prediction
    })

    # Use prediction for subsequent future days
    sales_history.append(prediction)

future_df = pd.DataFrame(future_results)

future_df.to_csv(
    "future_7_day_forecast.csv",
    index=False
)

print("\nNEXT 7 DAYS SALES FORECAST:")
print(future_df.to_string(index=False))

# Forecast graph
plt.figure(figsize=(10, 5))

plt.plot(
    future_df["date"],
    future_df["predicted_sales"],
    marker="o"
)

plt.title("Next 7 Days Retail Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Predicted Sales Quantity")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

plt.savefig("future_7_day_forecast.png")
plt.close()

print("\nFuture Forecast Graph Saved!")
print("STEP 8 Completed Successfully!")

# STEP 9A: Save Trained Machine Learning Model

import joblib

joblib.dump(
    final_model,
    "advanced_sales_model.pkl"
)

joblib.dump(
    feature_cols,
    "advanced_feature_columns.pkl"
)

print("\nAdvanced ML Model Saved Successfully!")