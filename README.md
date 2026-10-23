Retail Sales Forecasting Using Machine Learning

Project Overview

An end-to-end machine learning project that analyzes historical retail sales data and predicts future sales using time-series forecasting techniques.

The project includes feature engineering, machine learning model comparison, hyperparameter tuning, walk-forward validation, seven-day forecasting, and an interactive Streamlit dashboard.

Dataset

Dataset: Store Sales – Time Series Forecasting (Kaggle)

The original dataset contains approximately 3 million sales records.

Daily sales are aggregated to create a time-series dataset for forecasting total sales quantities.

Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib
- Git and GitHub

Machine Learning Techniques

1. Data Preprocessing

- Data cleaning and preparation
- Daily sales aggregation
- Chronological data splitting

2. Feature Engineering

- Lag features: 1, 7, 14, and 30 days
- Rolling averages: 7, 14, and 30 days
- Calendar-based features

3. Model Development

- Random Forest Regressor
- HistGradientBoosting Regressor
- Seasonal Naive Baseline

4. Model Optimization

- RandomizedSearchCV
- TimeSeriesSplit
- Hyperparameter tuning

5. Model Evaluation

- Mean Absolute Error (MAE)
- Chronological train-test evaluation
- Walk-forward backtesting
- Seven-day recursive forecasting evaluation
- Comparison with a seasonal baseline

Seven-Day Forecast Backtesting Results

The model was evaluated across three seven-day forecasting windows.

Metric| Average MAE
Machine Learning Model| 48,937.47
Seasonal Baseline| 64,294.05

The ML model achieved approximately 23.9% lower average MAE than the seasonal baseline across these three evaluation windows.

The baseline performed better in one of the three windows, so the results should not be interpreted as guaranteed performance on unseen future data.

Interactive Dashboard

The Streamlit dashboard presents:

- Model performance
- Walk-forward backtesting
- Seven-day sales forecasts
- Seven-day backtesting results
- Model and baseline MAE comparisons

Project Files

File| Description
"advanced_forecasting.py"| Feature engineering, training, evaluation, and forecasting
"advanced_dashboard.py"| Interactive Streamlit dashboard
"seven_day_backtest.py"| Seven-day recursive forecasting evaluation
"seven_day_backtest_results.csv"| Backtesting metrics
"seven_day_backtest.png"| Backtesting visualization
"walk_forward_results.csv"| Walk-forward evaluation results
"future_7_day_forecast.csv"| Seven-day forecast output
"future_7_day_forecast.png"| Forecast visualization

How to Run

1. Download the Kaggle Store Sales – Time Series Forecasting dataset.
2. Place "train.csv" inside the project's "data" folder.
3. Install the required Python dependencies.
4. Run "advanced_forecasting.py".
5. Run "seven_day_backtest.py".
6. Launch the dashboard using:

python -m streamlit run advanced_dashboard.py

Limitations

- The project predicts aggregated daily sales quantities rather than individual store-level sales.
- The seven-day backtesting evaluation uses three historical windows.
- Forecast dates follow the historical dataset period rather than the current calendar date.
- The system is a portfolio demonstration and is not a production forecasting service.

Conclusion

This project demonstrates practical machine learning and time-series forecasting skills, including data preprocessing, feature engineering, model training, hyperparameter optimization, baseline comparisons, backtesting, forecasting, and dashboard development.

Project Category: Machine Learning | Time Series Forecasting | Retail Analytics
