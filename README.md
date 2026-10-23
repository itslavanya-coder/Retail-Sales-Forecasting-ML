Retail Sales Forecasting Using Machine Learning

Project Overview

This project analyzes historical retail sales data and builds machine learning regression models to predict sales quantities.

The project includes data preprocessing, exploratory data analysis, feature engineering, model training, evaluation, visualization, and a Streamlit dashboard.

Dataset

Dataset: Store Sales – Time Series Forecasting
Source: Kaggle

Dataset Link: https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data

Download train.csv from Kaggle.

Place train.csv inside the data folder before running sales_forecasting.py.
The dataset contains historical sales records with information such as dates, stores, product families, and promotions.

Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit

Machine Learning Models

Four regression algorithms were evaluated:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. K-Nearest Neighbors Regressor

Feature Engineering

The project uses calendar-based features, store information, product families, and promotional information.

Model Evaluation

A date-based train/test split was used for evaluation.

The model experiment uses a subset of 20,000 rows from the original dataset.

Best Model: Random Forest

Evaluation Metric| Result
MAE| 104.33
RMSE| 285.20
R² Score| 0.9415

These results describe performance on the selected test subset and do not guarantee future forecasting accuracy.

Visualizations

Top 10 Product Families

"Top 10 Sales" (top_10_sales.png)

Monthly Sales Trend

"Monthly Sales" (monthly_sales.png)

Model Comparison

"Model Comparison" (model_comparison.png)

Actual vs Predicted Sales

"Actual vs Predicted" (actual_vs_predicted.png)

Dashboard

A Streamlit dashboard displays model evaluation metrics, model comparisons, and sales visualizations.

Run the Project

Install the required libraries:

pip install -r requirements.txt

Run the machine learning project:

python sales_forecasting.py

Launch the dashboard:

python -m streamlit run dashboard.py

Limitations

- Model training uses a 20,000-row subset rather than the complete dataset.
- The current model uses historical records and a limited date-based evaluation.
- Additional testing across longer future periods is needed before using the model for production forecasting.
- Predictions represent sales quantities, not monetary revenue.

Conclusion

This project demonstrates an end-to-end machine learning workflow for retail sales analysis and regression modeling, supported by visualizations and a Streamlit dashboard.
