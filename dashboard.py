import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Retail Sales Forecasting",
    layout="wide"
)

st.title("Retail Sales Forecasting Dashboard")

st.write("Machine Learning Regression Project")

st.subheader("Best Model: Random Forest")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", "104.33")
col2.metric("RMSE", "285.20")
col3.metric("R² Score", "0.9415")

st.subheader("Model Comparison")

results = pd.read_csv("model_comparison.csv")
st.dataframe(results, use_container_width=True)

st.subheader("Sales Analysis")

for filename, title in [
    ("top_10_sales.png", "Top 10 Product Families"),
    ("monthly_sales.png", "Monthly Sales Trend"),
    ("model_comparison.png", "Model Comparison"),
    ("actual_vs_predicted.png", "Actual vs Predicted Sales")
]:
    if Path(filename).exists():
        st.subheader(title)
        st.image(filename, use_container_width=True)
    else:
        st.warning(f"{filename} not found")

st.success("Dashboard Loaded Successfully!")