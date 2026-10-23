import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/train.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Dataset information
print("\nDataset Shape:")
print(df.shape)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicates
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Data Cleaning

df["date"] = pd.to_datetime(df["date"])

df = df.drop_duplicates()

df = df.dropna()

print("\nData Cleaning Completed!")

print("Cleaned Dataset Shape:")
print(df.shape)

# Exploratory Data Analysis (EDA)

# Total Sales by Product Family
family_sales = df.groupby("family")["sales"].sum().sort_values(
    ascending=False
).head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x=family_sales.values, y=family_sales.index)

plt.title("Top 10 Product Families by Sales")
plt.xlabel("Total Sales")
plt.ylabel("Product Family")
plt.tight_layout()
plt.savefig("top_10_sales.png")
plt.close()

# Monthly Sales Trend
monthly_sales = df.groupby(
    df["date"].dt.to_period("M")
)["sales"].sum()

plt.figure(figsize=(10, 5))
monthly_sales.index = monthly_sales.index.to_timestamp()

plt.plot(monthly_sales.index, monthly_sales.values)

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("monthly_sales.png")
plt.close()

print("EDA Completed Successfully!")
print("Two graphs saved!")

# Feature Engineering

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek

df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# Convert product family into numerical values
df = pd.get_dummies(df, columns=["family"], dtype=int)

print("\nFeature Engineering Completed!")

print("Updated Dataset Shape:")
print(df.shape)

# Regression Model Training

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Select recent data for faster training
model_df = df.sort_values("date").tail(20000).copy()

# Select features
features = [
    "store_nbr",
    "onpromotion",
    "year",
    "month",
    "day",
    "day_of_week",
    "is_weekend"
]

features += [col for col in model_df.columns if col.startswith("family_")]

X = model_df[features]
y = model_df["sales"]

# Proper Time-Based Validation

unique_dates = model_df["date"].sort_values().unique()

split_date = unique_dates[int(len(unique_dates) * 0.8)]

train_mask = model_df["date"] < split_date
test_mask = model_df["date"] >= split_date

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

print("Training Rows:", len(X_train))
print("Testing Rows:", len(X_test))

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ),
    "KNN": KNeighborsRegressor(n_neighbors=5)
}

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    print(f"\n{name}")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))

results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df)

results_df.to_csv("model_comparison.csv", index=False)

print("\nModel Training Completed!")

# Model Comparison Visualization

plt.figure(figsize=(9, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="RMSE"
)

plt.title("Regression Model Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("RMSE (Lower is Better)")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig("model_comparison.png")
plt.close()

# Find Best Model
best_model = results_df.loc[results_df["RMSE"].idxmin()]

print("\nBEST REGRESSION MODEL")
print("Model:", best_model["Model"])
print("MAE:", round(best_model["MAE"], 2))
print("RMSE:", round(best_model["RMSE"], 2))
print("R2 Score:", round(best_model["R2 Score"], 4))

print("\nModel Comparison Graph Saved!")

# Actual vs Predicted Sales Visualization

# Train the selected Random Forest model
final_model = models["Random Forest"]

final_model.fit(X_train, y_train)

predicted_sales = final_model.predict(X_test)

# Create comparison graph
plt.figure(figsize=(10, 5))

plt.scatter(
    y_test,
    predicted_sales,
    alpha=0.4,
    s=10
)

max_value = max(y_test.max(), predicted_sales.max())

plt.plot(
    [0, max_value],
    [0, max_value],
    "r--"
)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales - Random Forest")

plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.close()

print("\nActual vs Predicted Graph Saved!")