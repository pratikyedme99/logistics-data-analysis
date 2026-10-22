import pandas as pd
import matplotlib.pyplot as plt

# Week 1 Logistics Analytics Starter
# Replace the filename and column names with those in your selected dataset.

df = pd.read_csv("logistics_data.csv")

print(df.head())
print(df.info())
print("\nMissing values:")
print(df.isna().sum())

df = df.drop_duplicates()

# Example:
# df["OrderDate"] = pd.to_datetime(df["OrderDate"])
# df["Revenue"] = df["Quantity"] * df["UnitPrice"]

# Example monthly demand:
# monthly_demand = df.groupby(df["OrderDate"].dt.to_period("M"))["Quantity"].sum()
# monthly_demand.plot(kind="line", title="Monthly Demand")
# plt.show()

# Next steps:
# 1. Clean the dataset
# 2. Explore KPIs
# 3. Create demand/delivery features
# 4. Build a baseline predictive model
# 5. Try clustering if appropriate
# 6. Develop route optimization logic
