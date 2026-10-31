"""Week 4: Predictive modeling and optimization in logistics.
Dataset is synthetic for educational practice, not real company data.
Install: pip install pandas numpy scikit-learn
Keep this script and Week4_Logistics_Simulated_Dataset.csv in the same folder.
Run: python Week4_Logistics_Predictive_Modeling.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

FILE = Path("Week4_Logistics_Simulated_Dataset.csv")

def main():
    if not FILE.exists():
        raise FileNotFoundError("Put Week4_Logistics_Simulated_Dataset.csv beside this script.")
    df = pd.read_csv(FILE)
    print("Shape:", df.shape)
    print("\\nSample rows:\\n", df.head())
    print("\\nMissing values:\\n", df.isna().sum())
    print("\\nTarget summary:\\n", df["delivery_days"].describe())

    X = df.drop(columns=["delivery_days", "shipment_id"])
    y = df["delivery_days"]
    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, random_state=42
    )
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=250, random_state=42, min_samples_leaf=2, n_jobs=-1
        )
    }
    fitted, rows = {}, []
    for name, model in models.items():
        pipe = Pipeline([("preprocess", prep), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append({
            "model": name,
            "MAE_days": mean_absolute_error(y_test, pred),
            "RMSE_days": np.sqrt(mean_squared_error(y_test, pred)),
            "R2": r2_score(y_test, pred)
        })
        fitted[name] = pipe
    results = pd.DataFrame(rows).sort_values("MAE_days")
    results.to_csv("Week4_Model_Evaluation.csv", index=False)
    print("\\nTEST RESULTS:\\n", results.to_string(index=False))
    best_name = results.iloc[0]["model"]
    best = fitted[best_name]
    scenario = X_test.copy()
    scenario["route_congestion_pct"] = (scenario["route_congestion_pct"] - 20).clip(lower=0)
    baseline = best.predict(X_test)
    improved = best.predict(scenario)
    compare = pd.DataFrame({
        "actual_delivery_days": y_test.to_numpy(),
        "baseline_predicted_days": baseline,
        "scenario_predicted_days": improved,
        "predicted_days_saved": baseline - improved
    })
    compare.to_csv("Week4_Optimization_Scenario.csv", index=False)
    print("\\nBest model by MAE:", best_name)
    print("Mean baseline prediction:", round(float(baseline.mean()), 3), "days")
    print("Mean scenario prediction:", round(float(improved.mean()), 3), "days")
    print("Mean predicted days saved:", round(float(compare["predicted_days_saved"].mean()), 3))
    print("\\nSaved Week4_Model_Evaluation.csv and Week4_Optimization_Scenario.csv")
    print("Scenario is illustrative; it does not prove real-world causation.")

if __name__ == "__main__":
    main()
