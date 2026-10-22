"""Week 2 logistics preprocessing.
Download DataCoSupplyChainDataset.csv from https://data.mendeley.com/datasets/8gx2fvg2k6/5
and place it beside this script. Install: pip install pandas numpy scikit-learn
Run: python logistics_week2_cleaning_improved.py
"""
from pathlib import Path
import pandas as pd

INPUT = Path("DataCoSupplyChainDataset.csv")

def inspect(df, title):
    print(f"\n--- {title} ---")
    print("Shape:", df.shape)
    print(df.head())
    print("\nTypes:\n", df.dtypes)
    missing = df.isna().sum().sort_values(ascending=False)
    print("\nMissing values:\n", missing[missing > 0].head(20) if (missing > 0).any() else "None")
    print("Exact duplicate rows:", int(df.duplicated().sum()))

def main():
    if not INPUT.exists():
        print(f"ERROR: {INPUT} not found. Download it and put it beside this script.")
        return
    df = pd.read_csv(INPUT, encoding="latin-1", low_memory=False)
    original_shape = df.shape
    inspect(df, "BEFORE CLEANING")

    # Make column names consistent.
    df.columns = (df.columns.astype(str).str.strip().str.lower()
                  .str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_"))

    # Remove only rows that match across every column.
    duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    # Convert likely date/time text fields; invalid values become NaT.
    date_results = []
    for col in df.columns:
        if ("date" in col or "time" in col) and (
            pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])
        ):
            before = int(df[col].notna().sum())
            df[col] = pd.to_datetime(df[col], errors="coerce")
            after = int(df[col].notna().sum())
            date_results.append({"column": col, "valid_before": before,
                                 "valid_after": after, "new_invalid": max(0, before-after)})

    # Save a quality summary before imputing.
    miss = df.isna().sum()
    quality = pd.DataFrame({
        "column": df.columns,
        "data_type": [str(df[c].dtype) for c in df.columns],
        "missing_count": [int(miss[c]) for c in df.columns],
        "missing_percent": [round(float(miss[c]) / max(len(df), 1) * 100, 2) for c in df.columns],
        "unique_values": [int(df[c].nunique(dropna=True)) for c in df.columns]
    })
    quality.to_csv("DataCo_data_quality_summary.csv", index=False)

    # Baseline treatment: important fields still need business review.
    for col in df.select_dtypes(include="number").columns:
        if df[col].isna().any() and pd.notna(df[col].median()):
            df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object", "string", "category"]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna("Unknown")

    # IQR flags: do not delete automatically; unusual values may be genuine.
    outlier_rows = []
    for col in ["sales", "order_item_quantity", "order_item_product_price",
                "order_item_total", "benefit_per_order"]:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].dropna()
            if len(s):
                q1, q3 = s.quantile(.25), s.quantile(.75)
                iqr = q3-q1
                low, high = q1-1.5*iqr, q3+1.5*iqr
                n = int(((s < low) | (s > high)).sum())
                outlier_rows.append({"column":col,"q1":q1,"q3":q3,"iqr":iqr,
                    "lower_bound":low,"upper_bound":high,"outlier_count":n,
                    "outlier_percent":round(n/len(s)*100,2)})
    if outlier_rows:
        pd.DataFrame(outlier_rows).to_csv("DataCo_outlier_review.csv", index=False)

    # Optional StandardScaler demonstration; preserve original numeric columns.
    try:
        from sklearn.preprocessing import StandardScaler
        cols = [c for c in ["sales","order_item_quantity","order_item_product_price"]
                if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if cols:
            valid = df[cols].notna().all(axis=1)
            if valid.any():
                values = StandardScaler().fit_transform(df.loc[valid, cols])
                for i, col in enumerate(cols):
                    df.loc[valid, col+"_standardized"] = values[:, i]
                print("Standardized columns:", cols)
    except ImportError:
        print("Scaling skipped: run pip install scikit-learn to enable it.")

    df.to_csv("DataCoSupplyChain_cleaned.csv", index=False)
    inspect(df, "AFTER CLEANING")
    print("\n--- ACTUAL RESULTS FROM THIS FILE ---")
    print("Original shape:", original_shape)
    print("Rows after cleaning:", len(df))
    print("Exact duplicate rows removed:", duplicates)
    print("Saved: DataCoSupplyChain_cleaned.csv")
    print("Saved: DataCo_data_quality_summary.csv")
    if outlier_rows:
        print("Saved: DataCo_outlier_review.csv")
        print(pd.DataFrame(outlier_rows).to_string(index=False))
    if date_results:
        print("Date conversion review:", date_results)
    print("Review outputs and business meaning before using the data.")

if __name__ == "__main__":
    main()
