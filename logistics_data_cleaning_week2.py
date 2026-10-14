import pandas as pd
from sklearn.preprocessing import StandardScaler

INPUT = "DataCoSupplyChainDataset.csv"
OUTPUT = "DataCoSupplyChain_cleaned.csv"

# 1. Load data
df = pd.read_csv(INPUT, encoding="latin-1")
print("Original shape:", df.shape)

# 2. Inspect
print(df.head())
print(df.info())
print(df.isna().sum().sort_values(ascending=False).head(15))
print("Duplicate rows:", df.duplicated().sum())

# 3. Remove exact duplicate rows
df = df.drop_duplicates().copy()

# 4. Standardize column names
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
)

# 5. Convert date columns when they exist
for col in ["shipping_date_dateorders", "order_date_dateorders"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

# 6. Re-check missing values
print("\nMissing values after duplicate removal:")
print(df.isna().sum().sort_values(ascending=False).head(15))

# 7. Save cleaned working copy
df.to_csv(OUTPUT, index=False)
print("\nSaved:", OUTPUT)

# 8. Optional standardization for ML
features = ["order_item_quantity", "sales", "order_item_total"]
available = [c for c in features if c in df.columns]

if available:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(
        df[available].fillna(df[available].median())
    )
    print("Scaled columns:", available)
