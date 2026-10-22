"""
Week 3 Task - Advanced Data Analysis and Visualization in Logistics
Simulates a hypothetical Indian logistics dataset, performs EDA and builds visualizations.
Run:  python logistics_analysis.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

np.random.seed(42)
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.0)
OUT = "charts/"

# ---------------------------------------------------------------
# 1. SIMULATE THE LOGISTICS DATASET (2,500 shipments, 2025)
# ---------------------------------------------------------------
N = 2500
origins = ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Kolkata", "Pune", "Hyderabad", "Ahmedabad"]
dests = origins + ["Nagpur", "Jaipur", "Lucknow", "Kochi"]
modes = ["Road", "Rail", "Air", "Express Road"]
mode_p = [0.50, 0.20, 0.10, 0.20]
carriers = ["Carrier A", "Carrier B", "Carrier C", "Carrier D"]

df = pd.DataFrame({
    "shipment_id": [f"SHP{100000+i}" for i in range(N)],
    "ship_date": pd.to_datetime("2025-01-01") + pd.to_timedelta(np.random.randint(0, 365, N), unit="D"),
    "origin": np.random.choice(origins, N, p=[.18,.17,.14,.11,.10,.10,.10,.10]),
    "mode": np.random.choice(modes, N, p=mode_p),
    "carrier": np.random.choice(carriers, N, p=[.35,.30,.20,.15]),
})
df["destination"] = [np.random.choice([d for d in dests if d != o]) for o in df["origin"]]

# distance (km) - lognormal, mode dependent
base_dist = np.random.lognormal(mean=6.6, sigma=0.55, size=N).clip(80, 2500)
df["distance_km"] = np.where(df["mode"] == "Air", base_dist * 1.2, base_dist).round(0).clip(80, 2800)
# shipment weight (kg) and volume (m3)
df["weight_kg"] = np.random.lognormal(mean=5.3, sigma=0.9, size=N).clip(10, 8000).round(1)
df["volume_m3"] = (df["weight_kg"] / np.random.uniform(120, 260, N)).round(2)

# planned & actual transit time (days)
speed = {"Road": 450, "Rail": 380, "Air": 1800, "Express Road": 650}   # effective km/day
df["planned_days"] = (df["distance_km"] / df["mode"].map(speed) + np.where(df["mode"]=="Air", 0.5, 1.0)).round(1)
# delay risk: higher for road/rail, monsoon months (Jul-Sep), heavy loads, Carrier D
month = df["ship_date"].dt.month
monsoon = month.isin([7, 8, 9]).astype(int)
festive = month.isin([10, 11]).astype(int)
carrier_risk = df["carrier"].map({"Carrier A": 0.0, "Carrier B": 0.1, "Carrier C": 0.05, "Carrier D": 0.35})
mode_risk = df["mode"].map({"Road": 0.20, "Rail": 0.25, "Air": 0.05, "Express Road": 0.10})
delay_mean = 0.25 + 0.9*monsoon + 0.6*festive + 0.0003*df["weight_kg"] + 1.5*carrier_risk + 2.0*mode_risk
delay = np.random.gamma(shape=1.6, scale=delay_mean/1.6) - 0.35
df["delay_days"] = delay.clip(0, 8).round(1)
df["actual_days"] = (df["planned_days"] + df["delay_days"]).round(1)
df["on_time"] = (df["delay_days"] <= 0.3).astype(int)

# delay reasons (only for delayed shipments)
reasons = ["Traffic / Road Congestion", "Weather", "Vehicle Breakdown", "Customs / Documentation", "Warehouse Handling", "Driver Unavailability"]
def pick_reason(row):
    if row["on_time"] == 1: return "None"
    w = np.array([.28, .17, .10, .10, .22, .13])
    if row["ship_date"].month in (7, 8, 9): w = w * np.array([1.2, 2.2, 1, 1, 1, 1])
    if row["ship_date"].month in (10, 11): w = w * np.array([1.4, 1, 1, 1, 1.6, 1.2])
    if row["mode"] == "Air": w = w * np.array([0.2, 1.5, 0.1, 2.5, 1.2, 0.2])
    return np.random.choice(reasons, p=w / w.sum())
df["delay_reason"] = df.apply(pick_reason, axis=1)

# costs (INR)
rate = {"Road": 7.5, "Rail": 5.0, "Air": 38.0, "Express Road": 11.5}   # Rs per tonne-km-ish scaled
df["transport_cost"] = (df["distance_km"] * df["mode"].map(rate) * (0.6 + df["weight_kg"]/900)
                        * np.random.normal(1, 0.08, N)).round(0)
df["fuel_surcharge"] = (df["transport_cost"] * np.random.uniform(0.06, 0.14, N)).round(0)
df["handling_cost"] = (150 + df["weight_kg"] * np.random.uniform(0.4, 0.9, N)).round(0)
df["delay_penalty"] = (df["delay_days"] * np.random.uniform(120, 260, N) * (df["delay_days"] > 0.3)).round(0)
df["total_cost"] = df[["transport_cost", "fuel_surcharge", "handling_cost", "delay_penalty"]].sum(axis=1)
df["cost_per_km"] = (df["total_cost"] / df["distance_km"]).round(2)
df["cost_per_kg"] = (df["total_cost"] / df["weight_kg"]).round(2)
df["customer_rating"] = (5 - 0.55*df["delay_days"] - np.random.normal(0, 0.5, N)).clip(1, 5).round(1)

# inject a few missing values (realistic data quality issue) and clean them
for col, frac in [("customer_rating", 0.03), ("weight_kg", 0.01)]:
    idx = np.random.choice(df.index, int(N * frac), replace=False)
    df.loc[idx, col] = np.nan

df.to_csv("logistics_dataset.csv", index=False)

# ---------------------------------------------------------------
# 2. DATA QUALITY & CLEANING
# ---------------------------------------------------------------
res = {}
res["n_rows"], res["n_cols"] = df.shape
res["missing"] = df.isna().sum()[df.isna().sum() > 0].to_dict()
res["duplicates"] = int(df.duplicated("shipment_id").sum())
df["weight_kg"] = df["weight_kg"].fillna(df.groupby("mode")["weight_kg"].transform("median"))
df["customer_rating"] = df["customer_rating"].fillna(df["customer_rating"].median())
df["month"] = df["ship_date"].dt.to_period("M").dt.to_timestamp()
df["month_name"] = df["ship_date"].dt.strftime("%b")

# ---------------------------------------------------------------
# 3. DESCRIPTIVE STATISTICS
# ---------------------------------------------------------------
num_cols = ["distance_km", "weight_kg", "planned_days", "actual_days", "delay_days", "total_cost", "cost_per_km", "customer_rating"]
desc = df[num_cols].describe().T
desc["median"] = df[num_cols].median()
desc["skew"] = df[num_cols].skew()
desc["cv_%"] = desc["std"] / desc["mean"] * 100
res["describe"] = desc[["mean", "median", "std", "min", "max", "skew", "cv_%"]].round(2).reset_index().rename(columns={"index": "variable"}).to_dict("records")
mode_val = df["actual_days"].round(0).mode()[0]
res["mode_actual_days"] = float(mode_val)
res["on_time_pct"] = round(df["on_time"].mean() * 100, 1)
res["avg_delay_delayed"] = round(df.loc[df.on_time == 0, "delay_days"].mean(), 2)
res["total_cost_cr"] = round(df["total_cost"].sum() / 1e7, 2)
res["total_cost_lakh"] = round(df["total_cost"].sum() / 1e5, 1)
res["avg_cost_ship"] = round(df["total_cost"].mean(), 0)
res["total_weight_t"] = round(df["weight_kg"].sum() / 1000, 1)

# group tables
g_mode = df.groupby("mode").agg(shipments=("shipment_id", "count"), avg_distance=("distance_km", "mean"),
        avg_actual_days=("actual_days", "mean"), avg_delay=("delay_days", "mean"),
        on_time_pct=("on_time", lambda s: s.mean()*100), avg_cost=("total_cost", "mean"),
        cost_per_km=("cost_per_km", "mean")).round(2).reset_index()
res["by_mode"] = g_mode.to_dict("records")
g_car = df.groupby("carrier").agg(shipments=("shipment_id", "count"), avg_delay=("delay_days", "mean"),
        on_time_pct=("on_time", lambda s: s.mean()*100), avg_cost=("total_cost", "mean"),
        rating=("customer_rating", "mean")).round(2).reset_index()
res["by_carrier"] = g_car.to_dict("records")
g_org = df.groupby("origin").agg(shipments=("shipment_id", "count"), on_time_pct=("on_time", lambda s: s.mean()*100),
        avg_delay=("delay_days", "mean")).round(2).sort_values("on_time_pct").reset_index()
res["by_origin"] = g_org.to_dict("records")
g_month = df.groupby("month").agg(shipments=("shipment_id", "count"), on_time_pct=("on_time", lambda s: s.mean()*100),
        avg_delay=("delay_days", "mean"), avg_cost=("total_cost", "mean")).round(2).reset_index()
g_month["month"] = g_month["month"].dt.strftime("%b")
res["by_month"] = g_month.to_dict("records")
cost_comp = df[["transport_cost", "fuel_surcharge", "handling_cost", "delay_penalty"]].sum()
res["cost_share"] = (cost_comp / cost_comp.sum() * 100).round(1).to_dict()
reason_ct = df[df.on_time == 0]["delay_reason"].value_counts()
res["reasons"] = (reason_ct / reason_ct.sum() * 100).round(1).to_dict()

# correlations + hypothesis tests
corr_cols = ["distance_km", "weight_kg", "planned_days", "actual_days", "delay_days", "total_cost", "customer_rating"]
corr = df[corr_cols].corr()
res["corr_dist_cost"] = round(corr.loc["distance_km", "total_cost"], 2)
res["corr_weight_cost"] = round(corr.loc["weight_kg", "total_cost"], 2)
res["corr_delay_rating"] = round(corr.loc["delay_days", "customer_rating"], 2)
res["corr_dist_actual"] = round(corr.loc["distance_km", "actual_days"], 2)
f, p = stats.f_oneway(*[g["delay_days"].values for _, g in df.groupby("carrier")])
res["anova_carrier_F"], res["anova_carrier_p"] = round(f, 2), float(p)
monsoon_d = df[df.ship_date.dt.month.isin([7, 8, 9])]["delay_days"]
other_d = df[~df.ship_date.dt.month.isin([7, 8, 9])]["delay_days"]
t, p2 = stats.ttest_ind(monsoon_d, other_d, equal_var=False)
res["monsoon_mean"], res["other_mean"] = round(monsoon_d.mean(), 2), round(other_d.mean(), 2)
res["ttest_t"], res["ttest_p"] = round(t, 2), float(p2)
q1, q3 = df["total_cost"].quantile([.25, .75]); iqr = q3 - q1
res["cost_outliers"] = int(((df["total_cost"] > q3 + 1.5*iqr)).sum())
res["cost_q1"], res["cost_q3"] = float(q1), float(q3)

# ---------------------------------------------------------------
# 4. VISUALIZATIONS
# ---------------------------------------------------------------
mode_order = ["Road", "Rail", "Express Road", "Air"]
pal = dict(zip(mode_order, sns.color_palette("deep", 4)))

# Fig 1 distribution of actual delivery time
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
sns.histplot(df["actual_days"], bins=30, kde=True, color="#2E5C8A", ax=ax[0])
ax[0].axvline(df["actual_days"].mean(), color="crimson", ls="--", label=f"Mean {df['actual_days'].mean():.1f}")
ax[0].axvline(df["actual_days"].median(), color="green", ls=":", label=f"Median {df['actual_days'].median():.1f}")
ax[0].set(title="Distribution of Actual Delivery Time", xlabel="Days", ylabel="Shipments"); ax[0].legend()
sns.histplot(df["delay_days"], bins=30, kde=True, color="#E07B39", ax=ax[1])
ax[1].set(title="Distribution of Delay (Actual - Planned)", xlabel="Delay (days)", ylabel="Shipments")
plt.tight_layout(); plt.savefig(OUT + "fig1_distributions.png", dpi=170); plt.close()

# Fig 2 boxplot delivery time by mode + violin delay
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
sns.boxplot(data=df, x="mode", y="actual_days", order=mode_order, palette=pal, ax=ax[0])
ax[0].set(title="Delivery Time by Transport Mode", xlabel="", ylabel="Actual days")
sns.violinplot(data=df, x="mode", y="delay_days", order=mode_order, palette=pal, cut=0, ax=ax[1])
ax[1].set(title="Delay Spread by Transport Mode", xlabel="", ylabel="Delay (days)")
plt.tight_layout(); plt.savefig(OUT + "fig2_mode_boxplot.png", dpi=170); plt.close()

# Fig 3 monthly trend
fig, ax1 = plt.subplots(figsize=(10.5, 4.2))
mm = df.groupby("month").agg(ship=("shipment_id", "count"), ot=("on_time", "mean"), cost=("total_cost", "mean")).reset_index()
ax1.bar(mm["month"].dt.strftime("%b"), mm["ship"], color="#9DB9D9", label="Shipments")
ax1.set_ylabel("Shipments"); ax1.set_title("Monthly Shipment Volume and On-Time Delivery Rate")
ax2 = ax1.twinx(); ax2.plot(mm["month"].dt.strftime("%b"), mm["ot"]*100, color="crimson", marker="o", lw=2.2, label="On-time %")
ax2.set_ylabel("On-time %"); ax2.grid(False); ax2.set_ylim(0, 100)
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax1.legend(h1+h2, l1+l2, loc="lower left")
plt.tight_layout(); plt.savefig(OUT + "fig3_monthly_trend.png", dpi=170); plt.close()

# Fig 4 distance vs cost scatter
fig, ax = plt.subplots(figsize=(8.5, 5))
sns.scatterplot(data=df.sample(900, random_state=1), x="distance_km", y="total_cost", hue="mode", hue_order=mode_order,
                palette=pal, alpha=.65, s=28, ax=ax)
ax.set(title="Distance vs Total Shipment Cost", xlabel="Distance (km)", ylabel="Total cost (Rs)")
plt.tight_layout(); plt.savefig(OUT + "fig4_distance_cost.png", dpi=170); plt.close()

# Fig 5 correlation heatmap
fig, ax = plt.subplots(figsize=(7.5, 5.8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, linewidths=.5, ax=ax, cbar_kws={"shrink": .8})
ax.set_title("Correlation Matrix of Key Logistics Metrics")
plt.tight_layout(); plt.savefig(OUT + "fig5_correlation.png", dpi=170); plt.close()

# Fig 6 carrier performance + origin on-time
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
gc = g_car.sort_values("on_time_pct")
sns.barplot(data=gc, x="on_time_pct", y="carrier", color="#2E5C8A", ax=ax[0])
for i, v in enumerate(gc["on_time_pct"]): ax[0].text(v + 0.5, i, f"{v:.1f}%", va="center")
ax[0].set(title="On-Time Rate by Carrier", xlabel="On-time %", ylabel=""); ax[0].set_xlim(0, 100)
go = g_org.sort_values("on_time_pct")
sns.barplot(data=go, x="on_time_pct", y="origin", color="#E07B39", ax=ax[1])
for i, v in enumerate(go["on_time_pct"]): ax[1].text(v + 0.5, i, f"{v:.1f}%", va="center")
ax[1].set(title="On-Time Rate by Origin Hub", xlabel="On-time %", ylabel=""); ax[1].set_xlim(0, 100)
plt.tight_layout(); plt.savefig(OUT + "fig6_carrier_origin.png", dpi=170); plt.close()

# Fig 7 cost drivers: stacked cost by mode + delay reasons
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
cc = df.groupby("mode")[["transport_cost", "fuel_surcharge", "handling_cost", "delay_penalty"]].mean().loc[mode_order]
cc.plot(kind="bar", stacked=True, ax=ax[0], color=["#2E5C8A", "#E07B39", "#6BAA75", "#C0392B"], rot=0)
ax[0].set(title="Average Cost Structure by Mode", xlabel="", ylabel="Avg cost per shipment (Rs)")
ax[0].legend(["Transport", "Fuel surcharge", "Handling", "Delay penalty"], fontsize=8)
rs = reason_ct.sort_values()
ax[1].barh(rs.index, rs.values, color="#8E6BBF")
for i, v in enumerate(rs.values): ax[1].text(v + 5, i, str(v), va="center", fontsize=9)
ax[1].set(title="Root Causes of Delay (delayed shipments)", xlabel="Shipments")
plt.tight_layout(); plt.savefig(OUT + "fig7_cost_reasons.png", dpi=170); plt.close()

# Fig 8 heatmap month x reason / weekday-month on-time
pv = df.pivot_table(index="origin", columns="month_name", values="on_time", aggfunc="mean") * 100
pv = pv[["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]]
fig, ax = plt.subplots(figsize=(10.5, 4.6))
sns.heatmap(pv, annot=True, fmt=".0f", cmap="RdYlGn", vmin=20, vmax=95, linewidths=.4, ax=ax, cbar_kws={"label": "On-time %"})
ax.set(title="On-Time % by Origin Hub and Month", xlabel="", ylabel="")
plt.tight_layout(); plt.savefig(OUT + "fig8_hub_month_heatmap.png", dpi=170); plt.close()
res["worst_cell"] = pv.stack().idxmin(); res["worst_cell_val"] = round(pv.stack().min(), 0)
res["best_cell"] = pv.stack().idxmax(); res["best_cell_val"] = round(pv.stack().max(), 0)

with open("results.json", "w") as f: json.dump(res, f, indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
