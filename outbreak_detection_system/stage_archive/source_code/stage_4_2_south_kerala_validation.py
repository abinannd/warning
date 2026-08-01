"""
Stage 4.2 — South Kerala Independent Validation
Patient-level synthetic dataset → daily aggregation → Z-score pipeline + Two-Tier alerts.
6 South Kerala districts, 17 diseases, 2020-2024.
"""
import os
import json
import warnings
import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

base_dir    = r"C:\BRAIN-STORM\HT\warning\outbreak_detection_system"
raw_file    = os.path.join(base_dir, "data", "raw", "raw1", "south_kerala_synthetic_patients.xlsx")
proc_dir    = os.path.join(base_dir, "data", "processed")
reports_dir = os.path.join(base_dir, "reports", "south_kerala_validation")
os.makedirs(reports_dir, exist_ok=True)

EPSILON      = 1e-6
MIN_DURATION = 2
MIN_PEAK     = 2      # Confirmed-Tier threshold

# ── Step 1: Load + inspect ────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading and inspecting dataset")
print("=" * 60)
df_raw = pd.read_excel(raw_file)
df_raw["diagnosis_date"] = pd.to_datetime(df_raw["diagnosis_date"])

print(f"  Records: {len(df_raw):,}")
print(f"  Date range: {df_raw.diagnosis_date.min().date()} to {df_raw.diagnosis_date.max().date()}")
print(f"  Districts ({df_raw.district.nunique()}): {sorted(df_raw.district.unique())}")
print(f"  Diseases  ({df_raw.disease_name.nunique()}): {sorted(df_raw.disease_name.unique())}")
print(f"  Missing comorbidity: {df_raw.comorbidity.isna().sum()} rows")
print(f"  Severity breakdown:\n{df_raw.severity.value_counts().to_string()}")

# ── Step 2: Aggregate to daily counts ────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Aggregating to daily case counts")
print("=" * 60)
daily = (
    df_raw.groupby(["district", "disease_name", "diagnosis_date"])
    .size()
    .reset_index(name="case_count")
)

districts = sorted(daily["district"].unique().tolist())
diseases  = sorted(daily["disease_name"].unique().tolist())

# Reindex every (district, disease) to a continuous daily range
reindexed = []
for (dist, dis), g in daily.groupby(["district", "disease_name"]):
    g = g.set_index("diagnosis_date")[["case_count"]].sort_index()
    full_idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
    g = g.reindex(full_idx, fill_value=0).reset_index()
    g.columns = ["diagnosis_date", "case_count"]
    g["district"]     = dist
    g["disease_name"] = dis
    reindexed.append(g)

df_daily = pd.concat(reindexed, ignore_index=True)
df_daily = df_daily.sort_values(["district", "disease_name", "diagnosis_date"]).reset_index(drop=True)

print(f"  Daily rows after reindex: {len(df_daily):,}")
print("\n  Coverage per district:")
cov = df_daily.groupby(["district", "disease_name"]).agg(
    Days=("diagnosis_date", "count"),
    Total=("case_count", "sum"),
    Min=("diagnosis_date", "min"),
    Max=("diagnosis_date", "max")
)
cov["Min"] = cov["Min"].dt.date
cov["Max"] = cov["Max"].dt.date
print(tabulate(cov.reset_index(), headers="keys", showindex=False, tablefmt="github"))

csv_path = os.path.join(reports_dir, "south_kerala_daily.csv")
df_daily.to_csv(csv_path, index=False)
print(f"\n  Saved daily CSV: {csv_path}")

# ── Step 3: Gap-corrected Z-score pipeline ────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Running gap-corrected Z-score engine")
print("=" * 60)

def classify_risk(z):
    return np.select(
        [z < 2.0, (z >= 2.0) & (z < 2.5), (z >= 2.5) & (z < 3.0), z >= 3.0],
        ["Low", "Medium", "High", "Critical"], default="Low"
    )

stat_results = []
for (dist, dis), g in df_daily.groupby(["district", "disease_name"]):
    g = g.copy().sort_values("diagnosis_date").reset_index(drop=True)

    # Gap-corrected baseline: [T-37, T-8] shifted by 8 days
    b_mean = g["case_count"].rolling(30, min_periods=10).mean().shift(8).fillna(0)
    b_std  = g["case_count"].rolling(30, min_periods=10).std().shift(8).fillna(0)
    r_mean = g["case_count"].rolling(7, min_periods=1).mean()

    std_safe = b_std.clip(lower=EPSILON)
    std_safe[b_std == 0] = np.nan

    g["rolling_mean_30"] = b_mean
    g["rolling_std_30"]  = b_std
    g["ewma_14"]         = g["case_count"].ewm(span=14, adjust=False).mean()
    g["rolling_z_score"] = ((r_mean - b_mean) / std_safe).fillna(0)
    g["risk_level"]      = classify_risk(g["rolling_z_score"])
    stat_results.append(g)

df_stat = pd.concat(stat_results, ignore_index=True)
print(f"  Z-score computed for {df_stat.groupby(['district','disease_name']).ngroups} series")
print(f"  Max Z-score: {df_stat['rolling_z_score'].max():.4f}")
print(f"  Risk level distribution:\n{df_stat['risk_level'].value_counts().to_string()}")

# ── Step 4: Two-Tier event bundling ──────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Two-Tier Event Extraction")
print("=" * 60)

risk_order   = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
reverse_risk = {1: "Medium", 2: "High", 3: "Critical"}
all_events   = []
ctr = 1

df_stat["tier"] = "None"

for (dist, dis), grp in df_stat.groupby(["district", "disease_name"]):
    grp      = grp.sort_values("diagnosis_date")
    in_event = False
    cur      = None

    for idx, row in grp.iterrows():
        risk = row["risk_level"]
        if risk != "Low":
            if not in_event:
                in_event = True
                cur = {
                    "District": dist, "Disease": dis,
                    "Start": row["diagnosis_date"],
                    "End": row["diagnosis_date"],
                    "Peak Cases": row["case_count"],
                    "Peak Z": round(row["rolling_z_score"], 3),
                    "max_risk": risk_order[risk],
                    "idxs": [idx]
                }
            else:
                cur["End"]        = row["diagnosis_date"]
                cur["Peak Cases"] = max(cur["Peak Cases"], row["case_count"])
                cur["Peak Z"]     = max(cur["Peak Z"], row["rolling_z_score"])
                cur["max_risk"]   = max(cur["max_risk"], risk_order[risk])
                cur["idxs"].append(idx)
        else:
            if in_event:
                in_event = False
                dur = len(cur["idxs"])
                if dur >= MIN_DURATION:
                    tier = "Confirmed-Tier Event" if cur["Peak Cases"] >= MIN_PEAK else "Watch-Tier Event"
                    df_stat.loc[cur["idxs"], "tier"] = tier
                    cur["Event ID"]    = f"SK-{ctr:04d}"; ctr += 1
                    cur["Duration"]    = dur
                    cur["Tier"]        = tier
                    cur["Highest Risk"] = reverse_risk.get(cur["max_risk"], "Medium")
                    all_events.append({k: v for k, v in cur.items() if k != "idxs"})
                cur = None

    if in_event and cur:
        dur = len(cur["idxs"])
        if dur >= MIN_DURATION:
            tier = "Confirmed-Tier Event" if cur["Peak Cases"] >= MIN_PEAK else "Watch-Tier Event"
            df_stat.loc[cur["idxs"], "tier"] = tier
            cur["Event ID"]    = f"SK-{ctr:04d}"; ctr += 1
            cur["Duration"]    = dur
            cur["Tier"]        = tier
            cur["Highest Risk"] = reverse_risk.get(cur["max_risk"], "Medium")
            all_events.append({k: v for k, v in cur.items() if k != "idxs"})

ev_df = pd.DataFrame(all_events)
print(f"  Total events detected: {len(ev_df)}")
if not ev_df.empty:
    print("\n  Events by District & Tier:")
    ct = ev_df.groupby(["District", "Tier"]).size().unstack(fill_value=0)
    print(tabulate(ct.reset_index(), headers="keys", showindex=False, tablefmt="github"))
    print("\n  Events by Disease:")
    print(tabulate(ev_df.groupby("Disease")["Event ID"].count().reset_index(name="Count")
                   .sort_values("Count", ascending=False), headers="keys", showindex=False, tablefmt="github"))
    ev_df.to_csv(os.path.join(reports_dir, "south_kerala_events.csv"), index=False)

# ── Step 5: Seasonal heatmap plot ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Generating seasonal heatmap")
print("=" * 60)

df_daily["month"] = pd.to_datetime(df_daily["diagnosis_date"]).dt.month
monthly = df_daily.groupby(["disease_name", "month"])["case_count"].sum().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(14, 7))
im = ax.imshow(monthly.values, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(12))
ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
ax.set_yticks(range(len(monthly.index)))
ax.set_yticklabels(monthly.index, fontsize=9)
ax.set_title("South Kerala — Monthly Disease Case Totals (2020-2024)", fontsize=13, pad=12)
ax.set_xlabel("Month"); ax.set_ylabel("Disease")
plt.colorbar(im, ax=ax, label="Total Cases")
plt.tight_layout()
hm_path = os.path.join(reports_dir, "south_kerala_seasonal_heatmap.png")
plt.savefig(hm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Heatmap saved: {hm_path}")

# ── Step 6: Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY — South Kerala Independent Validation")
print("=" * 60)
confirmed = ev_df[ev_df["Tier"] == "Confirmed-Tier Event"] if not ev_df.empty else pd.DataFrame()
watch     = ev_df[ev_df["Tier"] == "Watch-Tier Event"]     if not ev_df.empty else pd.DataFrame()

summary = {
    "total_records": int(len(df_raw)),
    "districts": districts,
    "diseases": diseases,
    "date_range": [str(df_raw.diagnosis_date.min().date()), str(df_raw.diagnosis_date.max().date())],
    "total_events": int(len(ev_df)),
    "confirmed_tier": int(len(confirmed)),
    "watch_tier": int(len(watch)),
    "max_z_score": round(float(df_stat["rolling_z_score"].max()), 4),
    "top_disease_by_events": str(ev_df["Disease"].value_counts().idxmax()) if not ev_df.empty else "N/A"
}

with open(os.path.join(reports_dir, "south_kerala_summary.json"), "w") as f:
    json.dump(summary, f, indent=4)

print(f"  Total patient records:      {summary['total_records']:,}")
print(f"  Total events detected:      {summary['total_events']}")
print(f"    Confirmed-Tier Events:    {summary['confirmed_tier']}")
print(f"    Watch-Tier Events:        {summary['watch_tier']}")
print(f"  Max Z-score achieved:       {summary['max_z_score']}")
print(f"  Most-flagged disease:       {summary['top_disease_by_events']}")
print(f"\nAll outputs saved to: {reports_dir}/")
print("Stage 4.2 complete.")
