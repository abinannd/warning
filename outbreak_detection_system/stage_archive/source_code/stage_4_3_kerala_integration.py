"""
Stage 4.3 — Full Kerala Integration
Merges Malabar (North Kerala, 2018-2023) + South Kerala synthetic patient data (2020-2024)
into a unified 12-district daily dataset and reruns the Z-score + Two-Tier pipeline.

Disease name normalization:
  Malabar → Unified     |   South Kerala → Unified
  Dengue   → Dengue     |   Dengue Fever  → Dengue
  Flu      → Influenza  |   (already Influenza)
  Chickenpox→ Chickenpox|   Chicken Pox   → Chickenpox
  (rest unchanged)
"""
import os, json, warnings
import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
base_dir    = r"C:\BRAIN-STORM\HT\warning\outbreak_detection_system"
data_dir    = os.path.join(base_dir, "data", "processed")
raw_sk      = r"C:\BRAIN-STORM\HT\warning\outbreak_detection_system\data\raw\raw1\south_kerala_synthetic_patients.xlsx"
reports_dir = os.path.join(base_dir, "reports", "kerala_integrated")
os.makedirs(reports_dir, exist_ok=True)

EPSILON      = 1e-6
MIN_DURATION = 2
MIN_PEAK     = 2

# ── Disease name normalization map ────────────────────────────────────────────
DISEASE_NORM = {
    # Malabar → unified
    "Dengue":      "Dengue",
    "Flu":         "Influenza",
    "Chickenpox":  "Chickenpox",
    # South Kerala → unified
    "Dengue Fever":"Dengue",
    "Chicken Pox": "Chickenpox",
    # All others pass through unchanged
}

def normalize_disease(name):
    return DISEASE_NORM.get(name, name)

# ──────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 1: Load Malabar training + test data (2018-2025)")
print("=" * 65)

df_train = pd.read_pickle(os.path.join(data_dir, "train_timeseries.pkl"))
df_test  = pd.read_pickle(os.path.join(data_dir, "test_detection_results.pkl"))

# Combine train + test for Malabar (we want the full daily series)
df_test_ts = df_test[["district", "disease_name", "diagnosis_date", "case_count"]].copy()
df_malabar = pd.concat(
    [df_train[["district","disease_name","diagnosis_date","case_count"]],
     df_test_ts],
    ignore_index=True
)
df_malabar["diagnosis_date"] = pd.to_datetime(df_malabar["diagnosis_date"])
df_malabar["disease_name"]   = df_malabar["disease_name"].map(normalize_disease)
df_malabar["region"]         = "North Kerala (Malabar)"

print(f"  Malabar rows: {len(df_malabar):,}")
print(f"  Districts: {sorted(df_malabar.district.unique())}")
print(f"  Diseases: {sorted(df_malabar.disease_name.unique())}")
print(f"  Date range: {df_malabar.diagnosis_date.min().date()} to {df_malabar.diagnosis_date.max().date()}")

# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2: Load + aggregate South Kerala patient data (2020-2024)")
print("=" * 65)

df_raw = pd.read_excel(raw_sk)
df_raw["diagnosis_date"] = pd.to_datetime(df_raw["diagnosis_date"])
df_raw["disease_name"]   = df_raw["disease_name"].map(normalize_disease)

# Aggregate patient records → daily counts per (district, disease, date)
daily_sk = (
    df_raw.groupby(["district", "disease_name", "diagnosis_date"])
    .size()
    .reset_index(name="case_count")
)

# Reindex to continuous daily date range per (district, disease) series
sk_reindexed = []
for (dist, dis), g in daily_sk.groupby(["district", "disease_name"]):
    g = g.set_index("diagnosis_date")[["case_count"]].sort_index()
    idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
    g = g.reindex(idx, fill_value=0).reset_index()
    g.columns = ["diagnosis_date", "case_count"]
    g["district"]     = dist
    g["disease_name"] = dis
    sk_reindexed.append(g)

df_sk = pd.concat(sk_reindexed, ignore_index=True)
df_sk["region"] = "South Kerala"

print(f"  South Kerala rows: {len(df_sk):,}")
print(f"  Districts: {sorted(df_sk.district.unique())}")
print(f"  Diseases: {sorted(df_sk.disease_name.unique())}")
print(f"  Date range: {df_sk.diagnosis_date.min().date()} to {df_sk.diagnosis_date.max().date()}")

# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3: Merge into unified all-Kerala dataset")
print("=" * 65)

df_all = pd.concat(
    [df_malabar[["district","disease_name","diagnosis_date","case_count","region"]],
     df_sk[["district","disease_name","diagnosis_date","case_count","region"]]],
    ignore_index=True
).sort_values(["district","disease_name","diagnosis_date"]).reset_index(drop=True)

print(f"  Combined rows:   {len(df_all):,}")
print(f"  Total districts: {df_all.district.nunique()} — {sorted(df_all.district.unique())}")
print(f"  Total diseases:  {df_all.disease_name.nunique()} — {sorted(df_all.disease_name.unique())}")
print(f"  Date range:      {df_all.diagnosis_date.min().date()} to {df_all.diagnosis_date.max().date()}")

# Save combined daily dataset
out_csv = os.path.join(reports_dir, "kerala_integrated_daily.csv")
df_all.to_csv(out_csv, index=False)
print(f"  Saved: {out_csv}")

# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4: Gap-corrected Z-score engine (all 12 districts)")
print("=" * 65)

def classify_risk(z):
    return np.select(
        [z < 2.0, (z >= 2.0) & (z < 2.5), (z >= 2.5) & (z < 3.0), z >= 3.0],
        ["Low", "Medium", "High", "Critical"], default="Low"
    )

stat_results = []
for (dist, dis), g in df_all.groupby(["district","disease_name"]):
    g = g.copy().sort_values("diagnosis_date").reset_index(drop=True)
    b_mean = g["case_count"].rolling(30, min_periods=10).mean().shift(8).fillna(0)
    b_std  = g["case_count"].rolling(30, min_periods=10).std().shift(8).fillna(0)
    r_mean = g["case_count"].rolling(7, min_periods=1).mean()
    std_safe = b_std.clip(lower=EPSILON)
    std_safe[b_std == 0] = np.nan
    g["rolling_mean"] = b_mean
    g["rolling_std"]  = b_std
    g["z_score"]      = ((r_mean - b_mean) / std_safe).fillna(0)
    g["risk_level"]   = classify_risk(g["z_score"])
    stat_results.append(g)

df_stat = pd.concat(stat_results, ignore_index=True)
n_series = df_stat.groupby(["district","disease_name"]).ngroups
print(f"  Z-score computed for {n_series} series across 12 districts")
print(f"  Max Z-score: {df_stat['z_score'].max():.4f}")
print(f"  Risk distribution:\n{df_stat['risk_level'].value_counts().to_string()}")

# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5: Two-Tier Event Bundling")
print("=" * 65)

risk_order = {"Low":0,"Medium":1,"High":2,"Critical":3}
rev_risk   = {1:"Medium",2:"High",3:"Critical"}
all_events = []
ctr = 1
df_stat["tier"] = "None"

for (dist, dis), grp in df_stat.groupby(["district","disease_name"]):
    grp = grp.sort_values("diagnosis_date")
    in_event, cur = False, None
    for idx, row in grp.iterrows():
        risk = row["risk_level"]
        if risk != "Low":
            if not in_event:
                in_event = True
                cur = {"District":dist,"Disease":dis,"Start":row["diagnosis_date"],
                       "End":row["diagnosis_date"],"Peak Cases":row["case_count"],
                       "Peak Z":round(row["z_score"],3),"max_risk":risk_order[risk],"idxs":[idx]}
            else:
                cur["End"]        = row["diagnosis_date"]
                cur["Peak Cases"] = max(cur["Peak Cases"], row["case_count"])
                cur["Peak Z"]     = max(cur["Peak Z"], row["z_score"])
                cur["max_risk"]   = max(cur["max_risk"], risk_order[risk])
                cur["idxs"].append(idx)
        else:
            if in_event:
                in_event = False
                dur = len(cur["idxs"])
                if dur >= MIN_DURATION:
                    tier = "Confirmed-Tier Event" if cur["Peak Cases"] >= MIN_PEAK else "Watch-Tier Event"
                    df_stat.loc[cur["idxs"], "tier"] = tier
                    cur.update({"Event ID":f"KL-{ctr:04d}","Duration":dur,"Tier":tier,
                                "Highest Risk":rev_risk.get(cur["max_risk"],"Medium")})
                    all_events.append({k:v for k,v in cur.items() if k!="idxs"})
                    ctr += 1
                cur = None
    if in_event and cur:
        dur = len(cur["idxs"])
        if dur >= MIN_DURATION:
            tier = "Confirmed-Tier Event" if cur["Peak Cases"] >= MIN_PEAK else "Watch-Tier Event"
            df_stat.loc[cur["idxs"], "tier"] = tier
            cur.update({"Event ID":f"KL-{ctr:04d}","Duration":dur,"Tier":tier,
                        "Highest Risk":rev_risk.get(cur["max_risk"],"Medium")})
            all_events.append({k:v for k,v in cur.items() if k!="idxs"})
            ctr += 1

ev_df = pd.DataFrame(all_events)
print(f"  Total events detected: {len(ev_df)}")

if not ev_df.empty:
    ev_df["Region"] = ev_df["District"].apply(
        lambda d: "North Kerala" if d in sorted(df_malabar.district.unique()) else "South Kerala"
    )
    print("\n  Events by Region & Tier:")
    ct = ev_df.groupby(["Region","Tier"]).size().unstack(fill_value=0)
    print(tabulate(ct.reset_index(), headers="keys", showindex=False, tablefmt="github"))

    print("\n  Events by District:")
    dc = ev_df.groupby(["District","Tier"]).size().unstack(fill_value=0)
    print(tabulate(dc.reset_index(), headers="keys", showindex=False, tablefmt="github"))

    print("\n  Top diseases by event count:")
    print(tabulate(
        ev_df.groupby("Disease")["Event ID"].count().sort_values(ascending=False).reset_index(name="Events"),
        headers="keys", showindex=False, tablefmt="github"
    ))

    ev_df.to_csv(os.path.join(reports_dir, "kerala_integrated_events.csv"), index=False)

# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6: Heatmap — all 12 districts × top diseases")
print("=" * 65)

df_all["month"] = df_all["diagnosis_date"].dt.month
# Pivot: rows = district-disease, cols = month
heat = df_all.groupby(["district","disease_name","month"])["case_count"].sum().reset_index()
heat_piv = heat.pivot_table(index=["district","disease_name"], columns="month", values="case_count", fill_value=0)
# Filter to disease-district pairs with meaningful activity (total > 10)
heat_piv = heat_piv[heat_piv.sum(axis=1) > 10]

fig, ax = plt.subplots(figsize=(16, max(8, len(heat_piv)*0.35)))
im = ax.imshow(heat_piv.values, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(12))
ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], fontsize=9)
ax.set_yticks(range(len(heat_piv)))
ylabels = [f"{d} / {dis}" for d, dis in heat_piv.index]
ax.set_yticklabels(ylabels, fontsize=7)
ax.set_title("Full Kerala Integration — Monthly Case Totals by District-Disease (All Years)", fontsize=12, pad=12)
plt.colorbar(im, ax=ax, label="Total Cases (all years)")
plt.tight_layout()
hm_path = os.path.join(reports_dir, "kerala_integrated_heatmap.png")
plt.savefig(hm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Heatmap saved: {hm_path}")

# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 7: Summary")
print("=" * 65)
conf  = len(ev_df[ev_df["Tier"]=="Confirmed-Tier Event"]) if not ev_df.empty else 0
watch = len(ev_df[ev_df["Tier"]=="Watch-Tier Event"])     if not ev_df.empty else 0
n_north = conf + watch if not ev_df.empty else 0

summary = {
    "total_rows_combined": int(len(df_all)),
    "total_districts": int(df_all.district.nunique()),
    "total_diseases": int(df_all.disease_name.nunique()),
    "date_range": [str(df_all.diagnosis_date.min().date()), str(df_all.diagnosis_date.max().date())],
    "total_events": int(len(ev_df)),
    "confirmed_tier": int(conf),
    "watch_tier": int(watch),
    "max_z_score": round(float(df_stat["z_score"].max()), 4),
    "top_disease": str(ev_df["Disease"].value_counts().idxmax()) if not ev_df.empty else "N/A",
    "north_kerala_events": int(len(ev_df[ev_df["Region"]=="North Kerala"])) if not ev_df.empty else 0,
    "south_kerala_events": int(len(ev_df[ev_df["Region"]=="South Kerala"])) if not ev_df.empty else 0,
}

with open(os.path.join(reports_dir, "kerala_integrated_summary.json"), "w") as f:
    json.dump(summary, f, indent=4)

print(f"  Combined daily rows:         {summary['total_rows_combined']:,}")
print(f"  Districts:                   {summary['total_districts']} (6 Malabar + 6 South Kerala)")
print(f"  Unique diseases:             {summary['total_diseases']}")
print(f"  Date span:                   {summary['date_range'][0]} to {summary['date_range'][1]}")
print(f"  Total events detected:       {summary['total_events']}")
print(f"    Confirmed-Tier Events:     {summary['confirmed_tier']}")
print(f"    Watch-Tier Events:         {summary['watch_tier']}")
print(f"    North Kerala (Malabar):    {summary['north_kerala_events']}")
print(f"    South Kerala:              {summary['south_kerala_events']}")
print(f"  Max Z-score:                 {summary['max_z_score']}")
print(f"  Most-flagged disease:        {summary['top_disease']}")
print(f"\nAll outputs -> {reports_dir}/")
print("Stage 4.3 complete.")
