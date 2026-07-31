"""
Stage 4.1 - Independent Validation on COVID-19 Dataset
Applies the full outbreak detection pipeline (Z-score + Prophet) to a new 
unseen dataset (Kerala district COVID-19).
"""

import os
import json
import warnings
import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prophet import Prophet

warnings.filterwarnings('ignore')

base_dir    = r"C:\BRAIN-STORM\HT\warning\outbreak_detection_system"
raw_file    = os.path.join(base_dir, "data", "raw", "raw1", "kerala_district_covid_combined.csv")
proc_dir    = os.path.join(base_dir, "data", "processed")
reports_dir = os.path.join(base_dir, "reports", "covid_validation")
os.makedirs(reports_dir, exist_ok=True)

EPSILON      = 1e-6
MIN_DURATION = 2

def classify_risk(z_series):
    return np.select(
        [z_series < 2.0,
         (z_series >= 2.0) & (z_series < 2.5),
         (z_series >= 2.5) & (z_series < 3.0),
         z_series >= 3.0],
        ['Low', 'Medium', 'High', 'Critical'],
        default='Low'
    )

# --------------------------------------------------------------------------
# STEP 1 & 2: Load, format, filter to scope
# --------------------------------------------------------------------------
print("Step 1 & 2: Loading and formatting COVID-19 dataset...")
df = pd.read_csv(raw_file)

# Parse dates using mixed format
df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)

# Standardize district names
district_map = {
    'kaserigod': 'Kasaragod',
    'kozhikod': 'Kozhikode',
    'malpurram': 'Malappuram',
    'kannur': 'Kannur',
    'palakkad': 'Palakkad',
    'wayanad': 'Wayanad'
}
df['District'] = df['District'].str.lower().replace(district_map).str.title()

malabar_districts = ['Kasaragod', 'Kannur', 'Wayanad', 'Kozhikode', 'Malappuram', 'Palakkad']
df = df[df['District'].isin(malabar_districts)].copy()

# Keep only Confirmed as daily count
df = df[['District', 'Date', 'Confirmed']].rename(columns={
    'District': 'district',
    'Date': 'diagnosis_date',
    'Confirmed': 'case_count'
})
df['disease_name'] = 'COVID-19'

# --------------------------------------------------------------------------
# STEP 3: Fill missing dates
# --------------------------------------------------------------------------
print("Step 3: Building continuous daily date ranges...")
reindexed_dfs = []
for dist, g in df.groupby('district'):
    g = g.drop_duplicates(subset=['diagnosis_date']).set_index('diagnosis_date').sort_index()
    min_d, max_d = g.index.min(), g.index.max()
    idx = pd.date_range(min_d, max_d, freq='D')
    g = g.reindex(idx)
    g['district'] = dist
    g['disease_name'] = 'COVID-19'
    g['case_count'] = g['case_count'].fillna(0)
    g = g.reset_index().rename(columns={'index': 'diagnosis_date'})
    reindexed_dfs.append(g)

df_clean = pd.concat(reindexed_dfs, ignore_index=True)
df_clean = df_clean.sort_values(['district', 'diagnosis_date']).reset_index(drop=True)

clean_path = os.path.join(proc_dir, "covid_malabar_daily.csv")
df_clean.to_csv(clean_path, index=False)
print(f"  Saved cleaned dataset to: {clean_path}")

# Display row counts per district to pick flagship
print("\n  Data Coverage per District:")
cov_stats = df_clean.groupby('district').agg(
    Days=('diagnosis_date', 'count'),
    Total_Cases=('case_count', 'sum'),
    Min_Date=('diagnosis_date', 'min'),
    Max_Date=('diagnosis_date', 'max')
)
cov_stats['Min_Date'] = cov_stats['Min_Date'].dt.date
cov_stats['Max_Date'] = cov_stats['Max_Date'].dt.date
print(tabulate(cov_stats, headers='keys', tablefmt='github'))

# Pick flagship: district with most total cases
FLAGSHIP_DISTRICT = cov_stats['Total_Cases'].idxmax()
print(f"\n  => Selecting {FLAGSHIP_DISTRICT} as Flagship for Prophet Modeling.")

# --------------------------------------------------------------------------
# STEP 4: Run Statistical Engine (Gap-Corrected Z-Score + EWMA + Two-Tier)
# --------------------------------------------------------------------------
print("\nStep 4: Running gap-corrected statistical engine...")
results_stat = []
for dist, g in df_clean.groupby('district'):
    g = g.copy().reset_index(drop=True)
    b_mean = g['case_count'].rolling(30, min_periods=15).mean().shift(8).fillna(0)
    b_std  = g['case_count'].rolling(30, min_periods=15).std().shift(8).fillna(0)
    r_mean = g['case_count'].rolling(7, min_periods=1).mean()
    ewma   = g['case_count'].ewm(span=14, adjust=False).mean().shift(8).fillna(0)
    
    std_safe = b_std.copy()
    std_safe[std_safe > 0] = std_safe[std_safe > 0].clip(lower=EPSILON)
    std_safe[std_safe == 0] = np.nan
    
    z_raw = (r_mean - b_mean) / std_safe
    g['gap_z'] = z_raw.fillna(0)
    
    g['rolling_mean_30'] = b_mean
    g['rolling_std_30']  = b_std
    g['ewma_14']         = ewma
    g['rolling_z_score'] = g['gap_z']
    g['risk_level']      = classify_risk(g['rolling_z_score'])
    results_stat.append(g)

df_stat = pd.concat(results_stat, ignore_index=True)

# Two-Tier event extraction
print("  Extracting Two-Tier Events (Confirmed-Tier vs Watch-Tier)...")
df_stat['tier'] = 'None'
risk_order = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
reverse_risk = {1: 'Medium', 2: 'High', 3: 'Critical'}

all_events = []
ctr = 1
for dist, grp in df_stat.groupby('district'):
    grp = grp.sort_values('diagnosis_date')
    in_event = False
    cur = None
    
    for idx, row in grp.iterrows():
        risk = row['risk_level']
        if risk != 'Low':
            if not in_event:
                in_event = True
                cur = {
                    'District': dist, 'Disease': 'COVID-19',
                    'Start': row['diagnosis_date'],
                    'End': row['diagnosis_date'],
                    'Peak Cases': row['case_count'],
                    'Peak Z': row['rolling_z_score'],
                    'max_risk': risk_order[risk],
                    'idxs': [idx]
                }
            else:
                cur['End'] = row['diagnosis_date']
                cur['Peak Cases'] = max(cur['Peak Cases'], row['case_count'])
                cur['Peak Z'] = max(cur['Peak Z'], row['rolling_z_score'])
                cur['max_risk'] = max(cur['max_risk'], risk_order[risk])
                cur['idxs'].append(idx)
        else:
            if in_event:
                in_event = False
                dur = len(cur['idxs'])
                if dur >= MIN_DURATION:
                    tier_label = 'Confirmed-Tier Event' if cur['Peak Cases'] >= 2 else 'Watch-Tier Event'
                    df_stat.loc[cur['idxs'], 'tier'] = tier_label
                    cur['Event ID'] = f"CVD-{ctr:04d}"; ctr += 1
                    cur['Duration'] = dur
                    cur['Tier'] = tier_label
                    cur['Highest Risk'] = reverse_risk.get(cur['max_risk'], 'Medium')
                    all_events.append(dict(cur))
                cur = None

    if in_event:
        dur = len(cur['idxs'])
        if dur >= MIN_DURATION:
            tier_label = 'Confirmed-Tier Event' if cur['Peak Cases'] >= 2 else 'Watch-Tier Event'
            df_stat.loc[cur['idxs'], 'tier'] = tier_label
            cur['Event ID'] = f"CVD-{ctr:04d}"; ctr += 1
            cur['Duration'] = dur
            cur['Tier'] = tier_label
            cur['Highest Risk'] = reverse_risk.get(cur['max_risk'], 'Medium')
            all_events.append(dict(cur))

ev_df = pd.DataFrame(all_events)
if not ev_df.empty:
    ev_df = ev_df.drop(columns=['idxs', 'max_risk'])
    ev_out_path = os.path.join(reports_dir, "covid_zscore_events.csv")
    ev_df.to_csv(ev_out_path, index=False)

# --------------------------------------------------------------------------
# STEP 5: Prophet Model on Flagship District
# --------------------------------------------------------------------------
print(f"\nStep 5: Training Prophet model on {FLAGSHIP_DISTRICT} COVID series...")
flagship_df = df_stat[df_stat['district'] == FLAGSHIP_DISTRICT].copy()
prophet_df = flagship_df[['diagnosis_date', 'case_count']].rename(
    columns={'diagnosis_date': 'ds', 'case_count': 'y'}
)

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    interval_width=0.95,
    changepoint_prior_scale=0.05
)
model.fit(prophet_df)
future = model.predict(prophet_df[['ds']])

results = prophet_df.merge(
    future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
    on='ds', how='left'
)
results['yhat_lower'] = results['yhat_lower'].clip(lower=0)
results['yhat']       = results['yhat'].clip(lower=0)
results['is_anomaly_high'] = results['y'] > results['yhat_upper']

anomalies_high = results[results['is_anomaly_high']]
mae = np.mean(np.abs(results['y'] - results['yhat']))

print(f"  Prophet anomalies (high-side): {len(anomalies_high)}")
print(f"  MAE over full {FLAGSHIP_DISTRICT} series: {mae:.4f}")

prophet_csv_path = os.path.join(reports_dir, f"prophet_predictions_{FLAGSHIP_DISTRICT.lower()}_covid.csv")
results.to_csv(prophet_csv_path, index=False)

# --------------------------------------------------------------------------
# STEP 6: Cross-Validate Prophet vs Z-score
# --------------------------------------------------------------------------
print("\nStep 6: Comparing Prophet vs Z-score events...")
zscore_tier_days = set(
    flagship_df[flagship_df['tier'].isin(['Confirmed-Tier Event', 'Watch-Tier Event'])]
    ['diagnosis_date'].dt.normalize()
)
prophet_flagged = set(anomalies_high['ds'].dt.normalize())

overlap_tier      = prophet_flagged & zscore_tier_days
prophet_only_tier = prophet_flagged - zscore_tier_days
zscore_only_tier  = zscore_tier_days - prophet_flagged

# --------------------------------------------------------------------------
# STEP 7: Output & Summary
# --------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY: COVID-19 Independent Validation")
print("=" * 60)
if not ev_df.empty:
    print("Z-Score Events Detected per District:")
    evt_counts = ev_df.groupby(['District', 'Tier']).size().unstack(fill_value=0).reset_index()
    print(tabulate(evt_counts, headers='keys', showindex=False, tablefmt='github'))
    print(f"\nTotal Z-Score events across all 6 districts: {len(ev_df)}")
else:
    print("No Z-score events detected.")

print(f"\nFlagship District ({FLAGSHIP_DISTRICT}) Prophet Validation:")
print(f"  Prophet anomalies (high-side, actual > yhat_upper): {len(anomalies_high)}")
print(f"  Z-score Confirmed/Watch-Tier Event days:            {len(zscore_tier_days)}")
print(f"  Overlapping detections (both agree):                {len(overlap_tier)}")
print(f"  Prophet-only flags:                                 {len(prophet_only_tier)}")
print(f"  Z-score Tier Events missed by Prophet:              {len(zscore_only_tier)}")
print(f"  Prophet MAE:                                        {mae:.4f}")

# Save comparison summary
summary_dict = {
    "total_events": len(ev_df) if not ev_df.empty else 0,
    "flagship_district": FLAGSHIP_DISTRICT,
    "prophet_mae": round(mae, 4),
    "overlap_days": len(overlap_tier),
    "prophet_only_days": len(prophet_only_tier),
    "zscore_only_days": len(zscore_only_tier)
}
summary_path = os.path.join(reports_dir, "covid_validation_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary_dict, f, indent=4)

print(f"\nOutputs saved to: {reports_dir}/")
print("=" * 60)
print("Stage 4.1 complete.")
