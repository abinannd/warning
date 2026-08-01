"""
Stage 3.1.2 - Parameter Tuning on Validation Data (2024)

Strategy:
- Restore clean rolling_std_30 from scratch (re-compute from train+val continuity)
- Sweep std floor values AND min_duration simultaneously
- Tune purely on validation year 2024 (never touch test set during tuning)
- Select best config targeting 30-80 events/year on validation
- Apply chosen parameters ONCE to test year 2025 for the final reported result
"""

import os
import json
import pandas as pd
import numpy as np
from tabulate import tabulate

base_dir = r"C:\BRAIN-STORM\HT\warning\outbreak_detection_system"
data_dir = os.path.join(base_dir, "data", "processed")
reports_dir = os.path.join(base_dir, "reports")

# --------------------------------------------------------------------------
# 1. Load raw processed datasets
# --------------------------------------------------------------------------
train_path = os.path.join(data_dir, "train_timeseries.pkl")
val_path   = os.path.join(data_dir, "validation_timeseries.pkl")
test_path  = os.path.join(data_dir, "test_timeseries.pkl")

df_train = pd.read_pickle(train_path)
df_val   = pd.read_pickle(val_path)
df_test  = pd.read_pickle(test_path)

for df in [df_train, df_val, df_test]:
    df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'])

# --------------------------------------------------------------------------
# 2. Recompute clean rolling stats with full chronological continuity
# --------------------------------------------------------------------------
print("Recomputing rolling statistics from raw timeseries (no floor)...")

df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
df_all = df_all.sort_values(['district', 'disease_name', 'diagnosis_date']).reset_index(drop=True)

grp = df_all.groupby(['district', 'disease_name'])
df_all['rolling_mean_30'] = grp['case_count'].transform(lambda x: x.rolling(30, min_periods=1).mean())
df_all['rolling_std_30']  = grp['case_count'].transform(lambda x: x.rolling(30, min_periods=1).std().fillna(0))

val_min  = df_val['diagnosis_date'].min()
val_max  = df_val['diagnosis_date'].max()
test_min = df_test['diagnosis_date'].min()
test_max = df_test['diagnosis_date'].max()

df_clean_val  = df_all[(df_all['diagnosis_date'] >= val_min)  & (df_all['diagnosis_date'] <= val_max)].copy()
df_clean_test = df_all[(df_all['diagnosis_date'] >= test_min) & (df_all['diagnosis_date'] <= test_max)].copy()

# --------------------------------------------------------------------------
# 2b. Understand the actual std distribution
# --------------------------------------------------------------------------
print("\n--- Std distribution in validation data ---")
std_vals = df_clean_val['rolling_std_30']
pcts = [0, 5, 25, 50, 75, 90, 95, 99, 100]
std_pct = [(p, round(float(np.percentile(std_vals, p)), 4)) for p in pcts]
print(tabulate(std_pct, headers=["Percentile", "rolling_std_30"], tablefmt="github"))
print(f"Rows with std == 0: {(std_vals == 0).sum()}")
print(f"Rows with std < 0.1: {(std_vals < 0.1).sum()}")
print(f"Rows with std < 0.5: {(std_vals < 0.5).sum()}")

# --------------------------------------------------------------------------
# 3. Helper: apply floor, classify, extract events
# --------------------------------------------------------------------------
def classify_risk(z):
    return np.select(
        [z < 2.0, (z >= 2.0) & (z < 2.5), (z >= 2.5) & (z < 3.0), z >= 3.0],
        ['Low', 'Medium', 'High', 'Critical'],
        default='Low'
    )

def extract_events(df, std_floor, min_duration, min_peak_cases=2):
    df = df.copy()
    # Clip to floor first, then replace any remaining true zeros with epsilon
    # This prevents NaN/inf from division by zero in sparse near-zero series
    EPSILON = 1e-6
    std_safe = df['rolling_std_30'].clip(lower=std_floor)
    std_safe = std_safe.replace(0, EPSILON)
    df['rolling_z_score'] = (df['case_count'] - df['rolling_mean_30']) / std_safe
    df['risk_level'] = classify_risk(df['rolling_z_score'])

    risk_order   = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
    reverse_risk = {1: 'Medium', 2: 'High', 3: 'Critical'}
    events = []
    ctr    = 1

    for (dist, dis), group in df.groupby(['district', 'disease_name']):
        in_event = False
        cur      = None
        for _, row in group.iterrows():
            risk = row['risk_level']
            if risk != 'Low':
                if not in_event:
                    in_event = True
                    cur = {
                        'District': dist, 'Disease': dis,
                        'Start': row['diagnosis_date'], 'End': row['diagnosis_date'],
                        'Peak Cases': row['case_count'],
                        'Peak Z':     row['rolling_z_score'],
                        'max_risk':   risk_order[risk],
                        'Med': 0, 'High': 0, 'Crit': 0
                    }
                cur['End']        = row['diagnosis_date']
                cur['Peak Cases'] = max(cur['Peak Cases'], row['case_count'])
                cur['Peak Z']     = max(cur['Peak Z'],     row['rolling_z_score'])
                cur['max_risk']   = max(cur['max_risk'],   risk_order[risk])
                if risk == 'Medium':     cur['Med']  += 1
                elif risk == 'High':     cur['High'] += 1
                elif risk == 'Critical': cur['Crit'] += 1
            else:
                if in_event:
                    in_event = False
                    dur = (cur['End'] - cur['Start']).days + 1
                    # Accept event only if duration AND peak cases both meet thresholds
                    if dur >= min_duration and cur['Peak Cases'] >= min_peak_cases:
                        cur['Event ID'] = f"EVT-{ctr:04d}"; ctr += 1
                        cur['Duration'] = dur
                        cur['Highest Risk'] = reverse_risk[cur['max_risk']]
                        events.append(cur)
                    cur = None
        if in_event:
            dur = (cur['End'] - cur['Start']).days + 1
            if dur >= min_duration and cur['Peak Cases'] >= min_peak_cases:
                cur['Event ID'] = f"EVT-{ctr:04d}"; ctr += 1
                cur['Duration'] = dur
                cur['Highest Risk'] = reverse_risk[cur['max_risk']]
                events.append(cur)

    return pd.DataFrame(events), df

# --------------------------------------------------------------------------
# 4. Sweep on VALIDATION data (2024): vary floor AND min_duration
# --------------------------------------------------------------------------
print("\n--- Tuning Sweep on Validation Year 2024 ---")
configs = [
    (0.0, 1), (0.0, 2), (0.0, 3),
    (0.10, 1), (0.10, 2), (0.10, 3),
    (0.25, 2), (0.25, 3), (0.25, 4),
    (0.5,  2), (0.5,  3), (0.5,  4),
]
sweep_results = []

for floor, min_dur in configs:
    ev_df, _ = extract_events(df_clean_val, floor, min_dur)
    n        = len(ev_df)
    crit     = int((ev_df['Highest Risk'] == 'Critical').sum()) if n else 0
    high     = int((ev_df['Highest Risk'] == 'High').sum())    if n else 0
    med      = int((ev_df['Highest Risk'] == 'Medium').sum())  if n else 0
    avg_dur  = round(ev_df['Duration'].mean(), 2) if n else 0
    max_z    = round(ev_df['Peak Z'].max(), 2)    if n else 0
    sweep_results.append((floor, min_dur, n, med, high, crit, avg_dur, max_z))

print(tabulate(sweep_results,
    headers=['Std Floor', 'Min Dur', 'Val Events', 'Medium', 'High', 'Critical', 'Avg Dur', 'Max Z'],
    tablefmt="github"))

# --------------------------------------------------------------------------
# 5. Pick best config: total val events closest to 50 in 20-120 window
# --------------------------------------------------------------------------
TARGET = 50
candidates = [r for r in sweep_results if 20 <= r[2] <= 120]
if candidates:
    best = min(candidates, key=lambda r: abs(r[2] - TARGET))
else:
    best = min(sweep_results, key=lambda r: abs(r[2] - TARGET))

best_floor, best_dur = best[0], best[1]
best_val_count = best[2]
print(f"\nBest config selected: std_floor={best_floor}, min_dur={best_dur}  (val events={best_val_count}, target ~{TARGET})")

# --------------------------------------------------------------------------
# 6. Apply ONCE to TEST year 2025 - final reported result
# --------------------------------------------------------------------------
print(f"\n--- Applying tuned params (floor={best_floor}, min_dur={best_dur}) to Test Year 2025 ---")
ev_test, df_test_final = extract_events(df_clean_test, best_floor, best_dur)
_, df_val_final        = extract_events(df_clean_val,  best_floor, best_dur)

# Save corrected datasets
df_test_final.to_pickle(os.path.join(data_dir, "test_detection_results.pkl"))
df_val_final.to_pickle(os.path.join(data_dir, "validation_detection_results.pkl"))

# --------------------------------------------------------------------------
# 7. Save events CSV
# --------------------------------------------------------------------------
if not ev_test.empty:
    ev_out = ev_test.copy()
    ev_out['Start'] = ev_out['Start'].dt.strftime('%Y-%m-%d')
    ev_out['End']   = ev_out['End'].dt.strftime('%Y-%m-%d')
    ev_out = ev_out.rename(columns={
        'Start': 'Start Date', 'End': 'End Date',
        'Peak Z': 'Peak Z-score', 'Med': 'Medium Days',
        'High': 'High Days', 'Crit': 'Critical Days'
    })
    keep = ['Event ID', 'District', 'Disease', 'Start Date', 'End Date', 'Duration',
            'Peak Cases', 'Peak Z-score', 'Highest Risk', 'Medium Days', 'High Days', 'Critical Days']
    ev_out[[c for c in keep if c in ev_out.columns]].to_csv(
        os.path.join(reports_dir, "historical_replay_events_tuned.csv"), index=False)

# --------------------------------------------------------------------------
# 8. Three-way comparison table
# --------------------------------------------------------------------------
n_test  = len(ev_test)
c_med   = int((ev_test['Highest Risk'] == 'Medium').sum())   if n_test else 0
c_high  = int((ev_test['Highest Risk'] == 'High').sum())     if n_test else 0
c_crit  = int((ev_test['Highest Risk'] == 'Critical').sum()) if n_test else 0
avg_dur = round(ev_test['Duration'].mean(), 2) if n_test else 0
max_z   = round(ev_test['Peak Z'].max(), 2)    if n_test else 0

print("\n### Three-Way Comparison")
three_way = [
    ["Total Events",     1192, 4,       n_test],
    ["Critical Events",  339,  4,       c_crit],
    ["High Events",      570,  0,       c_high],
    ["Medium Events",    283,  0,       c_med],
    ["Avg Duration",     1.11, 2.00,    avg_dur],
    ["Max Z-score",      5.29, 3.87,    max_z]
]
print(tabulate(three_way,
    headers=["Metric", "No Floor (Dur=1)", "Over-corr (Std=0.5,Dur=2)", f"Tuned (Std={best_floor},Dur={best_dur},PeakCases>=2)"],
    tablefmt="github"))

if n_test > 0:
    top_10 = ev_test.sort_values('Peak Z', ascending=False).head(10).copy()
    top_10['Rank'] = range(1, len(top_10) + 1)
    print(f"\n### Top Events (Test 2025, Tuned Params)")
    print(tabulate(
        top_10[['Rank','District','Disease','Peak Z','Peak Cases','Duration','Highest Risk']],
        headers='keys', showindex=False, tablefmt="github"))

    dist_sum = ev_test.groupby('District').agg(
        Events=('Event ID','count'),
        Medium=('Highest Risk', lambda x: (x=='Medium').sum()),
        High=('Highest Risk',   lambda x: (x=='High').sum()),
        Critical=('Highest Risk', lambda x: (x=='Critical').sum())
    ).reset_index()
    print("\n### District Summary")
    print(tabulate(dist_sum, headers='keys', showindex=False, tablefmt="github"))

    dis_sum = ev_test.groupby('Disease').agg(
        Events=('Event ID','count'),
        Medium=('Highest Risk', lambda x: (x=='Medium').sum()),
        High=('Highest Risk',   lambda x: (x=='High').sum()),
        Critical=('Highest Risk', lambda x: (x=='Critical').sum())
    ).reset_index()
    print("\n### Disease Summary")
    print(tabulate(dis_sum, headers='keys', showindex=False, tablefmt="github"))

# --------------------------------------------------------------------------
# 9. JSON
# --------------------------------------------------------------------------
summary = {
    "chosen_parameters": {"std_floor": best_floor, "min_event_duration": best_dur},
    "tuning_target": TARGET,
    "tuning_sweep": [
        {"std_floor": r[0], "min_dur": r[1], "validation_events": r[2],
         "medium": r[3], "high": r[4], "critical": r[5]}
        for r in sweep_results
    ],
    "test_results": {
        "total_events": n_test, "medium_events": c_med,
        "high_events": c_high, "critical_events": c_crit,
        "avg_duration": float(avg_dur), "max_z_score": float(max_z)
    }
}
with open(os.path.join(reports_dir, "historical_replay_tuned_summary.json"), "w") as f:
    json.dump(summary, f, indent=4)

print(f"\nTuning complete. Final test events: {n_test}  (floor={best_floor}, min_dur={best_dur})")
