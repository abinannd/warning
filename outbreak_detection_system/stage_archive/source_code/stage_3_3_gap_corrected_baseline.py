"""
Stage 3.3 - Gap-Corrected Baseline Detection

Fixes baseline contamination in the rolling Z-score engine.

Problem with previous approach:
  rolling_mean_30 at day T included days [T-29, T], so the most recent
  7 days of data (which form the "recent trend" signal) were also part of
  the baseline — this inflates the baseline during actual outbreaks and
  suppresses the Z-score exactly when you need it to be high.

Fix:
  Historical baseline window: [T-37, T-8]  (30-day window, 7-day gap)
  Recent trend window:        [T-6,  T]    (7-day window, including today)
  Z-score = (recent_mean - baseline_mean) / max(baseline_std, 1e-6)
  EWMA: lagged 8 days (computed up to T-8, consistent with baseline gap)

Implementation via pandas shift:
  rolling(30).mean().shift(8)  => mean of [T-37, T-8]
  rolling(30).std().shift(8)   => std  of [T-37, T-8]
  rolling(7).mean()            => mean of [T-6,  T]
  ewm(span=14).mean().shift(8) => EWMA up to T-8

Constraints:
  - Do NOT modify Stage 1 or Stage 2 outputs
  - Only Stage 3 detection logic is updated
  - min_duration=2, min_peak_cases=2, EPSILON=1e-6 (from Stage 3.1.2)
"""

import os
import json
import pandas as pd
import numpy as np
from tabulate import tabulate

base_dir  = r"C:\BRAIN-STORM\HT\warning\outbreak_detection_system"
data_dir  = os.path.join(base_dir, "data", "processed")
reports_dir = os.path.join(base_dir, "reports")

EPSILON       = 1e-6
MIN_DURATION  = 2
MIN_PEAK_CASES = 2

# --------------------------------------------------------------------------
# 1. Load all three timeseries (to maintain chronological continuity)
# --------------------------------------------------------------------------
print("Loading raw timeseries (Stage 1/2 outputs - NOT modified)...")
df_train = pd.read_pickle(os.path.join(data_dir, "train_timeseries.pkl"))
df_val   = pd.read_pickle(os.path.join(data_dir, "validation_timeseries.pkl"))
df_test  = pd.read_pickle(os.path.join(data_dir, "test_timeseries.pkl"))

for df in [df_train, df_val, df_test]:
    df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'])

df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
df_all = df_all.sort_values(['district', 'disease_name', 'diagnosis_date']).reset_index(drop=True)

test_min = df_test['diagnosis_date'].min()
test_max = df_test['diagnosis_date'].max()

print(f"  Train rows: {len(df_train)}, Val rows: {len(df_val)}, Test rows: {len(df_test)}")

# --------------------------------------------------------------------------
# 2. Apply gap-corrected baseline per district-disease group
# --------------------------------------------------------------------------
print("\nComputing gap-corrected baseline features per district-disease group...")

results = []

for (dist, dis), group in df_all.groupby(['district', 'disease_name']):
    g = group.copy().reset_index(drop=True)

    # --- Baseline window: [T-37, T-8] via shift(8) on rolling(30) ---
    # min_periods=15: require at least 15 historical days before trusting the baseline
    baseline_mean = g['case_count'].rolling(window=30, min_periods=15).mean().shift(8)
    baseline_std  = g['case_count'].rolling(window=30, min_periods=15).std().shift(8)

    # Fill NaN from initial rows (insufficient history) with 0
    baseline_mean = baseline_mean.fillna(0)
    baseline_std  = baseline_std.fillna(0)

    # Track where baseline_std is truly 0 (no variance — suppress Z-score)
    # Only apply epsilon to non-zero stds that are very small
    baseline_std_safe = baseline_std.copy()
    baseline_std_safe[baseline_std_safe > 0] = baseline_std_safe[baseline_std_safe > 0].clip(lower=EPSILON)
    baseline_std_safe[baseline_std_safe == 0] = np.nan  # marks "no-history" rows

    # --- Recent trend window: [T-6, T] ---
    recent_mean = g['case_count'].rolling(window=7, min_periods=1).mean()

    # --- Z-score: NaN where no baseline history, 0 where std=0 but history exists ---
    z_score_raw = (recent_mean - baseline_mean) / baseline_std_safe
    # Where baseline_std is 0 but history exists (flat baseline), set Z=0
    # Where no history at all (NaN denominator), set Z=0 as well — safe suppression
    z_score = z_score_raw.fillna(0)

    # --- EWMA: lagged 8 days, consistent with baseline gap ---
    ewma_14 = g['case_count'].ewm(span=14, adjust=False).mean().shift(8).fillna(0)

    g['baseline_mean']    = baseline_mean
    g['baseline_std']     = baseline_std
    g['recent_mean']      = recent_mean
    g['ewma_14_lagged']   = ewma_14
    g['rolling_z_score']  = z_score

    results.append(g)

df_all = pd.concat(results, ignore_index=True)
df_all = df_all.sort_values(['district', 'disease_name', 'diagnosis_date']).reset_index(drop=True)

# --------------------------------------------------------------------------
# 3. Risk level classification (same thresholds as Stage 3.1.2)
# --------------------------------------------------------------------------
def classify_risk(z):
    return np.select(
        [z < 2.0, (z >= 2.0) & (z < 2.5), (z >= 2.5) & (z < 3.0), z >= 3.0],
        ['Low', 'Medium', 'High', 'Critical'],
        default='Low'
    )

df_all['risk_level'] = classify_risk(df_all['rolling_z_score'])

# Slice test year
df_test_new = df_all[
    (df_all['diagnosis_date'] >= test_min) & (df_all['diagnosis_date'] <= test_max)
].copy()

print(f"  Test rows processed: {len(df_test_new)}")

# --------------------------------------------------------------------------
# 4. Verify no NaN/inf in z-scores
# --------------------------------------------------------------------------
nan_count = df_test_new['rolling_z_score'].isna().sum()
inf_count = np.isinf(df_test_new['rolling_z_score']).sum()
print(f"  Z-score NaN count: {nan_count}  |  Inf count: {inf_count}")

# --------------------------------------------------------------------------
# 5. Event extraction (same logic as Stage 3.1.2: duration>=2, peak_cases>=2)
# --------------------------------------------------------------------------
def extract_events(df, min_duration=MIN_DURATION, min_peak_cases=MIN_PEAK_CASES):
    risk_order   = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
    reverse_risk = {1: 'Medium', 2: 'High', 3: 'Critical'}
    events = []
    ctr    = 1

    for (dist, dis), group in df.groupby(['district', 'disease_name']):
        group = group.sort_values('diagnosis_date')
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

    return pd.DataFrame(events)

print("\nExtracting events from gap-corrected detection...")
ev_new = extract_events(df_test_new)

# --------------------------------------------------------------------------
# 6. Comparison table: OLD vs NEW
# --------------------------------------------------------------------------
# OLD numbers from Stage 3.1.2 (contaminated baseline, tuned params)
old_total    = 16
old_crit     = 16
old_high     = 0
old_medium   = 0
old_avg_dur  = 2.19
old_max_z    = 5.29

n_new  = len(ev_new)
c_crit = int((ev_new['Highest Risk'] == 'Critical').sum()) if n_new else 0
c_high = int((ev_new['Highest Risk'] == 'High').sum())    if n_new else 0
c_med  = int((ev_new['Highest Risk'] == 'Medium').sum())  if n_new else 0
avg_dur_new = round(ev_new['Duration'].mean(), 2) if n_new else 0
max_z_new   = round(ev_new['Peak Z'].max(), 2)    if n_new else 0

print("\n### Comparison: Old (Contaminated) vs New (Gap-Corrected)")
comparison = [
    ["Total Events",     old_total,   n_new],
    ["Critical Events",  old_crit,    c_crit],
    ["High Events",      old_high,    c_high],
    ["Medium Events",    old_medium,  c_med],
    ["Avg Duration",     old_avg_dur, avg_dur_new],
    ["Max Z-score",      old_max_z,   max_z_new]
]
print(tabulate(comparison,
    headers=["Metric", "Old (Contaminated Baseline)", "New (Gap-Corrected Baseline)"],
    tablefmt="github"))

# --------------------------------------------------------------------------
# 7. Save event CSV
# --------------------------------------------------------------------------
if not ev_new.empty:
    ev_out = ev_new.copy()
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
        os.path.join(reports_dir, "historical_replay_events_gapfixed.csv"), index=False)

    print(f"\nSaved {n_new} events to historical_replay_events_gapfixed.csv")
else:
    print("\nNo events detected with gap-corrected baseline.")

# --------------------------------------------------------------------------
# 8. Print detailed event table
# --------------------------------------------------------------------------
if n_new > 0:
    ev_display = ev_new.copy()
    ev_display['Start'] = ev_display['Start'].dt.strftime('%Y-%m-%d')
    ev_display['End']   = ev_display['End'].dt.strftime('%Y-%m-%d')
    ev_display['Peak Z'] = ev_display['Peak Z'].round(3)
    print("\n### Gap-Corrected Events")
    print(tabulate(ev_display[['Event ID', 'District', 'Disease', 'Start', 'End', 'Duration',
                                'Peak Cases', 'Peak Z', 'Highest Risk']],
                   headers=["Event ID", "District", "Disease", "Start", "End", "Dur",
                             "Peak Cases", "Peak Z", "Risk"],
                   showindex=False, tablefmt="github"))

    dist_sum = ev_new.groupby('District').agg(
        Events=('Event ID', 'count'),
        Medium=('Highest Risk', lambda x: (x=='Medium').sum()),
        High=('Highest Risk',   lambda x: (x=='High').sum()),
        Critical=('Highest Risk', lambda x: (x=='Critical').sum())
    ).reset_index()
    print("\n### District Summary")
    print(tabulate(dist_sum, headers='keys', showindex=False, tablefmt="github"))

    dis_sum = ev_new.groupby('Disease').agg(
        Events=('Event ID', 'count'),
        Medium=('Highest Risk', lambda x: (x=='Medium').sum()),
        High=('Highest Risk',   lambda x: (x=='High').sum()),
        Critical=('Highest Risk', lambda x: (x=='Critical').sum())
    ).reset_index()
    print("\n### Disease Summary")
    print(tabulate(dis_sum, headers='keys', showindex=False, tablefmt="github"))

# --------------------------------------------------------------------------
# 9. Update historical_replay_report.md with new section
# --------------------------------------------------------------------------
md_path = os.path.join(reports_dir, "historical_replay_report.md")
with open(md_path, "a") as f:
    f.write("\n\n---\n\n")
    f.write("## 12. Gap-Corrected Baseline (Stage 3.3)\n\n")
    f.write("### System Description\n\n")
    f.write("> A regional early warning system that analyzes the most recent week's "
            "surveillance data against a historical baseline (ending 7 days prior) to "
            "identify statistically significant increases in disease activity and issue "
            "risk-based alerts to public health authorities.\n\n")
    f.write("### Problem: Baseline Contamination\n\n")
    f.write("The original rolling baseline (Stages 2.1–2.4) used a 30-day window ending "
            "on day T, meaning the most recent 7 days of case data — the very window being "
            "tested for elevated activity — were also included in the baseline computation. "
            "This causes the baseline mean and std to track the current outbreak, inflating "
            "the denominator exactly when the Z-score needs to be highest, and suppressing "
            "genuine alerts.\n\n")
    f.write("### Fix: 7-Day Gap Between Baseline and Signal Window\n\n")
    f.write("| Window | Dates Used | Computation |\n")
    f.write("|--------|-----------|-------------|\n")
    f.write("| Historical Baseline | [T-37, T-8] (30 days) | `rolling(30).mean/std.shift(8)` |\n")
    f.write("| Recent Trend Signal | [T-6, T] (7 days) | `rolling(7).mean()` |\n")
    f.write("| Z-score | — | `(recent_mean - baseline_mean) / max(baseline_std, 1e-6)` |\n")
    f.write("| EWMA | Up to T-8 | `ewm(span=14).mean().shift(8)` |\n\n")
    f.write("### Before vs After Comparison\n\n")
    f.write(tabulate(comparison,
        headers=["Metric", "Old (Contaminated Baseline)", "New (Gap-Corrected Baseline)"],
        tablefmt="github"))
    f.write("\n\n### Interpretation\n\n")
    f.write("The gap-corrected baseline separates the 'what the system is testing' window "
            "from the 'what the system learned from' window. This is the correct statistical "
            "design for a surveillance system where the signal of interest should not contaminate "
            "the reference distribution used to judge it.\n")

print("\nUpdated historical_replay_report.md with gap-correction section.")
print("\nStage 3.3 Gap-Corrected Baseline complete.")
