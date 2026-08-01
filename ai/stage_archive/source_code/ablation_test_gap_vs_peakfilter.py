"""
Ablation Test: Gap Fix vs. Peak-Cases Filter
Diagnostic only — no saved Stage 3.x outputs are modified.

Configs:
  A: gap-corrected baseline,  min_dur=2, NO min_peak_cases
  B: gap-corrected baseline,  min_dur=2, min_peak_cases>=2  (Stage 3.3 result)
  C: contaminated baseline,   min_dur=2, NO min_peak_cases  (Stage 3.1.1 exact)
  D: contaminated baseline,   min_dur=2, min_peak_cases>=2  (Stage 3.1.2 exact)

Output: reports/ablation_test_gap_vs_peakfilter.csv
"""

import os
import pandas as pd
import numpy as np
from tabulate import tabulate

base_dir  = r"C:\BRAIN-STORM\HT\warning\outbreak_detection_system"
data_dir  = os.path.join(base_dir, "data", "processed")
reports_dir = os.path.join(base_dir, "reports")

EPSILON      = 1e-6
MIN_DURATION = 2

# --------------------------------------------------------------------------
# 1. Build full chronological dataset (raw timeseries, Stage 1 only)
# --------------------------------------------------------------------------
print("Loading raw Stage 1 timeseries (no Stage 3 contamination)...")
df_train = pd.read_pickle(os.path.join(data_dir, "train_timeseries.pkl"))
df_val   = pd.read_pickle(os.path.join(data_dir, "validation_timeseries.pkl"))
df_test  = pd.read_pickle(os.path.join(data_dir, "test_timeseries.pkl"))

for df in [df_train, df_val, df_test]:
    df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'])

df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
df_all = df_all.sort_values(['district', 'disease_name', 'diagnosis_date']).reset_index(drop=True)

test_min = df_test['diagnosis_date'].min()
test_max = df_test['diagnosis_date'].max()
n_series = df_test.groupby(['district','disease_name']).ngroups
print(f"  Test date range: {test_min.date()} to {test_max.date()}")
print(f"  District-disease series: {n_series}")

# --------------------------------------------------------------------------
# 2. Compute both baseline flavours across the full series
# --------------------------------------------------------------------------
print("\nComputing contaminated baseline (rolling window includes T)...")
grp = df_all.groupby(['district', 'disease_name'])
df_all['cont_mean'] = grp['case_count'].transform(
    lambda x: x.rolling(30, min_periods=1).mean())
df_all['cont_std']  = grp['case_count'].transform(
    lambda x: x.rolling(30, min_periods=1).std().fillna(0))

print("Computing gap-corrected baseline (30-day window ending at T-8)...")
results_gap = []
for (dist, dis), g in df_all.groupby(['district', 'disease_name']):
    g = g.copy().reset_index(drop=True)
    # Baseline window [T-37, T-8]: min_periods=15 for stability
    b_mean = g['case_count'].rolling(30, min_periods=15).mean().shift(8).fillna(0)
    b_std  = g['case_count'].rolling(30, min_periods=15).std().shift(8).fillna(0)
    r_mean = g['case_count'].rolling(7, min_periods=1).mean()

    # std_safe: NaN for zero-std (suppresses Z), epsilon for very-small nonzero
    std_safe = b_std.copy()
    std_safe[std_safe  > 0] = std_safe[std_safe > 0].clip(lower=EPSILON)
    std_safe[std_safe == 0] = np.nan   # forces z=0 via fillna below

    z_raw = (r_mean - b_mean) / std_safe
    g['gap_z'] = z_raw.fillna(0)
    results_gap.append(g)

df_all = pd.concat(results_gap, ignore_index=True)
df_all = df_all.sort_values(['district', 'disease_name', 'diagnosis_date']).reset_index(drop=True)

# Contaminated z-score
cont_std_safe = df_all['cont_std'].replace(0, EPSILON)
df_all['cont_z'] = (df_all['case_count'] - df_all['cont_mean']) / cont_std_safe

# Slice test year
df_test_all = df_all[
    (df_all['diagnosis_date'] >= test_min) & (df_all['diagnosis_date'] <= test_max)
].copy()

# --------------------------------------------------------------------------
# 3. Shared risk classifier
# --------------------------------------------------------------------------
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
# 4. Shared event extractor
# --------------------------------------------------------------------------
def extract_events(df, z_col, min_duration, min_peak_cases):
    """
    z_col: column name for the z-score to use
    min_peak_cases: set to 0 to disable the filter
    """
    df = df.copy()
    df['risk_level'] = classify_risk(df[z_col])

    risk_order   = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
    reverse_risk = {1: 'Medium', 2: 'High', 3: 'Critical'}
    events = []
    ctr    = 1

    for (dist, dis), grp in df.groupby(['district', 'disease_name']):
        grp = grp.sort_values('diagnosis_date')
        in_event = False
        cur      = None

        for _, row in grp.iterrows():
            risk = row['risk_level']
            if risk != 'Low':
                if not in_event:
                    in_event = True
                    cur = {
                        'District': dist, 'Disease': dis,
                        'Start':      row['diagnosis_date'],
                        'End':        row['diagnosis_date'],
                        'Peak Cases': row['case_count'],
                        'Peak Z':     row[z_col],
                        'max_risk':   risk_order[risk],
                        'Med': 0, 'High': 0, 'Crit': 0
                    }
                cur['End']        = row['diagnosis_date']
                cur['Peak Cases'] = max(cur['Peak Cases'], row['case_count'])
                cur['Peak Z']     = max(cur['Peak Z'],     row[z_col])
                cur['max_risk']   = max(cur['max_risk'],   risk_order[risk])
                if risk == 'Medium':     cur['Med']  += 1
                elif risk == 'High':     cur['High'] += 1
                elif risk == 'Critical': cur['Crit'] += 1
            else:
                if in_event:
                    in_event = False
                    dur = (cur['End'] - cur['Start']).days + 1
                    passes = dur >= min_duration and cur['Peak Cases'] >= min_peak_cases
                    if passes:
                        cur['Event ID'] = f"EVT-{ctr:04d}"; ctr += 1
                        cur['Duration'] = dur
                        cur['Highest Risk'] = reverse_risk[cur['max_risk']]
                        events.append(dict(cur))
                    cur = None

        if in_event:
            dur = (cur['End'] - cur['Start']).days + 1
            passes = dur >= min_duration and cur['Peak Cases'] >= min_peak_cases
            if passes:
                cur['Event ID'] = f"EVT-{ctr:04d}"; ctr += 1
                cur['Duration'] = dur
                cur['Highest Risk'] = reverse_risk[cur['max_risk']]
                events.append(dict(cur))

    return pd.DataFrame(events)

def summarise(ev_df, label):
    n = len(ev_df)
    if n == 0:
        return {
            'Config': label, 'Total': 0,
            'Critical': 0, 'High': 0, 'Medium': 0,
            'Avg Dur': 0, 'Max Z': 0,
            'Peak=1 Events': 0, 'Peak=1 %': '0%'
        }
    crit = int((ev_df['Highest Risk'] == 'Critical').sum())
    high = int((ev_df['Highest Risk'] == 'High').sum())
    med  = int((ev_df['Highest Risk'] == 'Medium').sum())
    p1   = int((ev_df['Peak Cases'] == 1).sum())
    return {
        'Config':         label,
        'Total':          n,
        'Critical':       crit,
        'High':           high,
        'Medium':         med,
        'Avg Dur':        round(ev_df['Duration'].mean(), 2),
        'Max Z':          round(ev_df['Peak Z'].max(), 4),
        'Peak=1 Events':  p1,
        'Peak=1 %':       f"{round(p1/n*100, 1)}%"
    }

# --------------------------------------------------------------------------
# 5. Run all four configs
# --------------------------------------------------------------------------
print("\nRunning Config A: gap-corrected, min_dur=2, NO peak filter...")
ev_A = extract_events(df_test_all, 'gap_z', MIN_DURATION, min_peak_cases=1)

print("Running Config B: gap-corrected, min_dur=2, peak>=2...")
ev_B = extract_events(df_test_all, 'gap_z', MIN_DURATION, min_peak_cases=2)

print("Running Config C: contaminated, min_dur=2, NO peak filter (Stage 3.1.1)...")
ev_C = extract_events(df_test_all, 'cont_z', MIN_DURATION, min_peak_cases=1)

print("Running Config D: contaminated, min_dur=2, peak>=2 (Stage 3.1.2)...")
ev_D = extract_events(df_test_all, 'cont_z', MIN_DURATION, min_peak_cases=2)

# --------------------------------------------------------------------------
# 6. Comparison table
# --------------------------------------------------------------------------
rows = [
    summarise(ev_A, "A: Gap+NoPeakFilter"),
    summarise(ev_B, "B: Gap+PeakCases>=2 (Stage 3.3)"),
    summarise(ev_C, "C: Contam+NoPeakFilter (Stage 3.1.1)"),
    summarise(ev_D, "D: Contam+PeakCases>=2 (Stage 3.1.2)")
]

print("\n")
print("=" * 70)
print("ABLATION RESULT: All 4 Configurations")
print("=" * 70)
print(tabulate(rows, headers="keys", tablefmt="github"))

# --------------------------------------------------------------------------
# 7. Direct answer: does gap fix alone eliminate peak_cases=1 artifacts?
# --------------------------------------------------------------------------
print("\n--- KEY FINDING ---")
a_p1 = rows[0]['Peak=1 Events']
c_p1 = rows[2]['Peak=1 Events']
a_total = rows[0]['Total']
c_total = rows[2]['Total']

print(f"Config C (contaminated, no peak filter):   {c_total} events, {c_p1} with peak_cases=1")
print(f"Config A (gap-corrected, no peak filter):  {a_total} events, {a_p1} with peak_cases=1")

if a_p1 == 0:
    print("""
CONCLUSION: The gap fix ALONE completely eliminates peak_cases=1 artifacts.
  min_peak_cases is REDUNDANT once the gap-corrected baseline is applied.
  -> Config A's event count is the correct final result.
  -> Config B over-filters by applying an unnecessary post-hoc patch.
""")
elif a_p1 < c_p1:
    print(f"""
CONCLUSION: The gap fix REDUCES but does NOT eliminate peak_cases=1 artifacts
  ({c_p1} -> {a_p1} events with peak_cases=1 surviving the gap fix).
  Both filters address DIFFERENT failure modes:
    Gap fix:          prevents contaminated baseline from suppressing Z during outbreaks
    min_peak_cases:   directly removes single-case statistical flukes
  -> Both filters are needed; Config B (4 events) is the correct final result.
  -> Flag for presentation: 'sparsity limitation — our dataset's disease series
     are highly zero-inflated, making single-case detections an inherent 
     challenge for Z-score methods.'
""")
else:
    print(f"""
CONCLUSION: The gap fix does NOT reduce peak_cases=1 count ({a_p1} events remain).
  Both filters are independently necessary.
""")

# --------------------------------------------------------------------------
# 8. Show Config A events in detail (key diagnostic output)
# --------------------------------------------------------------------------
if len(ev_A) > 0:
    ev_A_display = ev_A.copy()
    ev_A_display['Start'] = ev_A_display['Start'].dt.strftime('%Y-%m-%d')
    ev_A_display['End']   = ev_A_display['End'].dt.strftime('%Y-%m-%d')
    ev_A_display['Peak Z'] = ev_A_display['Peak Z'].round(3)
    print("\nConfig A Events (gap-corrected, no peak-case filter):")
    print(tabulate(
        ev_A_display[['Event ID','District','Disease','Start','End',
                       'Duration','Peak Cases','Peak Z','Highest Risk']],
        headers=["Event ID","District","Disease","Start","End",
                 "Dur","Peak Cases","Peak Z","Risk"],
        showindex=False, tablefmt="github"))

# --------------------------------------------------------------------------
# 9. Save CSV
# --------------------------------------------------------------------------
out_rows = []
for ev_df, label in [(ev_A,"A"),(ev_B,"B"),(ev_C,"C"),(ev_D,"D")]:
    if len(ev_df) > 0:
        tmp = ev_df.copy()
        tmp['Config'] = label
        tmp['Start'] = tmp['Start'].dt.strftime('%Y-%m-%d')
        tmp['End']   = tmp['End'].dt.strftime('%Y-%m-%d')
        out_rows.append(tmp)

if out_rows:
    out_df = pd.concat(out_rows, ignore_index=True)
    out_df.to_csv(os.path.join(reports_dir, "ablation_test_gap_vs_peakfilter.csv"), index=False)
    print(f"\nSaved all events to: reports/ablation_test_gap_vs_peakfilter.csv")

print("\nAblation test complete (READ ONLY — no Stage 3.x files modified).")
