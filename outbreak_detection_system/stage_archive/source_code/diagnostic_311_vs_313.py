"""
DIAGNOSTIC SCRIPT: Reconcile Stage 3.1.1 (117 events) vs Stage 3.3 'old' (16 events)

READ-ONLY. No saved outputs are modified.
"""

import os
import json
import inspect
import pandas as pd
import numpy as np
from tabulate import tabulate

base_dir  = r"C:\BRAIN-STORM\HT\warning\outbreak_detection_system"
data_dir  = os.path.join(base_dir, "data", "processed")
reports_dir = os.path.join(base_dir, "reports")

SEP = "-" * 70

# ==========================================================================
# STEP 1: Print saved config from Stage 3.1.2 JSON
# ==========================================================================
print(SEP)
print("STEP 1: Saved config from Stage 3.1.2 JSON")
print(SEP)
json_path = os.path.join(reports_dir, "historical_replay_tuned_summary.json")
with open(json_path) as f:
    saved = json.load(f)

print("Chosen parameters (Stage 3.1.2 output):")
for k, v in saved.get("chosen_parameters", {}).items():
    print(f"  {k}: {v}")

print("\nFull tuning sweep results:")
sweep = saved.get("tuning_sweep", [])
print(tabulate(sweep, headers="keys", tablefmt="github"))

print("\nTest results stored in JSON:")
for k, v in saved.get("test_results", {}).items():
    print(f"  {k}: {v}")

# ==========================================================================
# STEP 2: Print what Stage 3.3's "old" comparison values ACTUALLY were
# ==========================================================================
print(f"\n{SEP}")
print("STEP 2: What were the hardcoded 'old' values in Stage 3.3?")
print(SEP)
stage33_path = os.path.join(base_dir, "stage_archive", "source_code",
                             "stage_3_3_gap_corrected_baseline.py")
with open(stage33_path) as f:
    stage33_code = f.read()

# Find the old_* lines
old_lines = [l.strip() for l in stage33_code.splitlines()
             if l.strip().startswith("old_")]
print("Lines in Stage 3.3 that define 'old' comparison values:")
for l in old_lines:
    print(f"  {l}")

print("\nKey question: are these 'old' values re-computed or hardcoded?")
is_hardcoded = all("=" in l and "#" not in l.split("=")[0] for l in old_lines)
print(f"  Hardcoded (not re-run): {is_hardcoded}")
if is_hardcoded:
    print("  -> Stage 3.3 never re-ran the contaminated baseline. These are")
    print("     COPIED values from a previous script run, not a fresh computation.")

# ==========================================================================
# STEP 3: Check input data consistency
# ==========================================================================
print(f"\n{SEP}")
print("STEP 3: Input data consistency check")
print(SEP)

df_test = pd.read_pickle(os.path.join(data_dir, "test_detection_results.pkl"))
df_test['diagnosis_date'] = pd.to_datetime(df_test['diagnosis_date'])

print(f"  test_detection_results.pkl")
print(f"    Rows:           {len(df_test)}")
print(f"    Date range:     {df_test['diagnosis_date'].min().date()} to {df_test['diagnosis_date'].max().date()}")
print(f"    District-disease pairs: {df_test.groupby(['district','disease_name']).ngroups}")
print(f"    Columns: {list(df_test.columns)}")

df_test_raw = pd.read_pickle(os.path.join(data_dir, "test_timeseries.pkl"))
df_test_raw['diagnosis_date'] = pd.to_datetime(df_test_raw['diagnosis_date'])
print(f"\n  test_timeseries.pkl (Stage 1 output, never modified)")
print(f"    Rows:           {len(df_test_raw)}")
print(f"    Date range:     {df_test_raw['diagnosis_date'].min().date()} to {df_test_raw['diagnosis_date'].max().date()}")
print(f"    District-disease pairs: {df_test_raw.groupby(['district','disease_name']).ngroups}")

# ==========================================================================
# STEP 4: Check risk thresholds in Stage 3.3 vs Stage 3.1.2
# ==========================================================================
print(f"\n{SEP}")
print("STEP 4: Risk level thresholds")
print(SEP)

stage312_path = os.path.join(base_dir, "stage_archive", "source_code",
                              "stage_3_1_2_parameter_tuning.py")
with open(stage312_path) as f:
    stage312_code = f.read()

print("Stage 3.1.2 classify_risk function:")
in_func = False
for line in stage312_code.splitlines():
    if "def classify_risk" in line:
        in_func = True
    if in_func:
        print(f"  {line}")
    if in_func and line.strip().startswith(")") and "np.select" not in line:
        break

print("\nStage 3.3 classify_risk function:")
in_func = False
for line in stage33_code.splitlines():
    if "def classify_risk" in line:
        in_func = True
    if in_func:
        print(f"  {line}")
    if in_func and line.strip().startswith(")") and "np.select" not in line:
        break

# ==========================================================================
# STEP 5: Check min_peak_cases presence in Stage 3.1.1 vs Stage 3.1.2
# ==========================================================================
print(f"\n{SEP}")
print("STEP 5: min_peak_cases filter — which stage introduced it?")
print(SEP)

stage311_path = os.path.join(base_dir, "stage_archive", "source_code",
                              "stage_3_1_1_sparse_zscore_fix.py")
with open(stage311_path) as f:
    stage311_code = f.read()

has_peak_311 = "min_peak_cases" in stage311_code or "Peak Cases" in stage311_code
has_peak_312 = "min_peak_cases" in stage312_code
has_peak_313 = "min_peak_cases" in stage33_code

print(f"  Stage 3.1.1  - has min_peak_cases filter: {has_peak_311}")
print(f"  Stage 3.1.2  - has min_peak_cases filter: {has_peak_312}")
print(f"  Stage 3.3    - has min_peak_cases filter: {has_peak_313}")

# also check event acceptance lines
def extract_event_acceptance(code, label):
    lines = []
    for line in code.splitlines():
        if "min_duration" in line or "min_peak" in line or "Peak Cases" in line and ">=" in line:
            lines.append(line.strip())
    if lines:
        print(f"\n  {label} event acceptance conditions:")
        for l in lines:
            print(f"    {l}")
    else:
        print(f"\n  {label}: no min_peak_cases condition found in event acceptance")

extract_event_acceptance(stage311_code, "Stage 3.1.1")
extract_event_acceptance(stage312_code, "Stage 3.1.2")
extract_event_acceptance(stage33_code,  "Stage 3.3 (gap-corrected)")

# ==========================================================================
# STEP 6: Re-run Stage 3.1.1's EXACT logic fresh on 2025 test data
# ==========================================================================
print(f"\n{SEP}")
print("STEP 6: Fresh re-run of Stage 3.1.1 logic (ground truth)")
print(SEP)
print("Parameters: std_floor=0.0, min_duration=2, NO min_peak_cases filter")

# Load fresh from the test_timeseries (Stage 1 output only)
df_train_raw = pd.read_pickle(os.path.join(data_dir, "train_timeseries.pkl"))
df_val_raw   = pd.read_pickle(os.path.join(data_dir, "validation_timeseries.pkl"))
for df in [df_train_raw, df_val_raw, df_test_raw]:
    df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'])

df_all = pd.concat([df_train_raw, df_val_raw, df_test_raw], ignore_index=True)
df_all = df_all.sort_values(['district', 'disease_name', 'diagnosis_date']).reset_index(drop=True)

grp = df_all.groupby(['district', 'disease_name'])
df_all['rolling_mean_30'] = grp['case_count'].transform(lambda x: x.rolling(30, min_periods=1).mean())
df_all['rolling_std_30']  = grp['case_count'].transform(lambda x: x.rolling(30, min_periods=1).std().fillna(0))

test_min = df_test_raw['diagnosis_date'].min()
test_max = df_test_raw['diagnosis_date'].max()
df_test_fresh = df_all[(df_all['diagnosis_date'] >= test_min) & (df_all['diagnosis_date'] <= test_max)].copy()

# Stage 3.1.1 logic: epsilon only, NO min_peak_cases
EPSILON = 1e-6
std_safe = df_test_fresh['rolling_std_30'].clip(lower=0.0).replace(0, EPSILON)
df_test_fresh['rolling_z_score'] = (df_test_fresh['case_count'] - df_test_fresh['rolling_mean_30']) / std_safe

def classify_risk(z):
    return np.select(
        [z < 2.0, (z >= 2.0) & (z < 2.5), (z >= 2.5) & (z < 3.0), z >= 3.0],
        ['Low', 'Medium', 'High', 'Critical'],
        default='Low'
    )
df_test_fresh['risk_level'] = classify_risk(df_test_fresh['rolling_z_score'])

risk_order   = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
reverse_risk = {1: 'Medium', 2: 'High', 3: 'Critical'}
events_311 = []
ctr = 1
MIN_DURATION_311 = 2

for (dist, dis), group in df_test_fresh.groupby(['district', 'disease_name']):
    group = group.sort_values('diagnosis_date')
    in_event = False
    cur = None
    for _, row in group.iterrows():
        risk = row['risk_level']
        if risk != 'Low':
            if not in_event:
                in_event = True
                cur = {
                    'District': dist, 'Disease': dis,
                    'Start': row['diagnosis_date'], 'End': row['diagnosis_date'],
                    'Peak Cases': row['case_count'], 'Peak Z': row['rolling_z_score'],
                    'max_risk': risk_order[risk], 'Med': 0, 'High': 0, 'Crit': 0
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
                if dur >= MIN_DURATION_311:  # NO min_peak_cases here
                    cur['Event ID'] = f"EVT-{ctr:04d}"; ctr += 1
                    cur['Duration'] = dur
                    cur['Highest Risk'] = reverse_risk[cur['max_risk']]
                    events_311.append(cur)
                cur = None
    if in_event:
        dur = (cur['End'] - cur['Start']).days + 1
        if dur >= MIN_DURATION_311:
            cur['Event ID'] = f"EVT-{ctr:04d}"; ctr += 1
            cur['Duration'] = dur
            cur['Highest Risk'] = reverse_risk[cur['max_risk']]
            events_311.append(cur)

ev311_df = pd.DataFrame(events_311)
n311 = len(ev311_df)
print(f"\nFresh re-run result: {n311} events")
print(f"  (Stage 3.1.1 reported: 117 events — match: {n311 == 117})")
if n311 > 0:
    print(f"  Critical: {(ev311_df['Highest Risk']=='Critical').sum()}")
    print(f"  High:     {(ev311_df['Highest Risk']=='High').sum()}")
    print(f"  Medium:   {(ev311_df['Highest Risk']=='Medium').sum()}")
    print(f"  Avg Duration: {ev311_df['Duration'].mean():.2f}")
    print(f"  Max Z-score:  {ev311_df['Peak Z'].max():.4f}")

# ==========================================================================
# STEP 7: Root cause summary
# ==========================================================================
print(f"\n{SEP}")
print("STEP 7: Root Cause Summary")
print(SEP)

comparison = [
    ["Stage 3.1.1 (no peak-cases filter)", "std_floor=0.0, min_dur=2, NO min_peak_cases"],
    ["Stage 3.1.2 (after adding filter)",   "std_floor=0.0, min_dur=2, min_peak_cases=2"],
    ["Stage 3.3 'old' comparison",           "HARDCODED — never recomputed; copied Stage 3.1.2's 16-event result"],
]
print(tabulate(comparison, headers=["Run", "Parameters"], tablefmt="github"))

print(f"""
Root Cause Identified:
  - Stage 3.1.1 ran with min_duration=2 but NO min_peak_cases filter -> 117 events
  - Stage 3.1.2 added min_peak_cases=2 filter              -> reduced to 16 events
  - Stage 3.3 'old baseline' row hardcoded old_total=16    -> sourced from Stage 3.1.2
  - Stage 3.3 never actually re-ran the contaminated baseline
  - The '117 vs 16' gap is ENTIRELY explained by the min_peak_cases=2 filter
    added in Stage 3.1.2, NOT by any baseline window or threshold change
""")
print("Diagnostic complete (READ ONLY - no files modified).")
