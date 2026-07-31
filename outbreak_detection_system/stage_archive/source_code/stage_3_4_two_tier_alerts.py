"""
Stage 3.4 - Two-Tier Alerting System

Tier 1 'Confirmed': gap-corrected baseline, min_duration>=2, min_peak_cases>=2
Tier 2 'Watch':     gap-corrected baseline, min_duration>=2, peak_cases=1
                    (events that pass duration/gap but fail peak-case filter)

Tiers are mutually exclusive by construction:
  - Run Config A (gap, no peak filter) -> all candidate events
  - peak_cases >= 2  -> Tier 1 Confirmed
  - peak_cases == 1  -> Tier 2 Watch

Saves: reports/two_tier_alerts_2025.csv
Updates: reports/historical_replay_report.md (new section)
Does NOT modify Stage 3.1.1, 3.1.2, or 3.3 saved outputs.
"""

import os
import json
import pandas as pd
import numpy as np
from tabulate import tabulate

base_dir    = r"C:\BRAIN-STORM\HT\warning\outbreak_detection_system"
data_dir    = os.path.join(base_dir, "data", "processed")
reports_dir = os.path.join(base_dir, "reports")

EPSILON      = 1e-6
MIN_DURATION = 2

# --------------------------------------------------------------------------
# 1. Load raw Stage 1 timeseries
# --------------------------------------------------------------------------
print("Loading raw Stage 1 timeseries (unchanged)...")
df_train = pd.read_pickle(os.path.join(data_dir, "train_timeseries.pkl"))
df_val   = pd.read_pickle(os.path.join(data_dir, "validation_timeseries.pkl"))
df_test  = pd.read_pickle(os.path.join(data_dir, "test_timeseries.pkl"))

for df in [df_train, df_val, df_test]:
    df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'])

df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
df_all = df_all.sort_values(['district', 'disease_name', 'diagnosis_date']).reset_index(drop=True)

test_min = df_test['diagnosis_date'].min()
test_max = df_test['diagnosis_date'].max()
print(f"  Test: {test_min.date()} to {test_max.date()}, "
      f"{df_test.groupby(['district','disease_name']).ngroups} series")

# --------------------------------------------------------------------------
# 2. Gap-corrected baseline (identical to Stage 3.3 logic)
# --------------------------------------------------------------------------
print("Computing gap-corrected baseline (baseline [T-37,T-8], recent [T-6,T])...")
results = []
for (dist, dis), g in df_all.groupby(['district', 'disease_name']):
    g = g.copy().reset_index(drop=True)
    b_mean = g['case_count'].rolling(30, min_periods=15).mean().shift(8).fillna(0)
    b_std  = g['case_count'].rolling(30, min_periods=15).std().shift(8).fillna(0)
    r_mean = g['case_count'].rolling(7, min_periods=1).mean()

    std_safe = b_std.copy()
    std_safe[std_safe  > 0] = std_safe[std_safe > 0].clip(lower=EPSILON)
    std_safe[std_safe == 0] = np.nan

    z_raw = (r_mean - b_mean) / std_safe
    g['gap_z'] = z_raw.fillna(0)
    results.append(g)

df_all = pd.concat(results, ignore_index=True)
df_all = df_all.sort_values(['district', 'disease_name', 'diagnosis_date']).reset_index(drop=True)

# --------------------------------------------------------------------------
# 3. Risk classification (same thresholds throughout)
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

df_test_gap = df_all[
    (df_all['diagnosis_date'] >= test_min) & (df_all['diagnosis_date'] <= test_max)
].copy()
df_test_gap['risk_level'] = classify_risk(df_test_gap['gap_z'])

# --------------------------------------------------------------------------
# 4. Extract ALL candidate events (min_duration>=2, no peak filter yet)
# --------------------------------------------------------------------------
risk_order   = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
reverse_risk = {1: 'Medium', 2: 'High', 3: 'Critical'}

all_events = []
ctr = 1

for (dist, dis), grp in df_test_gap.groupby(['district', 'disease_name']):
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
                    'Peak Z':     row['gap_z'],
                    'max_risk':   risk_order[risk],
                    'Med': 0, 'High': 0, 'Crit': 0
                }
            cur['End']        = row['diagnosis_date']
            cur['Peak Cases'] = max(cur['Peak Cases'], row['case_count'])
            cur['Peak Z']     = max(cur['Peak Z'],     row['gap_z'])
            cur['max_risk']   = max(cur['max_risk'],   risk_order[risk])
            if risk == 'Medium':     cur['Med']  += 1
            elif risk == 'High':     cur['High'] += 1
            elif risk == 'Critical': cur['Crit'] += 1
        else:
            if in_event:
                in_event = False
                dur = (cur['End'] - cur['Start']).days + 1
                if dur >= MIN_DURATION:
                    cur['Event ID'] = f"EVT-{ctr:04d}"; ctr += 1
                    cur['Duration'] = dur
                    cur['Highest Risk'] = reverse_risk.get(cur['max_risk'], 'Medium')
                    all_events.append(dict(cur))
                cur = None

    if in_event:
        dur = (cur['End'] - cur['Start']).days + 1
        if dur >= MIN_DURATION:
            cur['Event ID'] = f"EVT-{ctr:04d}"; ctr += 1
            cur['Duration'] = dur
            cur['Highest Risk'] = reverse_risk.get(cur['max_risk'], 'Medium')
            all_events.append(dict(cur))

ev_df = pd.DataFrame(all_events)
print(f"  Total candidate events (min_dur>=2, no peak filter): {len(ev_df)}")

# --------------------------------------------------------------------------
# 5. Assign tiers — mutually exclusive by peak_cases
# --------------------------------------------------------------------------
ev_df['Tier']  = ev_df['Peak Cases'].apply(lambda x: 'Confirmed' if x >= 2 else 'Watch')
# Watch events: label as Watch, no severity classification needed
ev_df['Alert Label'] = ev_df.apply(
    lambda r: r['Highest Risk'] if r['Tier'] == 'Confirmed' else 'Watch', axis=1
)

# --------------------------------------------------------------------------
# 6. Console summary
# --------------------------------------------------------------------------
confirmed = ev_df[ev_df['Tier'] == 'Confirmed']
watch     = ev_df[ev_df['Tier'] == 'Watch']

print(f"\n  Tier 1 Confirmed: {len(confirmed)} events")
print(f"  Tier 2 Watch:     {len(watch)} events")
print(f"  Total:            {len(ev_df)} events")
print(f"  Tiers mutually exclusive: {set(confirmed['Event ID']) & set(watch['Event ID']) == set()}")

# ---- Full event table ----
ev_display = ev_df.copy()
ev_display['Start'] = ev_display['Start'].dt.strftime('%Y-%m-%d')
ev_display['End']   = ev_display['End'].dt.strftime('%Y-%m-%d')
ev_display['Peak Z'] = ev_display['Peak Z'].round(3)

print("\n### Combined Two-Tier Alert Table")
print(tabulate(
    ev_display[['Event ID', 'Tier', 'District', 'Disease', 'Start', 'End',
                'Duration', 'Peak Cases', 'Peak Z', 'Alert Label']],
    headers=["Event ID", "Tier", "District", "Disease", "Start", "End",
             "Dur", "Peak Cases", "Peak Z", "Alert"],
    showindex=False, tablefmt="github"
))

# ---- Tier summary counts ----
print("\n### Tier Summary")
tier_summary = [
    ["Confirmed (Tier 1)", len(confirmed),
     int((confirmed['Alert Label']=='Critical').sum()),
     int((confirmed['Alert Label']=='High').sum()),
     int((confirmed['Alert Label']=='Medium').sum()),
     round(confirmed['Duration'].mean(), 2) if len(confirmed) else 0,
     round(confirmed['Peak Z'].max(), 3) if len(confirmed) else 0],
    ["Watch (Tier 2)", len(watch), 0, 0, 0,
     round(watch['Duration'].mean(), 2) if len(watch) else 0,
     round(watch['Peak Z'].max(), 3) if len(watch) else 0],
]
print(tabulate(tier_summary,
    headers=["Tier", "Events", "Critical", "High", "Medium", "Avg Dur", "Max Z"],
    tablefmt="github"))

# ---- District breakdown ----
print("\n### District Breakdown")
dist_tbl = ev_df.groupby(['District', 'Tier']).agg(
    Events=('Event ID', 'count')
).unstack(fill_value=0).reset_index()
dist_tbl.columns = ['District', 'Confirmed', 'Watch']
dist_tbl['Total'] = dist_tbl['Confirmed'] + dist_tbl['Watch']
print(tabulate(dist_tbl, headers='keys', showindex=False, tablefmt="github"))

# ---- Disease breakdown ----
print("\n### Disease Breakdown")
dis_tbl = ev_df.groupby(['Disease', 'Tier']).agg(
    Events=('Event ID', 'count')
).unstack(fill_value=0).reset_index()
dis_tbl.columns = ['Disease', 'Confirmed', 'Watch']
dis_tbl['Total'] = dis_tbl['Confirmed'] + dis_tbl['Watch']
print(tabulate(dis_tbl, headers='keys', showindex=False, tablefmt="github"))

# --------------------------------------------------------------------------
# 7. Save CSV
# --------------------------------------------------------------------------
out_df = ev_display[['Event ID', 'Tier', 'Alert Label', 'District', 'Disease',
                      'Start', 'End', 'Duration', 'Peak Cases', 'Peak Z',
                      'Med', 'High', 'Crit']].copy()
out_df = out_df.rename(columns={'Med': 'Medium Days', 'High': 'High Days', 'Crit': 'Critical Days'})
out_df.to_csv(os.path.join(reports_dir, "two_tier_alerts_2025.csv"), index=False)
print(f"\nSaved: reports/two_tier_alerts_2025.csv")

# --------------------------------------------------------------------------
# 8. Update historical_replay_report.md
# --------------------------------------------------------------------------
md_path = os.path.join(reports_dir, "historical_replay_report.md")
with open(md_path, "a", encoding="utf-8") as f:
    f.write("\n\n---\n\n")
    f.write("## 13. Two-Tier Alerting System (Stage 3.4)\n\n")
    f.write("### System Description\n\n")
    f.write("> The system uses a two-tier alerting model, mirroring real-world surveillance "
            "practice: a sensitive **Watch** tier for early signals worth monitoring, and a "
            "high-precision **Confirmed** tier reserved for sustained, statistically robust "
            "anomalies (peak cases \u22652, duration \u22652 days, corrected against a "
            "gap-separated historical baseline).\n\n")
    f.write("### Tier Definitions\n\n")
    f.write("| Tier | Label | min_duration | min_peak_cases | Baseline | Purpose |\n")
    f.write("|------|-------|-------------|----------------|----------|---------|\n")
    f.write("| Tier 1 | **Confirmed** | \u22652 days | \u22652 cases | Gap-corrected [T-37, T-8] | "
            "High-precision confirmed outbreak alert |\n")
    f.write("| Tier 2 | **Watch** | \u22652 days | \u22651 case (any) | Gap-corrected [T-37, T-8] | "
            "Sensitive early signal for monitoring |\n\n")
    f.write("*Tiers are mutually exclusive: events qualifying for Tier 1 are excluded from Tier 2.*\n\n")
    f.write("### Results on 2025 Test Set\n\n")
    f.write(tabulate(tier_summary,
        headers=["Tier", "Events", "Critical", "High", "Medium", "Avg Dur", "Max Z"],
        tablefmt="github"))
    f.write("\n\n### District Breakdown\n\n")
    f.write(tabulate(dist_tbl, headers='keys', showindex=False, tablefmt="github"))
    f.write("\n\n### Disease Breakdown\n\n")
    f.write(tabulate(dis_tbl, headers='keys', showindex=False, tablefmt="github"))
    f.write("\n\n### Design Rationale\n\n")
    f.write("The two-tier model was motivated by the ablation test (Stage 3.4 diagnostic), which "
            "showed that the gap-corrected baseline alone reduces peak_cases=1 artifacts from "
            "101 (contaminated baseline) to 18 — a substantial improvement, but not complete "
            "elimination. Rather than discarding these 18 signals entirely, they are preserved "
            "as low-confidence Watch-tier alerts, allowing health authorities to apply domain "
            "judgment on whether to investigate further.\n")

print("Updated: reports/historical_replay_report.md")
print("\nStage 3.4 Two-Tier Alerting complete.")
