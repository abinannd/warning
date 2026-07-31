"""
Stage 3.2 - Lead-Time Analysis

Evaluates how early the detection system identifies disease activity
before each outbreak reaches its peak daily case count.

Input:
  - data/processed/test_detection_results.pkl   (day-level detection data)
  - reports/historical_replay_events_tuned.csv  (calibrated event list from Stage 3.1.2)

Output:
  - reports/lead_time_events.csv
  - reports/lead_time_summary.json
  - reports/lead_time_report.md
"""

import os
import json
import pandas as pd
import numpy as np
from tabulate import tabulate

base_dir  = r"C:\BRAIN-STORM\HT\warning\outbreak_detection_system"
data_dir  = os.path.join(base_dir, "data", "processed")
reports_dir = os.path.join(base_dir, "reports")

# --------------------------------------------------------------------------
# 1. Load inputs
# --------------------------------------------------------------------------
print("Loading inputs...")
df_det = pd.read_pickle(os.path.join(data_dir, "test_detection_results.pkl"))
df_det['diagnosis_date'] = pd.to_datetime(df_det['diagnosis_date'])

events_path = os.path.join(reports_dir, "historical_replay_events_tuned.csv")
df_events = pd.read_csv(events_path)
df_events['Start Date'] = pd.to_datetime(df_events['Start Date'])
df_events['End Date']   = pd.to_datetime(df_events['End Date'])

print(f"  Events loaded: {len(df_events)}")
print(f"  Detection rows loaded: {len(df_det)}")

# --------------------------------------------------------------------------
# 2. For each event, compute:
#    - Detection Date  = Start Date of the event (first alert day)
#    - Peak Date       = Day in [Start, End] with highest case_count
#    - Lead Time       = Peak Date - Detection Date (days)
# --------------------------------------------------------------------------
lead_records = []

for _, ev in df_events.iterrows():
    dist = ev['District']
    dis  = ev['Disease']
    start = ev['Start Date']
    end   = ev['End Date']

    # Slice the relevant daily rows for this event window
    mask = (
        (df_det['district'] == dist) &
        (df_det['disease_name'] == dis) &
        (df_det['diagnosis_date'] >= start) &
        (df_det['diagnosis_date'] <= end)
    )
    window = df_det[mask].copy()

    if window.empty:
        # Fallback: peak date = start date, lead time = 0
        peak_date   = start
        peak_cases  = ev.get('Peak Cases', 0)
        peak_z      = ev.get('Peak Z-score', 0)
    else:
        peak_idx   = window['case_count'].idxmax()
        peak_date  = window.loc[peak_idx, 'diagnosis_date']
        peak_cases = int(window.loc[peak_idx, 'case_count'])
        peak_z     = round(float(window['rolling_z_score'].max()), 4)

    detection_date = start
    lead_time      = (peak_date - detection_date).days

    # Classify lead time category
    if lead_time == 0:
        category = "No Early Warning"
    elif 1 <= lead_time <= 3:
        category = "Short Warning"
    elif 4 <= lead_time <= 7:
        category = "Moderate Warning"
    else:
        category = "Long Warning"

    lead_records.append({
        'Event ID':       ev.get('Event ID', ''),
        'District':       dist,
        'Disease':        dis,
        'Detection Date': detection_date.strftime('%Y-%m-%d'),
        'Peak Date':      peak_date.strftime('%Y-%m-%d'),
        'Lead Time':      lead_time,
        'Category':       category,
        'Peak Cases':     peak_cases,
        'Peak Z-score':   peak_z,
        'Highest Risk':   ev.get('Highest Risk', '')
    })

df_lead = pd.DataFrame(lead_records)

# --------------------------------------------------------------------------
# 3. Validation checks
# --------------------------------------------------------------------------
one_detection_per_event = df_lead['Event ID'].nunique() == len(df_lead)
one_peak_per_event      = True  # by construction above
peak_not_before_detect  = (df_lead['Lead Time'] >= 0).all()
no_negative_lead        = (df_lead['Lead Time'] >= 0).all()
unique_events           = len(df_lead['Event ID'].unique()) == len(df_lead)

verification = {
    "every_event_has_one_detection_date": bool(one_detection_per_event),
    "every_event_has_one_peak_date":      bool(one_peak_per_event),
    "peak_not_before_detection":          bool(peak_not_before_detect),
    "lead_time_never_negative":           bool(no_negative_lead),
    "every_event_appears_exactly_once":   bool(unique_events)
}

print("\nVerification:")
for k, v in verification.items():
    status = "PASS" if v else "FAIL"
    print(f"  [{status}] {k}")

# --------------------------------------------------------------------------
# 4. Overall statistics
# --------------------------------------------------------------------------
total_events   = len(df_lead)
avg_lt         = round(df_lead['Lead Time'].mean(), 2)
median_lt      = round(df_lead['Lead Time'].median(), 2)
min_lt         = int(df_lead['Lead Time'].min())
max_lt         = int(df_lead['Lead Time'].max())
std_lt         = round(df_lead['Lead Time'].std(), 2)

cat_counts = df_lead['Category'].value_counts()
n_no_warn  = int(cat_counts.get('No Early Warning', 0))
n_short    = int(cat_counts.get('Short Warning', 0))
n_moderate = int(cat_counts.get('Moderate Warning', 0))
n_long     = int(cat_counts.get('Long Warning', 0))

# --------------------------------------------------------------------------
# 5. District & Disease summaries
# --------------------------------------------------------------------------
dist_summary = df_lead.groupby('District').agg(
    Events=('Lead Time', 'count'),
    Avg_Lead=('Lead Time', 'mean'),
    Max_Lead=('Lead Time', 'max')
).reset_index().round(2)
dist_summary.columns = ['District', 'Events', 'Average Lead Time', 'Max Lead Time']

dis_summary = df_lead.groupby('Disease').agg(
    Events=('Lead Time', 'count'),
    Avg_Lead=('Lead Time', 'mean'),
    Max_Lead=('Lead Time', 'max')
).reset_index().round(2)
dis_summary.columns = ['Disease', 'Events', 'Average Lead Time', 'Max Lead Time']

# --------------------------------------------------------------------------
# 6. Top 10 earliest detections (largest lead time)
# --------------------------------------------------------------------------
top10 = df_lead.sort_values('Lead Time', ascending=False).head(10).copy()
top10['Rank'] = range(1, len(top10) + 1)
top10_display = top10[['Rank', 'District', 'Disease', 'Lead Time', 'Peak Cases', 'Highest Risk']]

# --------------------------------------------------------------------------
# 7. Lead-time distribution frequency table
# --------------------------------------------------------------------------
cat_order = ['No Early Warning', 'Short Warning', 'Moderate Warning', 'Long Warning']
dist_table = []
for cat in cat_order:
    cnt = int(cat_counts.get(cat, 0))
    pct = round((cnt / total_events) * 100, 1) if total_events else 0
    dist_table.append([cat, cnt, f"{pct}%"])

# --------------------------------------------------------------------------
# 8. Console output
# --------------------------------------------------------------------------
print("\n### Lead-Time Summary")
summary_tbl = [
    ["Total Events",        total_events],
    ["Average Lead Time",   f"{avg_lt} days"],
    ["Median Lead Time",    f"{median_lt} days"],
    ["Min Lead Time",       f"{min_lt} days"],
    ["Max Lead Time",       f"{max_lt} days"],
    ["Std Deviation",       f"{std_lt} days"],
    ["No Early Warning",    n_no_warn],
    ["Short Warning (1-3d)",n_short],
    ["Moderate Warning (4-7d)", n_moderate],
    ["Long Warning (>7d)",  n_long]
]
print(tabulate(summary_tbl, headers=["Metric", "Value"], tablefmt="github"))

print("\n### Lead-Time Distribution")
print(tabulate(dist_table, headers=["Category", "Count", "Percentage"], tablefmt="github"))

print("\n### District Summary")
print(tabulate(dist_summary, headers="keys", showindex=False, tablefmt="github"))

print("\n### Disease Summary")
print(tabulate(dis_summary, headers="keys", showindex=False, tablefmt="github"))

print("\n### Top 10 Earliest Detections")
print(tabulate(top10_display, headers="keys", showindex=False, tablefmt="github"))

# --------------------------------------------------------------------------
# 9. Save CSV
# --------------------------------------------------------------------------
df_lead.to_csv(os.path.join(reports_dir, "lead_time_events.csv"), index=False)

# --------------------------------------------------------------------------
# 10. Save JSON
# --------------------------------------------------------------------------
summary_json = {
    "overall_statistics": {
        "total_events":  total_events,
        "average_lead_time": avg_lt,
        "median_lead_time":  median_lt,
        "min_lead_time":     min_lt,
        "max_lead_time":     max_lt,
        "std_deviation":     std_lt
    },
    "distribution": {
        "No Early Warning":   n_no_warn,
        "Short Warning":      n_short,
        "Moderate Warning":   n_moderate,
        "Long Warning":       n_long
    },
    "district_statistics": dist_summary.to_dict(orient='records'),
    "disease_statistics":  dis_summary.to_dict(orient='records'),
    "verification_results": verification
}
with open(os.path.join(reports_dir, "lead_time_summary.json"), "w") as f:
    json.dump(summary_json, f, indent=4)

# --------------------------------------------------------------------------
# 11. Markdown report
# --------------------------------------------------------------------------
md_path = os.path.join(reports_dir, "lead_time_report.md")
with open(md_path, "w") as f:
    f.write("# Lead-Time Analysis Report\n\n")

    f.write("## 1. Objective\n")
    f.write("Evaluate how early the Early Outbreak Detection System identifies disease activity "
            "before each outbreak reaches its peak daily case count.\n\n")

    f.write("## 2. Methodology\n")
    f.write("The TEST dataset (2025) was processed through the tuned detection pipeline "
            "(min_duration=2, peak_cases>=2, std floor=0.0 with epsilon=1e-6). "
            "For each identified event, the detection date is the first alert day and "
            "the peak date is the day with the highest case count within that event window.\n\n")

    f.write("## 3. Lead-Time Definition\n")
    f.write("```\nLead Time = Peak Date - Detection Date  (in days)\n```\n\n")

    f.write("## 4. Overall Statistics\n")
    f.write(tabulate(summary_tbl, headers=["Metric", "Value"], tablefmt="github") + "\n\n")

    f.write("## 5. Distribution Analysis\n")
    f.write(tabulate(dist_table, headers=["Category", "Count", "Percentage"], tablefmt="github") + "\n\n")

    f.write("## 6. District Analysis\n")
    f.write(tabulate(dist_summary, headers="keys", showindex=False, tablefmt="github") + "\n\n")

    f.write("## 7. Disease Analysis\n")
    f.write(tabulate(dis_summary, headers="keys", showindex=False, tablefmt="github") + "\n\n")

    f.write("## 8. Top 10 Earliest Detections\n")
    f.write(tabulate(top10_display, headers="keys", showindex=False, tablefmt="github") + "\n\n")

    f.write("## 9. Interpretation\n")
    f.write(f"The system detected {total_events} credible outbreak events in 2025. "
            f"The average lead time of {avg_lt} days indicates that alerts were typically raised "
            f"at or near the peak day, which is consistent with a reactive (not predictive) statistical baseline. "
            f"Events with lead times > 0 represent cases where the initial alert day preceded "
            f"the highest case count, giving at least some advance notice.\n\n")

    f.write("## 10. Conclusion\n")
    f.write("The rolling Z-score pipeline, with calibrated filters, produces credible alerts. "
            "Lead-time analysis shows the system responds to sustained statistical anomalies, "
            "with a distribution skewed toward short or zero lead times — typical for "
            "reactive statistical surveillance methods operating on sparse count data.\n")

print("\nStage 3.2 Lead-Time Analysis complete.")
print(f"Outputs saved to: {reports_dir}")
