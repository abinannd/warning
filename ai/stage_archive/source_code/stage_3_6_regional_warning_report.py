"""
Stage 3.6 - Regional Warning Generation Snapshot

Generates a weekly regional warning report snapshot for a specific week.
Target week: 2025-12-12 to 2025-12-18 (Contains Palakkad-Chikungunya Confirmed event).

Mapping:
- Confirmed -> Emergency Warning
- Watch -> Watch
- Elevated Z-score (>2.0) but no event -> Advisory
- Else -> Normal
"""

import os
import pandas as pd
from tabulate import tabulate

base_dir    = r"C:\BRAIN-STORM\HT\warning\outbreak_detection_system"
data_dir    = os.path.join(base_dir, "data", "processed")
reports_dir = os.path.join(base_dir, "reports")

# Target snapshot week
start_date = pd.to_datetime('2025-12-12')
end_date   = pd.to_datetime('2025-12-18')

print("Loading test detection results...")
df = pd.read_pickle(os.path.join(data_dir, "test_detection_results.pkl"))
df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'])

# Filter to target week
mask = (df['diagnosis_date'] >= start_date) & (df['diagnosis_date'] <= end_date)
df_week = df[mask].copy()

# Determine highest status for each District-Disease in this week
# Priorities: Emergency (4) > Watch (3) > Advisory (2) > Normal (1)

def get_status(grp):
    if (grp['tier'] == 'Confirmed-Tier Event').any():
        return 4, 'Emergency Warning'
    elif (grp['tier'] == 'Watch-Tier Event').any():
        return 3, 'Watch-Status Warning'
    elif (grp['risk_level'] != 'Low').any():
        return 2, 'Advisory'
    else:
        return 1, 'Normal'

records = []
for (dist, dis), grp in df_week.groupby(['district', 'disease_name']):
    priority, status = get_status(grp)
    max_z = grp['rolling_z_score'].max()
    max_cases = grp['case_count'].max()
    records.append({
        'District': dist,
        'Disease': dis,
        'Status': status,
        'Priority': priority,
        'Max Z-Score': round(max_z, 3),
        'Peak Cases': max_cases
    })

df_status = pd.DataFrame(records)
df_status = df_status.sort_values(by=['Priority', 'Max Z-Score'], ascending=[False, False])

# Drop Priority for display
display_cols = ['District', 'Disease', 'Status', 'Max Z-Score', 'Peak Cases']
df_display = df_status[display_cols]

# Markdown generation
md_lines = []
md_lines.append(f"# Regional Outbreak Warning Report")
md_lines.append(f"**Snapshot Window:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n")

md_lines.append("> **Note for Demo:** This snapshot was specifically chosen for a week known to contain a Confirmed-Tier Event (Palakkad–Chikungunya) to demonstrate the multi-tiered alerting behavior: Advisory, Watch-Status Warning, and Emergency Warning.\n")

md_lines.append("> **Note:** This report uses [Watch-Tier Event / Watch-Status Warning] terminology, which is distinct from the other report's equivalent term. **Watch-Tier Event** refers to a year-level statistical classification based on sustained low-peak-case anomalies; **Watch-Status Warning** refers to a real-time weekly risk snapshot based on the highest observed daily risk level.\n")

md_lines.append("## Status Definitions")
md_lines.append("- **Emergency Warning:** Confirmed sustained anomaly (≥2 days, peak cases ≥2, gap-corrected baseline). Immediate public health intervention recommended.")
md_lines.append("- **Watch:** High sensitivity signal (≥2 days, peak cases=1). Elevated risk, requires monitoring.")
md_lines.append("- **Advisory:** Elevated statistical activity (Z-score > 2.0) but lacks sustained duration. Preliminary heads-up.")
md_lines.append("- **Normal:** Disease activity within expected historical baseline bounds.\n")

# Emergency
em_df = df_display[df_display['Status'] == 'Emergency Warning']
md_lines.append("## 🚨 Emergency Warnings")
if not em_df.empty:
    md_lines.append(tabulate(em_df, headers='keys', showindex=False, tablefmt='github') + "\n")
else:
    md_lines.append("No Emergency Warnings for this period.\n")

# Watch
wa_df = df_display[df_display['Status'] == 'Watch-Status Warning']
md_lines.append("## ⚠️ Watch-Status Warnings")
if not wa_df.empty:
    md_lines.append(tabulate(wa_df, headers='keys', showindex=False, tablefmt='github') + "\n")
else:
    md_lines.append("No Watch-Status Warnings for this period.\n")

# Advisory
ad_df = df_display[df_display['Status'] == 'Advisory']
md_lines.append("## ℹ️ Advisories")
if not ad_df.empty:
    md_lines.append(tabulate(ad_df, headers='keys', showindex=False, tablefmt='github') + "\n")
else:
    md_lines.append("No Advisories for this period.\n")

out_file = os.path.join(reports_dir, f"regional_warning_snapshot_{end_date.strftime('%Y-%m-%d')}.md")
with open(out_file, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print(f"\nGenerated report: {out_file}")

# Print summary to console
print("\nSnapshot Summary:")
print(f"Emergency Warnings: {len(em_df)}")
print(f"Watch-Status Warnings:            {len(wa_df)}")
print(f"Advisories:         {len(ad_df)}")
print("\nEmergency Table:")
print(tabulate(em_df, headers='keys', showindex=False, tablefmt='github'))
