import os
import re

base_dir = r"C:\BRAIN-STORM\HT\warning\outbreak_detection_system"

def replace_in_file(rel_path, replacements, note=None):
    path = os.path.join(base_dir, rel_path)
    if not os.path.exists(path):
        return
    
    # Try reading with utf-8, fallback to latin-1
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            content = f.read()
            
    new_content = content
    for old, new in replacements:
        new_content = new_content.replace(old, new)
        
    if note and note not in new_content:
        new_content += "\n\n" + note + "\n"
        
    if new_content != content:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Updated {rel_path}")

# 1. Update Stage 3.4 code and output
stage_34_replacements = [
    ("'Confirmed'", "'Confirmed-Tier Event'"),
    ("'Watch'", "'Watch-Tier Event'"),
    ("Confirmed (Tier 1)", "Confirmed-Tier Event (Tier 1)"),
    ("Watch (Tier 2)", "Watch-Tier Event (Tier 2)"),
    ("Tier 1 Confirmed", "Tier 1 Confirmed-Tier Event"),
    ("Tier 2 Watch", "Tier 2 Watch-Tier Event")
]
replace_in_file(r"stage_archive\source_code\stage_3_4_two_tier_alerts.py", stage_34_replacements)

# Also CSV
csv_replacements = [
    ("Confirmed", "Confirmed-Tier Event"),
    ("Watch", "Watch-Tier Event")
]
replace_in_file(r"reports\two_tier_alerts_2025.csv", csv_replacements)


# 2. Update Stage 3.5 code
stage_35_replacements = [
    ("'Confirmed'", "'Confirmed-Tier Event'"),
    ("'Watch'", "'Watch-Tier Event'")
]
replace_in_file(r"stage_archive\source_code\stage_3_5_dataset_verification.py", stage_35_replacements)


# 3. Update Stage 3.6 code and output
stage_36_replacements = [
    ("'Watch'", "'Watch-Status Warning'"),
    ("Status == 'Watch'", "Status == 'Watch-Status Warning'"),
    ("== 'Watch'", "== 'Watch-Status Warning'"),
    ("Watches", "Watch-Status Warnings")
]
replace_in_file(r"stage_archive\source_code\stage_3_6_regional_warning_report.py", stage_36_replacements)


# Note for markdown files
clarifying_note = "> **Note:** This report uses [Watch-Tier Event / Watch-Status Warning] terminology, which is distinct from the other report's equivalent term. **Watch-Tier Event** refers to a year-level statistical classification based on sustained low-peak-case anomalies; **Watch-Status Warning** refers to a real-time weekly risk snapshot based on the highest observed daily risk level."

# 4. Update historical_replay_report.md
historical_replacements = [
    ("## 13. Two-Tier Alerting System (Stage 3.4)", "## 13. Two-Tier Alerting System (Stage 3.4)"),
    ("**Watch** tier", "**Watch-Tier Event** tier"),
    ("**Confirmed** tier", "**Confirmed-Tier Event** tier"),
    ("| **Confirmed** |", "| **Confirmed-Tier Event** |"),
    ("| **Watch** |", "| **Watch-Tier Event** |"),
    ("Confirmed (Tier 1)", "Confirmed-Tier Event"),
    ("Watch (Tier 2)", "Watch-Tier Event"),
    ("Watch-tier", "Watch-Tier Event"),
    ("Watch-Tier Event tier", "Watch-Tier Event layer"),
    ("Watch-Tier Event Alerts", "Watch-Tier Events")
]
replace_in_file(r"reports\historical_replay_report.md", historical_replacements, note=clarifying_note)

# 5. Update regional snapshot report
snapshot_replacements = [
    ("## ⚠️ Watches", "## ⚠️ Watch-Status Warnings"),
    ("**Watch:**", "**Watch-Status Warning:**"),
    ("| Watch |", "| Watch-Status Warning |"),
    ("Watches:", "Watch-Status Warnings:")
]
replace_in_file(r"reports\regional_warning_snapshot_2025-12-18.md", snapshot_replacements, note=clarifying_note)


# 6. Update the pkl file explicitly via pandas
import pandas as pd
pkl_path = os.path.join(base_dir, r"data\processed\test_detection_results.pkl")
if os.path.exists(pkl_path):
    df = pd.read_pickle(pkl_path)
    if 'tier' in df.columns:
        df['tier'] = df['tier'].replace({
            'Confirmed': 'Confirmed-Tier Event',
            'Watch': 'Watch-Tier Event'
        })
        df.to_pickle(pkl_path)
        print("Updated tier column in test_detection_results.pkl")

print("\nAll files successfully updated. Ready for git diff.")
