import pandas as pd
import numpy as np
import json
import sys
from tabulate import tabulate

in_path = r"C:\BRAIN-STORM\HT\warning\outbreak_detection_system\data\processed\train_zscore.pkl"
out_data = r"C:\BRAIN-STORM\HT\warning\outbreak_detection_system\data\processed\train_risk_levels.pkl"
out_json = r"C:\BRAIN-STORM\HT\warning\outbreak_detection_system\data\processed\train_risk_summary.json"

try:
    df = pd.read_pickle(in_path)
    
    # 1. Validation logic
    expected_cols = {'diagnosis_date', 'district', 'disease_name', 'case_count', 'rolling_mean_30', 'rolling_std_30', 'ewma_14', 'rolling_z_score'}
    actual_cols = set(df.columns)
    
    if not expected_cols.issubset(actual_cols):
        print(f"ABORT: Missing columns. Expected {expected_cols}, got {actual_cols}")
        sys.exit(1)
        
    is_sorted = df.groupby(['district', 'disease_name'])['diagnosis_date'].is_monotonic_increasing.all()
    if not is_sorted:
        print("ABORT: Dataset is not chronologically sorted.")
        sys.exit(1)
        
    if len(df) != 105168:
        print(f"ABORT: Total rows = {len(df)}, expected 105168.")
        sys.exit(1)
        
    counts_per_pair = df.groupby(['district', 'disease_name']).size()
    if len(counts_per_pair) != 48:
        print(f"ABORT: Number of district-disease series = {len(counts_per_pair)}, expected 48.")
        sys.exit(1)
        
    # 2. Assign risk levels
    conditions = [
        (df['rolling_z_score'] < 2.0),
        (df['rolling_z_score'] >= 2.0) & (df['rolling_z_score'] < 2.5),
        (df['rolling_z_score'] >= 2.5) & (df['rolling_z_score'] < 3.0),
        (df['rolling_z_score'] >= 3.0)
    ]
    choices = ['Low', 'Medium', 'High', 'Critical']
    df['risk_level'] = np.select(conditions, choices, default=np.nan)
    
    # 3. Post-execution Validation
    final_rows = len(df)
    no_rows_removed = (final_rows == 105168)
    no_dups = (df.duplicated().sum() == 0)
    is_sorted_after = df.groupby(['district', 'disease_name'])['diagnosis_date'].is_monotonic_increasing.all()
    
    valid_levels = {'Low', 'Medium', 'High', 'Critical'}
    all_valid_levels = set(df['risk_level'].unique()).issubset(valid_levels)
    no_missing = df['risk_level'].isna().sum() == 0
    
    verification_status = bool(no_rows_removed and no_dups and is_sorted_after and all_valid_levels and no_missing)
    
    # 4. Summary metrics
    val_counts = df['risk_level'].value_counts()
    count_low = int(val_counts.get('Low', 0))
    count_medium = int(val_counts.get('Medium', 0))
    count_high = int(val_counts.get('High', 0))
    count_critical = int(val_counts.get('Critical', 0))
    
    pct_low = (count_low / final_rows) * 100
    pct_medium = (count_medium / final_rows) * 100
    pct_high = (count_high / final_rows) * 100
    pct_critical = (count_critical / final_rows) * 100
    
    summary = {
        "total_rows": final_rows,
        "number_of_district_disease_series": int(len(counts_per_pair)),
        "count_low": count_low,
        "count_medium": count_medium,
        "count_high": count_high,
        "count_critical": count_critical,
        "percentage_low": pct_low,
        "percentage_medium": pct_medium,
        "percentage_high": pct_high,
        "percentage_critical": pct_critical,
        "verification_status": verification_status,
        "output_file_locations": {
            "dataset": out_data,
            "report": out_json
        }
    }
    
    df.to_pickle(out_data)
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=4)
        
    # Table 1 - Overall Risk Distribution
    table1 = [
        ["Low", count_low, f"{pct_low:.2f}%"],
        ["Medium", count_medium, f"{pct_medium:.2f}%"],
        ["High", count_high, f"{pct_high:.2f}%"],
        ["Critical", count_critical, f"{pct_critical:.2f}%"],
        ["Total", final_rows, "100.00%"]
    ]
    print("\n### Table 1 - Overall Risk Distribution")
    print(tabulate(table1, headers=["Risk Level", "Count", "Percentage"], tablefmt="github"))
    
    # Table 2 - Distribution by District and Disease
    dist_disease_dist = df.groupby(['district', 'disease_name', 'risk_level']).size().unstack(fill_value=0)
    for c in ['Low', 'Medium', 'High', 'Critical']:
        if c not in dist_disease_dist.columns:
            dist_disease_dist[c] = 0
    dist_disease_dist = dist_disease_dist[['Low', 'Medium', 'High', 'Critical']]
    dist_disease_dist['Total'] = dist_disease_dist.sum(axis=1)
    dist_disease_dist = dist_disease_dist.reset_index()
    
    print("\n### Table 2 - Risk Distribution by District and Disease")
    print(tabulate(dist_disease_dist, headers="keys", showindex=False, tablefmt="github"))
    
    # Table 3 - Sample Output
    display_df = df[['diagnosis_date', 'district', 'disease_name', 'case_count', 'rolling_mean_30', 'rolling_std_30', 'ewma_14', 'rolling_z_score', 'risk_level']].head(10).copy()
    display_df['diagnosis_date'] = display_df['diagnosis_date'].dt.strftime('%Y-%m-%d')
    for col in ['rolling_mean_30', 'rolling_std_30', 'ewma_14', 'rolling_z_score']:
        display_df[col] = display_df[col].round(3)
        
    print("\n### Table 3 - Sample Output")
    display_df.columns = ["Date", "District", "Disease", "Daily Cases", "Rolling Mean", "Rolling Std", "EWMA", "Z-score", "Risk Level"]
    print(tabulate(display_df, headers="keys", showindex=False, tablefmt="github"))
    
    print("\n--- CLASSIFICATION COMPLETE ---")

except Exception as e:
    print(f"Error computing risk classification: {e}")
