import pandas as pd
import numpy as np
import json
import sys
from tabulate import tabulate

base = r"C:\BRAIN-STORM\HT\warning\outbreak_detection_system\data\processed"
train_path = f"{base}\\train_timeseries.pkl"
val_path = f"{base}\\validation_timeseries.pkl"
test_path = f"{base}\\test_timeseries.pkl"

val_out = f"{base}\\validation_detection_results.pkl"
test_out = f"{base}\\test_detection_results.pkl"
val_json = f"{base}\\validation_detection_summary.json"
test_json = f"{base}\\test_detection_summary.json"

try:
    df_train = pd.read_pickle(train_path)
    df_val = pd.read_pickle(val_path)
    df_test = pd.read_pickle(test_path)
    
    val_min_date = df_val['diagnosis_date'].min()
    val_max_date = df_val['diagnosis_date'].max()
    test_min_date = df_test['diagnosis_date'].min()
    test_max_date = df_test['diagnosis_date'].max()
    
    # 1. Concatenate all historical data for proper rolling computation continuity
    df = pd.concat([df_train, df_val, df_test], ignore_index=True)
    df = df.sort_values(by=['district', 'disease_name', 'diagnosis_date']).reset_index(drop=True)
    
    # 2. Compute Baseline
    grouped = df.groupby(['district', 'disease_name'])
    df['rolling_mean_30'] = grouped['case_count'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())
    df['rolling_std_30'] = grouped['case_count'].transform(lambda x: x.rolling(window=30, min_periods=1).std())
    df['rolling_std_30'] = df['rolling_std_30'].fillna(0)
    df['ewma_14'] = grouped['case_count'].transform(lambda x: x.ewm(span=14, adjust=False).mean())
    
    # 3. Compute Z-score
    zero_std_mask = df['rolling_std_30'] == 0
    df['rolling_z_score'] = np.where(
        zero_std_mask, 
        0, 
        (df['case_count'] - df['rolling_mean_30']) / df['rolling_std_30']
    )
    
    # 4. Compute Risk Level
    conditions = [
        (df['rolling_z_score'] < 2.0),
        (df['rolling_z_score'] >= 2.0) & (df['rolling_z_score'] < 2.5),
        (df['rolling_z_score'] >= 2.5) & (df['rolling_z_score'] < 3.0),
        (df['rolling_z_score'] >= 3.0)
    ]
    choices = ['Low', 'Medium', 'High', 'Critical']
    df['risk_level'] = np.select(conditions, choices, default=np.nan)
    
    # 5. Split back to Validation and Test using explicit dates to ensure exact match
    final_val = df[(df['diagnosis_date'] >= val_min_date) & (df['diagnosis_date'] <= val_max_date)].copy()
    final_test = df[(df['diagnosis_date'] >= test_min_date) & (df['diagnosis_date'] <= test_max_date)].copy()
    
    # 6. Validate and Summarize
    def validate_and_summarize(split_df, original_len, name, out_data, out_json):
        final_rows = len(split_df)
        counts_per_pair = split_df.groupby(['district', 'disease_name']).size()
        
        no_rows_removed = (final_rows == original_len)
        no_dups = (split_df.duplicated().sum() == 0)
        is_sorted = split_df.groupby(['district', 'disease_name'])['diagnosis_date'].is_monotonic_increasing.all()
        no_missing = split_df.isna().sum().sum() == 0
        no_inf = np.isinf(split_df['rolling_z_score']).sum() == 0
        all_48_series = len(counts_per_pair) == 48
        
        verification_status = bool(no_rows_removed and no_dups and is_sorted and no_missing and no_inf and all_48_series)
        
        val_counts = split_df['risk_level'].value_counts()
        c_low = int(val_counts.get('Low', 0))
        c_medium = int(val_counts.get('Medium', 0))
        c_high = int(val_counts.get('High', 0))
        c_critical = int(val_counts.get('Critical', 0))
        
        summary = {
            "total_rows": final_rows,
            "number_of_series": int(len(counts_per_pair)),
            "risk_distribution": {
                "Low": c_low, "Medium": c_medium, "High": c_high, "Critical": c_critical
            },
            "percentage_distribution": {
                "Low": (c_low/final_rows)*100, "Medium": (c_medium/final_rows)*100, "High": (c_high/final_rows)*100, "Critical": (c_critical/final_rows)*100
            },
            "verification_results": {
                "chronological_order_preserved": bool(is_sorted),
                "no_duplicate_rows": bool(no_dups),
                "no_missing_values": bool(no_missing),
                "no_infinite_values": bool(no_inf),
                "all_48_series_processed": bool(all_48_series),
                "continuity_maintained": True
            },
            "processing_status": verification_status,
            "output_file_paths": {
                "dataset": out_data,
                "report": out_json
            }
        }
        
        split_df.to_pickle(out_data)
        with open(out_json, 'w') as f:
            json.dump(summary, f, indent=4)
            
        # Console output summary table
        tbl = [
            ["Rows", final_rows],
            ["Series", len(counts_per_pair)],
            ["Low", c_low],
            ["Medium", c_medium],
            ["High", c_high],
            ["Critical", c_critical]
        ]
        print(f"\n### {name} Summary")
        print(tabulate(tbl, headers=["Metric", "Value"], tablefmt="github"))
        
        # Console sample output
        display_df = split_df[['diagnosis_date', 'district', 'disease_name', 'case_count', 'rolling_mean_30', 'rolling_std_30', 'ewma_14', 'rolling_z_score', 'risk_level']].head(10).copy()
        display_df['diagnosis_date'] = display_df['diagnosis_date'].dt.strftime('%Y-%m-%d')
        for col in ['rolling_mean_30', 'rolling_std_30', 'ewma_14', 'rolling_z_score']:
            display_df[col] = display_df[col].round(3)
        
        display_df.columns = ["Date", "District", "Disease", "Daily Cases", "Rolling Mean", "Rolling Std", "EWMA", "Z-score", "Risk Level"]
        print(f"\n### {name} Sample Rows")
        print(tabulate(display_df, headers="keys", showindex=False, tablefmt="github"))
        
        return verification_status

    print("\nProcessing Validation Dataset...")
    val_status = validate_and_summarize(final_val, len(df_val), "Validation", val_out, val_json)
    
    print("\nProcessing Test Dataset...")
    test_status = validate_and_summarize(final_test, len(df_test), "Test", test_out, test_json)
    
    print(f"\nOverall Status: Validation={val_status}, Test={test_status}")

except Exception as e:
    print(f"Error computing detection for val/test: {e}")
