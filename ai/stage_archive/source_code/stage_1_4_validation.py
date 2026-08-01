import pandas as pd
import numpy as np
import json

in_path = r"C:\BRAIN-STORM\HT\warning\outbreak_detection_system\data\interim\continuous_daily_series.pkl"
out_csv = r"C:\BRAIN-STORM\HT\warning\outbreak_detection_system\data\processed\series_statistics.csv"
out_json = r"C:\BRAIN-STORM\HT\warning\outbreak_detection_system\data\processed\validation_report.json"

try:
    df = pd.read_pickle(in_path)
    
    # Validation logic
    validations = {}
    
    # 1. Verify every series contains exactly 2,919 daily observations
    counts = df.groupby(['district', 'disease_name']).size()
    validations["every_series_has_2919_days"] = bool((counts == 2919).all())
    
    # 2. Verify there are no duplicate dates within any series
    dups = df.duplicated(subset=['district', 'disease_name', 'diagnosis_date']).sum()
    validations["no_duplicate_dates_in_series"] = bool(dups == 0)
    
    # 3. Verify there are no missing dates
    def no_missing(group):
        return (group['diagnosis_date'].max() - group['diagnosis_date'].min()).days + 1 == 2919
    no_miss = df.groupby(['district', 'disease_name']).apply(no_missing).all()
    validations["no_missing_dates"] = bool(no_miss)
    
    # 4. Verify all case_count values are non-negative integers
    is_non_neg = bool((df['case_count'] >= 0).all())
    is_int = bool(pd.api.types.is_integer_dtype(df['case_count']))
    validations["all_non_negative_integers"] = bool(is_non_neg and is_int)
    
    # 5. Verify there are no NaN values
    has_nan = bool(df['case_count'].isna().any())
    validations["no_nan_values"] = not has_nan
    
    # Compute Statistics per series
    grouped = df.groupby(['district', 'disease_name'])
    
    stats_list = []
    for (dist, dis), group in grouped:
        total_days = len(group)
        total_cases = group['case_count'].sum()
        zero_days = (group['case_count'] == 0).sum()
        non_zero_days = total_days - zero_days
        percentage_zero_days = (zero_days / total_days) * 100
        mean_daily_cases = group['case_count'].mean()
        median_daily_cases = group['case_count'].median()
        standard_deviation = group['case_count'].std()
        minimum_daily_cases = group['case_count'].min()
        maximum_daily_cases = group['case_count'].max()
        variance = group['case_count'].var()
        first_date = group['diagnosis_date'].min().strftime('%Y-%m-%d')
        last_date = group['diagnosis_date'].max().strftime('%Y-%m-%d')
        
        stats_list.append({
            'district': dist,
            'disease_name': dis,
            'total_days': total_days,
            'total_cases': total_cases,
            'non_zero_days': non_zero_days,
            'zero_days': zero_days,
            'percentage_zero_days': percentage_zero_days,
            'mean_daily_cases': mean_daily_cases,
            'median_daily_cases': median_daily_cases,
            'standard_deviation': standard_deviation,
            'minimum_daily_cases': minimum_daily_cases,
            'maximum_daily_cases': maximum_daily_cases,
            'variance': variance,
            'first_date': first_date,
            'last_date': last_date
        })
        
    stats_df = pd.DataFrame(stats_list)
    
    # Overall summary
    total_series = len(stats_df)
    avg_total_cases = stats_df['total_cases'].mean()
    avg_pct_zero = stats_df['percentage_zero_days'].mean()
    
    highest_cases_series = stats_df.loc[stats_df['total_cases'].idxmax()]
    highest_cases_str = f"{highest_cases_series['district']} - {highest_cases_series['disease_name']} ({highest_cases_series['total_cases']} cases)"
    
    lowest_cases_series = stats_df.loc[stats_df['total_cases'].idxmin()]
    lowest_cases_str = f"{lowest_cases_series['district']} - {lowest_cases_series['disease_name']} ({lowest_cases_series['total_cases']} cases)"
    
    highest_max_series = stats_df.loc[stats_df['maximum_daily_cases'].idxmax()]
    highest_max_str = f"{highest_max_series['district']} - {highest_max_series['disease_name']} (Max: {highest_max_series['maximum_daily_cases']} cases)"
    
    overall_mean = df['case_count'].mean()
    overall_std = df['case_count'].std()
    
    print("--- OVERALL SUMMARY ---")
    print(f"Total number of time series: {total_series}")
    print(f"Average total cases per series: {avg_total_cases:.2f}")
    print(f"Average percentage of zero-case days: {avg_pct_zero:.2f}%")
    print(f"Series with highest total cases: {highest_cases_str}")
    print(f"Series with lowest total cases: {lowest_cases_str}")
    print(f"Series with highest maximum daily count: {highest_max_str}")
    print(f"Overall mean of daily case counts: {overall_mean:.4f}")
    print(f"Overall standard deviation: {overall_std:.4f}")
    
    print("\n--- FIRST 10 ROWS OF STATISTICS ---")
    print(stats_df.head(10).to_string(index=False))
    
    # Save statistics
    stats_df.to_csv(out_csv, index=False)
    
    # Save validation report
    with open(out_json, 'w') as f:
        json.dump(validations, f, indent=4)
        
    print(f"\nSaved statistics to: {out_csv}")
    print(f"Saved validation report to: {out_json}")
    
except Exception as e:
    print(f"Error computing statistics: {e}")
