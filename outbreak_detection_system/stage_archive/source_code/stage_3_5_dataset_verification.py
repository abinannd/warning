"""
Stage 3.5 - Dataset Verification and Preparation
Verifies the state of data/processed/test_detection_results.pkl.
If it contains contaminated baseline results (max_z > 4), it regenerates
the file using the gap-corrected baseline and two-tier logic from Stage 3.4.
"""

import os
import shutil
import pandas as pd
import numpy as np

base_dir    = r"C:\BRAIN-STORM\HT\warning\outbreak_detection_system"
data_dir    = os.path.join(base_dir, "data", "processed")
test_file   = os.path.join(data_dir, "test_detection_results.pkl")
backup_file = os.path.join(data_dir, "test_detection_results_ORIGINAL_BACKUP.pkl")

EPSILON      = 1e-6
MIN_DURATION = 2

def classify_risk(z_series):
    return np.select(
        [z_series < 2.0,
         (z_series >= 2.0) & (z_series < 2.5),
         (z_series >= 2.5) & (z_series < 3.0),
         z_series >= 3.0],
        ['Low', 'Medium', 'High', 'Critical'],
        default='Low'
    )

print("=== STEP 1: Inspecting current test_detection_results.pkl ===")
df = pd.read_pickle(test_file)
print(f"Columns present: {list(df.columns)}")

# Identify artifact pattern
artifact_mask = (df['case_count'] == 1) & (df['rolling_z_score'] > 4)
artifact_count = artifact_mask.sum()
print(f"Artifact pattern (case=1, z>4) count: {artifact_count}")

max_z = df['rolling_z_score'].max()
print(f"Max Z-score: {max_z:.4f}")

if max_z > 4.0:
    print("\nDetecting OLD contaminated baseline logic (max z > 4.0).")
    print("=== STEP 3: Regenerating test_detection_results.pkl ===")
    
    print(f"Backing up original file to: {backup_file}")
    shutil.copy2(test_file, backup_file)
    
    print("Loading full timeseries to compute gap-corrected baseline...")
    df_train = pd.read_pickle(os.path.join(data_dir, "train_timeseries.pkl"))
    df_val   = pd.read_pickle(os.path.join(data_dir, "validation_timeseries.pkl"))
    df_test  = pd.read_pickle(os.path.join(data_dir, "test_timeseries.pkl"))

    for d in [df_train, df_val, df_test]:
        d['diagnosis_date'] = pd.to_datetime(d['diagnosis_date'])

    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
    df_all = df_all.sort_values(['district', 'disease_name', 'diagnosis_date']).reset_index(drop=True)

    test_min = df_test['diagnosis_date'].min()
    test_max = df_test['diagnosis_date'].max()

    results = []
    for (dist, dis), g in df_all.groupby(['district', 'disease_name']):
        g = g.copy().reset_index(drop=True)
        # Gap-corrected baseline
        b_mean = g['case_count'].rolling(30, min_periods=15).mean().shift(8).fillna(0)
        b_std  = g['case_count'].rolling(30, min_periods=15).std().shift(8).fillna(0)
        r_mean = g['case_count'].rolling(7, min_periods=1).mean()
        ewma   = g['case_count'].ewm(span=14, adjust=False).mean().shift(8).fillna(0)
        
        std_safe = b_std.copy()
        std_safe[std_safe > 0] = std_safe[std_safe > 0].clip(lower=EPSILON)
        std_safe[std_safe == 0] = np.nan
        
        z_raw = (r_mean - b_mean) / std_safe
        g['gap_z'] = z_raw.fillna(0)
        
        g['rolling_mean_30'] = b_mean
        g['rolling_std_30']  = b_std
        g['ewma_14']         = ewma
        
        results.append(g)

    df_all = pd.concat(results, ignore_index=True)
    df_all = df_all.sort_values(['district', 'disease_name', 'diagnosis_date']).reset_index(drop=True)

    # Slice to test set
    df_test_gap = df_all[
        (df_all['diagnosis_date'] >= test_min) & (df_all['diagnosis_date'] <= test_max)
    ].copy()
    
    df_test_gap['rolling_z_score'] = df_test_gap['gap_z']
    df_test_gap['risk_level'] = classify_risk(df_test_gap['rolling_z_score'])

    # Tier assignment
    # Re-run event extraction to tag tiers to rows
    risk_order   = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
    df_test_gap['tier'] = 'None'
    
    for (dist, dis), grp in df_test_gap.groupby(['district', 'disease_name']):
        grp = grp.sort_values('diagnosis_date')
        in_event = False
        cur      = None
        
        for idx, row in grp.iterrows():
            risk = row['risk_level']
            if risk != 'Low':
                if not in_event:
                    in_event = True
                    cur = {
                        'start_idx': idx,
                        'Peak Cases': row['case_count'],
                        'idxs': [idx]
                    }
                else:
                    cur['Peak Cases'] = max(cur['Peak Cases'], row['case_count'])
                    cur['idxs'].append(idx)
            else:
                if in_event:
                    in_event = False
                    dur = len(cur['idxs'])
                    if dur >= MIN_DURATION:
                        tier_label = 'Confirmed' if cur['Peak Cases'] >= 2 else 'Watch'
                        df_test_gap.loc[cur['idxs'], 'tier'] = tier_label
                    cur = None

        if in_event:
            dur = len(cur['idxs'])
            if dur >= MIN_DURATION:
                tier_label = 'Confirmed' if cur['Peak Cases'] >= 2 else 'Watch'
                df_test_gap.loc[cur['idxs'], 'tier'] = tier_label

    cols_to_keep = ['district', 'disease_name', 'diagnosis_date', 'case_count',
                    'rolling_mean_30', 'rolling_std_30', 'ewma_14', 
                    'rolling_z_score', 'risk_level', 'tier']
    df_test_out = df_test_gap[cols_to_keep].copy()
    
    df_test_out.to_pickle(test_file)
    print(f"Overwrote {test_file} with gap-corrected and tiered dataset.")
    
    print("\n=== STEP 4: Final Confirmation ===")
    df_final = pd.read_pickle(test_file)
    print(f"Rows: {len(df_final)}")
    print(f"Max Z-score: {df_final['rolling_z_score'].max():.4f}")
    artifact_mask_final = (df_final['case_count'] == 1) & (df_final['rolling_z_score'] > 4)
    print(f"Artifact pattern (case=1, z>4) count: {artifact_mask_final.sum()}")
    print("Tier counts:")
    print(df_final['tier'].value_counts().to_string())
else:
    print("\n=== STEP 4: Final Confirmation ===")
    print("File already uses gap-corrected baseline (max z <= 4.0).")
    print(f"Rows: {len(df)}")
    print(f"Max Z-score: {max_z:.4f}")
    if 'tier' in df.columns:
        print("Tier counts:")
        print(df['tier'].value_counts().to_string())
    else:
        print("NOTE: 'tier' column is missing, file might need regeneration to add it.")
