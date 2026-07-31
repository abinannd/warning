"""
Stage 3.7 - Prophet-Based Forecasting and Anomaly Detection
Target series: Palakkad - Chikungunya

Steps:
  1. Prepare ds/y DataFrame
  2. Train on 2018-2024, test on 2025
  3. Fit Prophet model (yearly + weekly seasonality, 95% CI)
  4. Detect anomalies on test period (actual > yhat_upper)
  5. Compare against gap-corrected Z-score events from Stage 3.4
  6. Generate visualization
  7. Save outputs and print summary

Does NOT modify any existing z-score/EWMA pipeline outputs.
"""

import os
import pickle
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from prophet import Prophet

warnings.filterwarnings('ignore')

base_dir    = r"C:\BRAIN-STORM\HT\warning\outbreak_detection_system"
data_dir    = os.path.join(base_dir, "data", "processed")
reports_dir = os.path.join(base_dir, "reports")
models_dir  = os.path.join(base_dir, "models")

TARGET_DISTRICT = "Palakkad"
TARGET_DISEASE  = "Chikungunya"

# --------------------------------------------------------------------------
# 1. Load and prepare data
# --------------------------------------------------------------------------
print("Step 1: Loading data...")
df_train_ts = pd.read_pickle(os.path.join(data_dir, "train_timeseries.pkl"))
df_val_ts   = pd.read_pickle(os.path.join(data_dir, "validation_timeseries.pkl"))
df_test_ts  = pd.read_pickle(os.path.join(data_dir, "test_timeseries.pkl"))

for df in [df_train_ts, df_val_ts, df_test_ts]:
    df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'])

df_all = pd.concat([df_train_ts, df_val_ts, df_test_ts], ignore_index=True)

series = df_all[
    (df_all['district'] == TARGET_DISTRICT) &
    (df_all['disease_name'] == TARGET_DISEASE)
].sort_values('diagnosis_date').copy()

# Format for Prophet
prophet_df = series[['diagnosis_date', 'case_count']].rename(
    columns={'diagnosis_date': 'ds', 'case_count': 'y'}
)
print(f"  Full series: {len(prophet_df)} rows "
      f"({prophet_df['ds'].min().date()} to {prophet_df['ds'].max().date()})")
print(f"  Total cases: {int(prophet_df['y'].sum())}, Max daily: {int(prophet_df['y'].max())}")

# --------------------------------------------------------------------------
# 2. Train/Test split
# --------------------------------------------------------------------------
train_df = prophet_df[prophet_df['ds'] < '2025-01-01'].copy()
test_df  = prophet_df[prophet_df['ds'] >= '2025-01-01'].copy()
print(f"\nStep 2: Split complete")
print(f"  Train: {len(train_df)} days ({train_df['ds'].min().date()} to {train_df['ds'].max().date()})")
print(f"  Test:  {len(test_df)} days  ({test_df['ds'].min().date()} to {test_df['ds'].max().date()})")

# --------------------------------------------------------------------------
# 3. Fit Prophet model
# --------------------------------------------------------------------------
print("\nStep 3: Fitting Prophet model...")
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    interval_width=0.95,
    changepoint_prior_scale=0.05   # conservative — disease baselines shift slowly
)
model.fit(train_df)
print("  Model fitted successfully.")

model_path = os.path.join(models_dir, "prophet_palakkad_chikungunya.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"  Saved model to: {model_path}")

# --------------------------------------------------------------------------
# 4. Predict on test period & flag anomalies
# --------------------------------------------------------------------------
print("\nStep 4: Generating predictions on 2025 test period...")
future = model.predict(test_df[['ds']])

results = test_df.merge(
    future[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend',
            'yearly', 'weekly']],
    on='ds', how='left'
)

# Clip lower bound at 0 (can't have negative cases)
results['yhat_lower'] = results['yhat_lower'].clip(lower=0)
results['yhat']       = results['yhat'].clip(lower=0)

# Flag anomalies: actual > yhat_upper (high-side outbreak signal)
results['is_anomaly_high'] = results['y'] > results['yhat_upper']
results['is_anomaly_low']  = results['y'] < results['yhat_lower']
results['is_anomaly']      = results['is_anomaly_high'] | results['is_anomaly_low']

anomalies_high = results[results['is_anomaly_high']]
anomalies_low  = results[results['is_anomaly_low']]

mae = np.mean(np.abs(results['y'] - results['yhat']))
print(f"  Prophet anomalies (high-side, actual > yhat_upper): {len(anomalies_high)}")
print(f"  Prophet anomalies (low-side):                       {len(anomalies_low)}")
print(f"  MAE on 2025 test period: {mae:.4f} cases/day")

# --------------------------------------------------------------------------
# 5. Compare against Z-score events
# --------------------------------------------------------------------------
print("\nStep 5: Comparing Prophet vs Z-score events...")
df_det = pd.read_pickle(os.path.join(data_dir, "test_detection_results.pkl"))
df_det['diagnosis_date'] = pd.to_datetime(df_det['diagnosis_date'])

zscore_series = df_det[
    (df_det['district'] == TARGET_DISTRICT) &
    (df_det['disease_name'] == TARGET_DISEASE)
].sort_values('diagnosis_date')

# Days flagged by Z-score engine (risk != Low)
zscore_flagged = set(
    zscore_series[zscore_series['risk_level'] != 'Low']['diagnosis_date'].dt.normalize()
)
# Days flagged as Confirmed/Watch-Tier Event
zscore_tier_days = set(
    zscore_series[zscore_series['tier'].isin(['Confirmed-Tier Event', 'Watch-Tier Event'])]
    ['diagnosis_date'].dt.normalize()
)

# Days flagged by Prophet (high-side only, for outbreak relevance)
prophet_flagged = set(anomalies_high['ds'].dt.normalize())

overlap      = prophet_flagged & zscore_flagged
prophet_only = prophet_flagged - zscore_flagged
zscore_only  = zscore_flagged - prophet_flagged

overlap_tier      = prophet_flagged & zscore_tier_days
prophet_only_tier = prophet_flagged - zscore_tier_days
zscore_only_tier  = zscore_tier_days - prophet_flagged

print(f"\n  Z-score alert days (risk >= Medium):  {len(zscore_flagged)}")
print(f"  Prophet anomaly days (high-side):     {len(prophet_flagged)}")
print(f"\n  --- Agreement vs any Z-score flag ---")
print(f"  Overlap (both agree):       {len(overlap)}")
print(f"  Prophet only:               {len(prophet_only)}")
print(f"  Z-score only:               {len(zscore_only)}")
print(f"\n  --- Agreement vs Confirmed/Watch-Tier Events ---")
print(f"  Overlap with Tier Events:   {len(overlap_tier)}")
print(f"  Prophet only:               {len(prophet_only_tier)}")
print(f"  Tier Events missed by Prophet: {len(zscore_only_tier)}")

if overlap_tier:
    print(f"\n  Overlapping dates:")
    for d in sorted(overlap_tier):
        print(f"    {d.date()}")

# --------------------------------------------------------------------------
# 6. Visualization
# --------------------------------------------------------------------------
print("\nStep 6: Generating visualization...")
fig, ax = plt.subplots(figsize=(16, 6))
fig.patch.set_facecolor('#0f1117')
ax.set_facecolor('#0f1117')

ax.fill_between(
    results['ds'], results['yhat_lower'], results['yhat_upper'],
    alpha=0.25, color='#4a9eff', label='95% Confidence Band'
)
ax.plot(results['ds'], results['yhat'], color='#4a9eff',
        linewidth=1.5, label='Prophet Forecast', alpha=0.85)
ax.plot(results['ds'], results['y'], color='#e8e8e8',
        linewidth=1.0, alpha=0.8, label='Actual Cases')

# Prophet high-side anomalies
if len(anomalies_high) > 0:
    ax.scatter(anomalies_high['ds'], anomalies_high['y'],
               color='#ff4444', zorder=5, s=60,
               label=f'Prophet Anomaly (n={len(anomalies_high)})', marker='^')

# Z-score tiered event days on this series
tier_days_df = zscore_series[zscore_series['tier'] == 'Confirmed-Tier Event']
if len(tier_days_df) > 0:
    ax.scatter(tier_days_df['diagnosis_date'], tier_days_df['case_count'],
               color='#ffd700', zorder=5, s=60, marker='D',
               label=f'Confirmed-Tier Event (Z-score, n={len(tier_days_df)})')

watch_days_df = zscore_series[zscore_series['tier'] == 'Watch-Tier Event']
if len(watch_days_df) > 0:
    ax.scatter(watch_days_df['diagnosis_date'], watch_days_df['case_count'],
               color='#ff9900', zorder=5, s=40, marker='o', alpha=0.8,
               label=f'Watch-Tier Event (Z-score, n={len(watch_days_df)})')

# Shade June Confirmed event period
june_start = pd.to_datetime('2025-06-05')
june_end   = pd.to_datetime('2025-06-09')
dec_start  = pd.to_datetime('2025-12-12')
dec_end    = pd.to_datetime('2025-12-18')
ax.axvspan(june_start, june_end, alpha=0.15, color='#ffd700', label='Confirmed Event Windows')
ax.axvspan(dec_start, dec_end, alpha=0.15, color='#ffd700')

ax.set_title(f'Prophet Forecast vs Z-Score Detection\n'
             f'{TARGET_DISTRICT} – {TARGET_DISEASE} | Test Period: 2025',
             color='white', fontsize=14, pad=15)
ax.set_xlabel('Date', color='#aaaaaa', fontsize=11)
ax.set_ylabel('Daily Case Count', color='#aaaaaa', fontsize=11)
ax.tick_params(colors='#aaaaaa')
for spine in ax.spines.values():
    spine.set_edgecolor('#333333')

legend = ax.legend(loc='upper right', fontsize=9, framealpha=0.3,
                   facecolor='#1a1a2e', edgecolor='#333333',
                   labelcolor='white')
ax.grid(True, alpha=0.15, color='#444444')

text_lines = [
    f"Prophet anomalies (high-side): {len(anomalies_high)}",
    f"Z-score tier events (days):    {len(zscore_tier_days)}",
    f"Overlap: {len(overlap_tier)} days | MAE: {mae:.4f}"
]
ax.text(0.01, 0.97, '\n'.join(text_lines), transform=ax.transAxes,
        fontsize=8, color='#aaaaaa', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.5))

plt.tight_layout()
plot_path = os.path.join(reports_dir, "prophet_vs_zscore_palakkad_chikungunya.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print(f"  Saved plot to: {plot_path}")

# --------------------------------------------------------------------------
# 7. Save outputs
# --------------------------------------------------------------------------
results_out = results[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper',
                        'is_anomaly_high', 'is_anomaly_low', 'is_anomaly']].copy()
results_out.columns = ['date', 'actual', 'predicted', 'lower_95', 'upper_95',
                       'anomaly_high', 'anomaly_low', 'is_anomaly']
results_out['date'] = results_out['date'].dt.strftime('%Y-%m-%d')
results_out = results_out.round({'predicted': 4, 'lower_95': 4, 'upper_95': 4})
csv_path = os.path.join(reports_dir, "prophet_predictions_palakkad_chikungunya.csv")
results_out.to_csv(csv_path, index=False)

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY: Prophet vs Z-Score Detection — Palakkad Chikungunya")
print("=" * 60)
print(f"  Prophet anomalies (high-side, actual > yhat_upper): {len(anomalies_high)}")
print(f"  Z-score Confirmed/Watch-Tier Event days:            {len(zscore_tier_days)}")
print(f"  Overlapping detections (both agree):                {len(overlap_tier)}")
print(f"  Prophet-only flags:                                 {len(prophet_only_tier)}")
print(f"  Z-score Tier Events missed by Prophet:              {len(zscore_only_tier)}")
print(f"  Test MAE (cases/day):                               {mae:.4f}")
print(f"\n  Outputs:")
print(f"    {csv_path}")
print(f"    {plot_path}")
print(f"    {model_path}")
print("=" * 60)
print("\nStage 3.7 complete.")
