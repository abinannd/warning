"""
Stage 3.7.1 - Prophet Model Evaluation & Diagnostics
Target: Palakkad - Chikungunya (read-only evaluation, no model changes)

Steps:
  1. Point-accuracy metrics (MAE, RMSE, MAPE, sMAPE, MedianAE, R²)
  2. Variance / uncertainty calibration (coverage, interval width, residual variance)
  3. Residual diagnostics (time plot, histogram, bias)
  4. Prophet cross-validation (multi-window accuracy by horizon)
  5. Naive baseline comparison (30-day rolling mean)
  6. Output: JSON report, diagnostic plots, summary table
"""

import os
import json
import pickle
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
from tabulate import tabulate

warnings.filterwarnings('ignore')

base_dir    = r"C:\BRAIN-STORM\HT\warning\outbreak_detection_system"
data_dir    = os.path.join(base_dir, "data", "processed")
reports_dir = os.path.join(base_dir, "reports")
models_dir  = os.path.join(base_dir, "models")
diag_dir    = os.path.join(reports_dir, "prophet_diagnostics")
os.makedirs(diag_dir, exist_ok=True)

TARGET_DISTRICT = "Palakkad"
TARGET_DISEASE  = "Chikungunya"

# --------------------------------------------------------------------------
# Load model, data, and predictions (READ-ONLY — no modifications)
# --------------------------------------------------------------------------
print("Loading saved model and data...")
model_path = os.path.join(models_dir, "prophet_palakkad_chikungunya.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load series
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

prophet_df = series[['diagnosis_date', 'case_count']].rename(
    columns={'diagnosis_date': 'ds', 'case_count': 'y'}
)
train_df = prophet_df[prophet_df['ds'] < '2025-01-01'].copy()
test_df  = prophet_df[prophet_df['ds'] >= '2025-01-01'].copy()

# Generate predictions on test
future = model.predict(test_df[['ds']])
results = test_df.merge(
    future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
    on='ds', how='left'
)
results['yhat_lower'] = results['yhat_lower'].clip(lower=0)
results['yhat']       = results['yhat'].clip(lower=0)
results['residual']   = results['y'] - results['yhat']

actual    = results['y'].values
predicted = results['yhat'].values
residuals = results['residual'].values
n = len(results)

print(f"  Test period: {results['ds'].min().date()} to {results['ds'].max().date()} ({n} days)")

# ==========================================================================
# STEP 1: Point-accuracy metrics
# ==========================================================================
print("\n" + "=" * 60)
print("STEP 1: Point-Accuracy Metrics")
print("=" * 60)

mae       = np.mean(np.abs(residuals))
rmse      = np.sqrt(np.mean(residuals ** 2))
median_ae = np.median(np.abs(residuals))
r2        = 1 - (np.sum(residuals ** 2) / np.sum((actual - np.mean(actual)) ** 2))

# MAPE: only on days where actual > 0
nonzero_mask = actual > 0
if nonzero_mask.sum() > 0:
    mape = np.mean(np.abs(residuals[nonzero_mask]) / actual[nonzero_mask]) * 100
else:
    mape = float('nan')
n_zero_days = int((~nonzero_mask).sum())

# sMAPE (symmetric MAPE): defined even when actual=0
denom = (np.abs(actual) + np.abs(predicted)) / 2
denom_safe = np.where(denom == 0, 1, denom)
smape = np.mean(np.abs(residuals) / denom_safe) * 100

metrics_1 = {
    "MAE":          round(mae, 4),
    "RMSE":         round(rmse, 4),
    "Median AE":    round(median_ae, 4),
    "R²":           round(r2, 4),
    "MAPE (%)":     round(mape, 2),
    "sMAPE (%)":    round(smape, 2),
    "Zero-actual days (excluded from MAPE)": n_zero_days,
    "Non-zero days (used for MAPE)":         int(nonzero_mask.sum())
}

for k, v in metrics_1.items():
    print(f"  {k}: {v}")

# ==========================================================================
# STEP 2: Variance / Uncertainty Calibration
# ==========================================================================
print("\n" + "=" * 60)
print("STEP 2: Uncertainty Calibration")
print("=" * 60)

inside_ci = ((actual >= results['yhat_lower'].values) &
             (actual <= results['yhat_upper'].values))
coverage_rate = inside_ci.mean() * 100
n_inside      = int(inside_ci.sum())

interval_widths = results['yhat_upper'].values - results['yhat_lower'].values
avg_interval    = np.mean(interval_widths)
median_interval = np.median(interval_widths)

residual_var = np.var(residuals)
residual_std = np.std(residuals)
mean_bias    = np.mean(residuals)

# Check heteroscedasticity: split test into halves, compare variance
half = n // 2
var_first  = np.var(residuals[:half])
var_second = np.var(residuals[half:])
variance_ratio = var_second / var_first if var_first > 0 else float('inf')
# Ratio near 1 => homoscedastic; far from 1 => heteroscedastic
if 0.5 <= variance_ratio <= 2.0:
    homoscedasticity = "Roughly homoscedastic (variance ratio = {:.2f})".format(variance_ratio)
else:
    homoscedasticity = "Heteroscedastic (variance ratio = {:.2f})".format(variance_ratio)

if coverage_rate > 98:
    coverage_verdict = "UNDERCONFIDENT (model too wide — coverage >> 95%)"
elif coverage_rate < 90:
    coverage_verdict = "OVERCONFIDENT (model too narrow — coverage << 95%)"
else:
    coverage_verdict = "WELL CALIBRATED (close to target 95%)"

metrics_2 = {
    "Coverage rate (%)":        round(coverage_rate, 2),
    "Days inside 95% CI":       f"{n_inside}/{n}",
    "Coverage verdict":         coverage_verdict,
    "Avg interval width":       round(avg_interval, 4),
    "Median interval width":    round(median_interval, 4),
    "Residual variance":        round(residual_var, 4),
    "Residual std":             round(residual_std, 4),
    "Mean bias":                round(mean_bias, 4),
    "Homoscedasticity":         homoscedasticity
}

for k, v in metrics_2.items():
    print(f"  {k}: {v}")

# ==========================================================================
# STEP 3: Residual Diagnostics — Plots
# ==========================================================================
print("\n" + "=" * 60)
print("STEP 3: Residual Diagnostics")
print("=" * 60)

# --- Plot 1: Residuals over time ---
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.patch.set_facecolor('#0f1117')

ax1 = axes[0]
ax1.set_facecolor('#0f1117')
ax1.bar(results['ds'], residuals, color='#4a9eff', alpha=0.7, width=1)
ax1.axhline(0, color='#ff4444', linewidth=1, linestyle='--', alpha=0.7)
ax1.axhline(mean_bias, color='#ffd700', linewidth=1, linestyle='--', alpha=0.7,
            label=f'Mean bias = {mean_bias:.4f}')
ax1.set_title('Residuals Over Time (Actual − Predicted)', color='white', fontsize=13)
ax1.set_ylabel('Residual', color='#aaa', fontsize=11)
ax1.tick_params(colors='#aaa')
for s in ax1.spines.values(): s.set_edgecolor('#333')
ax1.legend(fontsize=9, facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
ax1.grid(True, alpha=0.15, color='#444')

# --- Plot 2: Histogram of residuals ---
ax2 = axes[1]
ax2.set_facecolor('#0f1117')
ax2.hist(residuals, bins=30, color='#4a9eff', alpha=0.8, edgecolor='#333')
ax2.axvline(0, color='#ff4444', linewidth=1.5, linestyle='--', label='Zero')
ax2.axvline(mean_bias, color='#ffd700', linewidth=1.5, linestyle='--',
            label=f'Mean = {mean_bias:.4f}')
ax2.set_title('Residual Distribution', color='white', fontsize=13)
ax2.set_xlabel('Residual (Actual − Predicted)', color='#aaa', fontsize=11)
ax2.set_ylabel('Frequency', color='#aaa', fontsize=11)
ax2.tick_params(colors='#aaa')
for s in ax2.spines.values(): s.set_edgecolor('#333')
ax2.legend(fontsize=9, facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
ax2.grid(True, alpha=0.15, color='#444')

plt.tight_layout()
resid_path = os.path.join(diag_dir, "residual_diagnostics.png")
plt.savefig(resid_path, dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print(f"  Saved: {resid_path}")

skewness = float(pd.Series(residuals).skew())
kurtosis = float(pd.Series(residuals).kurtosis())
print(f"  Residual skewness: {skewness:.4f} (0 = symmetric)")
print(f"  Residual kurtosis: {kurtosis:.4f} (0 = normal)")

# ==========================================================================
# STEP 4: Prophet Cross-Validation
# ==========================================================================
print("\n" + "=" * 60)
print("STEP 4: Prophet Cross-Validation (multi-window)")
print("=" * 60)

print("  Running cross_validation (initial=1095d, horizon=30d, period=180d)...")
print("  This may take a moment...")

cv_results = cross_validation(
    model,
    initial='1095 days',   # ~3 years of training before first cutoff
    horizon='30 days',     # predict 30 days ahead
    period='180 days'      # make a cut every 6 months
)

pm = performance_metrics(cv_results)
print(f"  Cross-validation complete: {len(cv_results)} predictions across multiple cutoffs")
print(f"\n  Performance metrics by horizon (first 5 rows):")
cols_to_show = [c for c in ['horizon', 'mae', 'rmse', 'mape', 'coverage'] if c in pm.columns]
pm_display = pm[cols_to_show].head(10).round(4)
print(tabulate(pm_display, headers='keys', showindex=False, tablefmt='github'))

# Average across all horizons
cv_avg_mae  = pm['mae'].mean()
cv_avg_rmse = pm['rmse'].mean()
cv_avg_mape = pm['mape'].mean() * 100 if 'mape' in pm.columns else float('nan')
cv_avg_cov  = pm['coverage'].mean() * 100 if 'coverage' in pm.columns else float('nan')
print(f"\n  CV Average MAE:      {cv_avg_mae:.4f}")
print(f"  CV Average RMSE:     {cv_avg_rmse:.4f}")
print(f"  CV Average Coverage: {cv_avg_cov:.2f}%")

# --- CV metric plot ---
fig_cv = plot_cross_validation_metric(cv_results, metric='mae')
fig_cv.patch.set_facecolor('#0f1117')
ax_cv = fig_cv.gca()
ax_cv.set_facecolor('#0f1117')
ax_cv.set_title('Cross-Validation MAE by Horizon', color='white', fontsize=13)
ax_cv.tick_params(colors='#aaa')
for s in ax_cv.spines.values(): s.set_edgecolor('#333')
cv_plot_path = os.path.join(diag_dir, "cv_mae_by_horizon.png")
fig_cv.savefig(cv_plot_path, dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close(fig_cv)
print(f"  Saved: {cv_plot_path}")

# ==========================================================================
# STEP 5: Naive Baseline Comparison
# ==========================================================================
print("\n" + "=" * 60)
print("STEP 5: Naive Baseline Comparison (30-day rolling mean)")
print("=" * 60)

# Full series (including train + val) for rolling computation
full_series = prophet_df.copy().sort_values('ds')
full_series['naive_pred'] = full_series['y'].rolling(30, min_periods=1).mean().shift(1)
full_series['naive_pred'] = full_series['naive_pred'].fillna(0)

naive_test = full_series[full_series['ds'] >= '2025-01-01'].copy()
naive_actual    = naive_test['y'].values
naive_predicted = naive_test['naive_pred'].values
naive_residuals = naive_actual - naive_predicted

naive_mae  = np.mean(np.abs(naive_residuals))
naive_rmse = np.sqrt(np.mean(naive_residuals ** 2))

prophet_vs_naive_mae  = ((naive_mae - mae) / naive_mae) * 100 if naive_mae > 0 else 0
prophet_vs_naive_rmse = ((naive_rmse - rmse) / naive_rmse) * 100 if naive_rmse > 0 else 0

metrics_5 = {
    "Naive MAE":  round(naive_mae, 4),
    "Naive RMSE": round(naive_rmse, 4),
    "Prophet improvement over naive (MAE %)":  round(prophet_vs_naive_mae, 2),
    "Prophet improvement over naive (RMSE %)": round(prophet_vs_naive_rmse, 2)
}

for k, v in metrics_5.items():
    print(f"  {k}: {v}")

# Verdict
if prophet_vs_naive_mae > 5:
    verdict = "Prophet OUTPERFORMS the naive 30-day rolling mean baseline."
elif prophet_vs_naive_mae < -5:
    verdict = "Prophet UNDERPERFORMS the naive baseline — simpler method wins."
else:
    verdict = "Prophet and naive baseline are roughly EQUIVALENT in accuracy."

print(f"\n  Verdict: {verdict}")

# ==========================================================================
# STEP 6: Final Summary Table + JSON Output
# ==========================================================================
print("\n" + "=" * 60)
print("STEP 6: Summary Comparison Table")
print("=" * 60)

summary_table = [
    ["MAE",      round(mae, 4),  round(naive_mae, 4),  f"{prophet_vs_naive_mae:+.1f}%"],
    ["RMSE",     round(rmse, 4), round(naive_rmse, 4), f"{prophet_vs_naive_rmse:+.1f}%"],
    ["Median AE", round(median_ae, 4), "—", "—"],
    ["MAPE (%)", f"{mape:.2f}", "—", "—"],
    ["sMAPE (%)", f"{smape:.2f}", "—", "—"],
    ["R²",       f"{r2:.4f}", "—", "—"],
    ["Coverage (95% CI)", f"{coverage_rate:.1f}%", "—", coverage_verdict],
]
print(tabulate(summary_table,
    headers=["Metric", "Prophet", "Naive Baseline", "Diff / Verdict"],
    tablefmt="github"))
print(f"\n  {verdict}")

# --- Save JSON ---
full_report = {
    "target_series": f"{TARGET_DISTRICT} - {TARGET_DISEASE}",
    "test_period": f"{results['ds'].min().date()} to {results['ds'].max().date()}",
    "test_days": n,
    "step_1_point_accuracy": metrics_1,
    "step_2_uncertainty_calibration": {
        "coverage_rate_pct":     round(coverage_rate, 2),
        "days_inside_ci":        n_inside,
        "total_days":            n,
        "coverage_verdict":      coverage_verdict,
        "avg_interval_width":    round(avg_interval, 4),
        "median_interval_width": round(median_interval, 4),
        "residual_variance":     round(residual_var, 4),
        "residual_std":          round(residual_std, 4),
        "mean_bias":             round(mean_bias, 4),
        "homoscedasticity":      homoscedasticity
    },
    "step_3_residual_diagnostics": {
        "skewness":  round(skewness, 4),
        "kurtosis":  round(kurtosis, 4),
        "mean_bias": round(mean_bias, 4)
    },
    "step_4_cross_validation": {
        "cv_avg_mae":      round(cv_avg_mae, 4),
        "cv_avg_rmse":     round(cv_avg_rmse, 4),
        "cv_avg_coverage_pct": round(cv_avg_cov, 2)
    },
    "step_5_naive_baseline": {
        "naive_mae":  round(naive_mae, 4),
        "naive_rmse": round(naive_rmse, 4),
        "prophet_improvement_mae_pct":  round(prophet_vs_naive_mae, 2),
        "prophet_improvement_rmse_pct": round(prophet_vs_naive_rmse, 2),
        "verdict": verdict
    }
}

json_path = os.path.join(reports_dir, "prophet_accuracy_report.json")
with open(json_path, "w") as f:
    json.dump(full_report, f, indent=4, default=str)
print(f"\nSaved: {json_path}")

print(f"\nAll diagnostic plots saved to: {diag_dir}/")
print("Stage 3.7.1 Prophet Evaluation complete (read-only — no model modified).")
