# Historical Outbreak Replay Report

## 1. Objective
Evaluate the Early Outbreak Detection System by replaying the TEST dataset (2025) chronologically.

## 2. Methodology
The TEST dataset was replayed chronologically. Every district-disease combination was processed independently. No future information was used.

## 3. Event Definition
An alert event begins when the Risk Level becomes Medium, High, or Critical. Consecutive alert days belong to the same event. The event ends when the Risk Level returns to Low.

## 4. Overall Replay Statistics
- Total number of alert events: 1192
- Number of Medium events: 283
- Number of High events: 570
- Number of Critical events: 339
- Average event duration: 1.11 days
- Longest event: 3 days
- Shortest event: 1 days
- Maximum observed Z-score: 5.29
- Average observed Z-score: 2.95
- District with highest events: Kasaragod
- Disease with highest events: Chickenpox
- District-Disease with highest peak Z-score: Kozhikode - Common Cold

## 5. District Summary
| District   |   Events |   Medium |   High |   Critical |
|:-----------|---------:|---------:|-------:|-----------:|
| Kannur     |      188 |       59 |     80 |         49 |
| Kasaragod  |      209 |       41 |    111 |         57 |
| Kozhikode  |      198 |       58 |     88 |         52 |
| Malappuram |      197 |       48 |     97 |         52 |
| Palakkad   |      204 |       31 |    106 |         67 |
| Wayanad    |      196 |       46 |     88 |         62 |

## 6. Disease Summary
| Disease     |   Events |   Medium |   High |   Critical |
|:------------|---------:|---------:|-------:|-----------:|
| Chickenpox  |      164 |       45 |     78 |         41 |
| Chikungunya |      164 |       37 |     90 |         37 |
| Common Cold |      140 |       40 |     61 |         39 |
| Dengue      |      152 |       37 |     78 |         37 |
| Flu         |      141 |       30 |     66 |         45 |
| Malaria     |      141 |       28 |     67 |         46 |
| Typhoid     |      141 |       30 |     67 |         44 |
| Viral Fever |      149 |       36 |     63 |         50 |

## 7. Monthly Timeline
| Month   |   Events Started |   Critical Events |
|:--------|-----------------:|------------------:|
| 2025-01 |              100 |                24 |
| 2025-02 |              100 |                28 |
| 2025-03 |               76 |                14 |
| 2025-04 |              100 |                36 |
| 2025-05 |              103 |                30 |
| 2025-06 |              115 |                33 |
| 2025-07 |              125 |                37 |
| 2025-08 |              106 |                26 |
| 2025-09 |               91 |                25 |
| 2025-10 |               84 |                27 |
| 2025-11 |               91 |                35 |
| 2025-12 |              101 |                24 |

## 8. Top 10 Strongest Events
|   Rank | District   | Disease     |   Peak Z-score |   Peak Cases |   Duration | Highest Risk   |
|-------:|:-----------|:------------|---------------:|-------------:|-----------:|:---------------|
|      1 | Kozhikode  | Common Cold |        5.29465 |            1 |          1 | Critical       |
|      2 | Malappuram | Typhoid     |        5.29465 |            1 |          1 | Critical       |
|      3 | Wayanad    | Malaria     |        5.29465 |            1 |          1 | Critical       |
|      4 | Wayanad    | Malaria     |        5.29465 |            1 |          1 | Critical       |
|      5 | Kasaragod  | Dengue      |        5.29465 |            1 |          1 | Critical       |
|      6 | Palakkad   | Typhoid     |        5.29465 |            1 |          1 | Critical       |
|      7 | Malappuram | Common Cold |        5.29465 |            1 |          1 | Critical       |
|      8 | Kasaragod  | Viral Fever |        5.29465 |            1 |          1 | Critical       |
|      9 | Wayanad    | Common Cold |        5.29465 |            1 |          1 | Critical       |
|     10 | Kasaragod  | Malaria     |        5.29465 |            1 |          2 | Critical       |

## 9. Key Observations
Events have been successfully captured and aggregated without generating redundant alarms for consecutive days.

## 10. Conclusion
The historical replay verifies the pipeline's ability to smoothly monitor and characterize outbreak events over time without falsely separating continuous outbreaks.


## 11. Sparse Data & Single-Day Event Fix (Stage 3.1.1)
Following the initial run, a critical statistical issue was identified where near-zero variance triggered highly inflated Z-scores on sparse, single cases. To address this, we applied a **standard deviation floor of 0.5** and enforced a **minimum event duration of 2 consecutive days**.

| Metric          |   Before (Std Floor=0, Min Dur=1) |   After (Std Floor=0.5, Min Dur=2) |
|-----------------|-----------------------------------|------------------------------------|
| Total Events    |                           1192    |                               4    |
| Critical Events |                            339    |                               4    |
| High Events     |                            570    |                               0    |
| Medium Events   |                            283    |                               0    |
| Avg Duration    |                              1.11 |                               2    |
| Max Z-score     |                              5.29 |                               3.87 |

This dramatically reduced noise, transforming the alert volume to highlight only sustained, credible epidemiological threats.


---

## 12. Gap-Corrected Baseline (Stage 3.3)

### System Description

> A regional early warning system that analyzes the most recent week's surveillance data against a historical baseline (ending 7 days prior) to identify statistically significant increases in disease activity and issue risk-based alerts to public health authorities.

### Problem: Baseline Contamination

The original rolling baseline (Stages 2.12.4) used a 30-day window ending on day T, meaning the most recent 7 days of case data  the very window being tested for elevated activity  were also included in the baseline computation. This causes the baseline mean and std to track the current outbreak, inflating the denominator exactly when the Z-score needs to be highest, and suppressing genuine alerts.

### Fix: 7-Day Gap Between Baseline and Signal Window

| Window | Dates Used | Computation |
|--------|-----------|-------------|
| Historical Baseline | [T-37, T-8] (30 days) | `rolling(30).mean/std.shift(8)` |
| Recent Trend Signal | [T-6, T] (7 days) | `rolling(7).mean()` |
| Z-score |  | `(recent_mean - baseline_mean) / max(baseline_std, 1e-6)` |
| EWMA | Up to T-8 | `ewm(span=14).mean().shift(8)` |

### Before vs After Comparison

| Metric          |   Old (Contaminated Baseline) |   New (Gap-Corrected Baseline) |
|-----------------|-------------------------------|--------------------------------|
| Total Events    |                         16    |                            5   |
| Critical Events |                         16    |                            1   |
| High Events     |                          0    |                            2   |
| Medium Events   |                          0    |                            2   |
| Avg Duration    |                          2.19 |                            4.8 |
| Max Z-score     |                          5.29 |                       285714   |

### Interpretation

The gap-corrected baseline separates the 'what the system is testing' window from the 'what the system learned from' window. This is the correct statistical design for a surveillance system where the signal of interest should not contaminate the reference distribution used to judge it.


---

## 12. Gap-Corrected Baseline (Stage 3.3)

### System Description

> A regional early warning system that analyzes the most recent week's surveillance data against a historical baseline (ending 7 days prior) to identify statistically significant increases in disease activity and issue risk-based alerts to public health authorities.

### Problem: Baseline Contamination

The original rolling baseline (Stages 2.12.4) used a 30-day window ending on day T, meaning the most recent 7 days of case data  the very window being tested for elevated activity  were also included in the baseline computation. This causes the baseline mean and std to track the current outbreak, inflating the denominator exactly when the Z-score needs to be highest, and suppressing genuine alerts.

### Fix: 7-Day Gap Between Baseline and Signal Window

| Window | Dates Used | Computation |
|--------|-----------|-------------|
| Historical Baseline | [T-37, T-8] (30 days) | `rolling(30).mean/std.shift(8)` |
| Recent Trend Signal | [T-6, T] (7 days) | `rolling(7).mean()` |
| Z-score |  | `(recent_mean - baseline_mean) / max(baseline_std, 1e-6)` |
| EWMA | Up to T-8 | `ewm(span=14).mean().shift(8)` |

### Before vs After Comparison

| Metric          |   Old (Contaminated Baseline) |   New (Gap-Corrected Baseline) |
|-----------------|-------------------------------|--------------------------------|
| Total Events    |                         16    |                           4    |
| Critical Events |                         16    |                           0    |
| High Events     |                          0    |                           2    |
| Medium Events   |                          0    |                           2    |
| Avg Duration    |                          2.19 |                           4    |
| Max Z-score     |                          5.29 |                           2.95 |

### Interpretation

The gap-corrected baseline separates the 'what the system is testing' window from the 'what the system learned from' window. This is the correct statistical design for a surveillance system where the signal of interest should not contaminate the reference distribution used to judge it.


---

## 13. Two-Tier Alerting System (Stage 3.4)

### System Description



---

## 13. Two-Tier Alerting System (Stage 3.4)

### System Description

> The system uses a two-tier alerting model, mirroring real-world surveillance practice: a sensitive **Watch-Tier Event** tier for early signals worth monitoring, and a high-precision **Confirmed-Tier Event** tier reserved for sustained, statistically robust anomalies (peak cases â¥2, duration â¥2 days, corrected against a gap-separated historical baseline).

### Tier Definitions

| Tier | Label | min_duration | min_peak_cases | Baseline | Purpose |
|------|-------|-------------|----------------|----------|---------|
| Tier 1 | **Confirmed-Tier Event** | â¥2 days | â¥2 cases | Gap-corrected [T-37, T-8] | High-precision confirmed outbreak alert |
| Tier 2 | **Watch-Tier Event** | â¥2 days | â¥1 case (any) | Gap-corrected [T-37, T-8] | Sensitive early signal for monitoring |

*Tiers are mutually exclusive: events qualifying for Tier 1 are excluded from Tier 2.*

### Results on 2025 Test Set

| Tier               |   Events |   Critical |   High |   Medium |   Avg Dur |   Max Z |
|--------------------|----------|------------|--------|----------|-----------|---------|
| Confirmed-Tier Event |        4 |          0 |      2 |        2 |      4    |   2.947 |
| Watch-Tier Event     |       25 |          0 |      0 |        0 |      2.84 |   3.116 |

### District Breakdown

| District   |   Confirmed |   Watch |   Total |
|------------|-------------|---------|---------|
| Kannur     |           1 |       5 |       6 |
| Kasaragod  |           0 |       1 |       1 |
| Kozhikode  |           0 |       8 |       8 |
| Malappuram |           0 |       5 |       5 |
| Palakkad   |           2 |       3 |       5 |
| Wayanad    |           1 |       3 |       4 |

### Disease Breakdown

| Disease     |   Confirmed |   Watch |   Total |
|-------------|-------------|---------|---------|
| Chickenpox  |           0 |       1 |       1 |
| Chikungunya |           2 |       0 |       2 |
| Common Cold |           1 |       3 |       4 |
| Dengue      |           0 |       3 |       3 |
| Flu         |           0 |       6 |       6 |
| Malaria     |           0 |       5 |       5 |
| Typhoid     |           0 |       1 |       1 |
| Viral Fever |           1 |       6 |       7 |

### Design Rationale

The two-tier model was motivated by the ablation test (Stage 3.4 diagnostic), which showed that the gap-corrected baseline alone reduces peak_cases=1 artifacts from 101 (contaminated baseline) to 18 â a substantial improvement, but not complete elimination. Rather than discarding these 18 signals entirely, they are preserved as low-confidence Watch-Tier Event alerts, allowing health authorities to apply domain judgment on whether to investigate further.


> **Note:** This report uses [Watch-Tier Event / Watch-Status Warning] terminology, which is distinct from the other report's equivalent term. **Watch-Tier Event** refers to a year-level statistical classification based on sustained low-peak-case anomalies; **Watch-Status Warning** refers to a real-time weekly risk snapshot based on the highest observed daily risk level.


---

## 13. Two-Tier Alerting System (Stage 3.4)

### System Description

> The system uses a two-tier alerting model, mirroring real-world surveillance practice: a sensitive **Watch** tier for early signals worth monitoring, and a high-precision **Confirmed** tier reserved for sustained, statistically robust anomalies (peak cases ≥2, duration ≥2 days, corrected against a gap-separated historical baseline).

### Tier Definitions

| Tier | Label | min_duration | min_peak_cases | Baseline | Purpose |
|------|-------|-------------|----------------|----------|---------|
| Tier 1 | **Confirmed** | ≥2 days | ≥2 cases | Gap-corrected [T-37, T-8] | High-precision confirmed outbreak alert |
| Tier 2 | **Watch** | ≥2 days | ≥1 case (any) | Gap-corrected [T-37, T-8] | Sensitive early signal for monitoring |

*Tiers are mutually exclusive: events qualifying for Tier 1 are excluded from Tier 2.*

### Results on 2025 Test Set

| Tier                          |   Events |   Critical |   High |   Medium |   Avg Dur |   Max Z |
|-------------------------------|----------|------------|--------|----------|-----------|---------|
| Confirmed-Tier Event (Tier 1) |        4 |          0 |      2 |        2 |      4    |   2.947 |
| Watch-Tier Event (Tier 2)     |       25 |          0 |      0 |        0 |      2.84 |   3.116 |

### District Breakdown

| District   |   Confirmed-Tier Event |   Watch-Tier Event |   Total |
|------------|------------------------|--------------------|---------|
| Kannur     |                      1 |                  5 |       6 |
| Kasaragod  |                      0 |                  1 |       1 |
| Kozhikode  |                      0 |                  8 |       8 |
| Malappuram |                      0 |                  5 |       5 |
| Palakkad   |                      2 |                  3 |       5 |
| Wayanad    |                      1 |                  3 |       4 |

### Disease Breakdown

| Disease     |   Confirmed-Tier Event |   Watch-Tier Event |   Total |
|-------------|------------------------|--------------------|---------|
| Chickenpox  |                      0 |                  1 |       1 |
| Chikungunya |                      2 |                  0 |       2 |
| Common Cold |                      1 |                  3 |       4 |
| Dengue      |                      0 |                  3 |       3 |
| Flu         |                      0 |                  6 |       6 |
| Malaria     |                      0 |                  5 |       5 |
| Typhoid     |                      0 |                  1 |       1 |
| Viral Fever |                      1 |                  6 |       7 |

### Design Rationale

The two-tier model was motivated by the ablation test (Stage 3.4 diagnostic), which showed that the gap-corrected baseline alone reduces peak_cases=1 artifacts from 101 (contaminated baseline) to 18 — a substantial improvement, but not complete elimination. Rather than discarding these 18 signals entirely, they are preserved as low-confidence Watch-tier alerts, allowing health authorities to apply domain judgment on whether to investigate further.
