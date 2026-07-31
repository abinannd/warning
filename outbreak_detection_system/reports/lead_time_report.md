# Lead-Time Analysis Report

## 1. Objective
Evaluate how early the Early Outbreak Detection System identifies disease activity before each outbreak reaches its peak daily case count.

## 2. Methodology
The TEST dataset (2025) was processed through the tuned detection pipeline (min_duration=2, peak_cases>=2, std floor=0.0 with epsilon=1e-6). For each identified event, the detection date is the first alert day and the peak date is the day with the highest case count within that event window.

## 3. Lead-Time Definition
```
Lead Time = Peak Date - Detection Date  (in days)
```

## 4. Overall Statistics
| Metric                  | Value     |
|-------------------------|-----------|
| Total Events            | 16        |
| Average Lead Time       | 0.94 days |
| Median Lead Time        | 1.0 days  |
| Min Lead Time           | 0 days    |
| Max Lead Time           | 2 days    |
| Std Deviation           | 0.68 days |
| No Early Warning        | 4         |
| Short Warning (1-3d)    | 12        |
| Moderate Warning (4-7d) | 0         |
| Long Warning (>7d)      | 0         |

## 5. Distribution Analysis
| Category         |   Count | Percentage   |
|------------------|---------|--------------|
| No Early Warning |       4 | 25.0%        |
| Short Warning    |      12 | 75.0%        |
| Moderate Warning |       0 | 0.0%         |
| Long Warning     |       0 | 0.0%         |

## 6. District Analysis
| District   |   Events |   Average Lead Time |   Max Lead Time |
|------------|----------|---------------------|-----------------|
| Kannur     |        7 |                0.86 |               2 |
| Kasaragod  |        2 |                1    |               1 |
| Kozhikode  |        1 |                1    |               1 |
| Malappuram |        2 |                0.5  |               1 |
| Palakkad   |        2 |                1    |               1 |
| Wayanad    |        2 |                1.5  |               2 |

## 7. Disease Analysis
| Disease     |   Events |   Average Lead Time |   Max Lead Time |
|-------------|----------|---------------------|-----------------|
| Chickenpox  |        2 |                0.5  |               1 |
| Chikungunya |        1 |                1    |               1 |
| Common Cold |        2 |                0    |               0 |
| Flu         |        3 |                1.33 |               2 |
| Malaria     |        4 |                1.25 |               2 |
| Typhoid     |        1 |                1    |               1 |
| Viral Fever |        3 |                1    |               2 |

## 8. Top 10 Earliest Detections
|   Rank | District   | Disease     |   Lead Time |   Peak Cases | Highest Risk   |
|--------|------------|-------------|-------------|--------------|----------------|
|      1 | Kannur     | Malaria     |           2 |            2 | Critical       |
|      2 | Kannur     | Viral Fever |           2 |            2 | Critical       |
|      3 | Wayanad    | Flu         |           2 |            2 | Critical       |
|      4 | Kannur     | Typhoid     |           1 |            2 | Critical       |
|      5 | Kannur     | Viral Fever |           1 |            2 | Critical       |
|      6 | Kasaragod  | Flu         |           1 |            2 | Critical       |
|      7 | Kasaragod  | Malaria     |           1 |            2 | Critical       |
|      8 | Kozhikode  | Chikungunya |           1 |            2 | Critical       |
|      9 | Malappuram | Malaria     |           1 |            2 | Critical       |
|     10 | Palakkad   | Chickenpox  |           1 |            2 | Critical       |

## 9. Interpretation
The system detected 16 credible outbreak events in 2025. The average lead time of 0.94 days indicates that alerts were typically raised at or near the peak day, which is consistent with a reactive (not predictive) statistical baseline. Events with lead times > 0 represent cases where the initial alert day preceded the highest case count, giving at least some advance notice.

## 10. Conclusion
The rolling Z-score pipeline, with calibrated filters, produces credible alerts. Lead-time analysis shows the system responds to sustained statistical anomalies, with a distribution skewed toward short or zero lead times — typical for reactive statistical surveillance methods operating on sparse count data.
