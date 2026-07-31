# AI-Based Early Outbreak Detection System

## Project Overview
The project aims to detect infectious disease outbreaks using Rolling Statistics, EWMA (Exponentially Weighted Moving Average), Rolling Z-score Anomaly Detection, Risk Scoring, Spatial Hotspot Detection, and an Interactive Dashboard.

## Folder Structure
```text
outbreak_detection_system/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── src/
│   ├── preprocessing/
│   ├── detection/
│   ├── spatial/
│   ├── visualization/
│   └── utils/
├── tests/
├── outputs/
├── notebooks/
├── docs/
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
└── pyproject.toml
```

## Installation Instructions
1. Clone the repository.
2. Create a virtual environment using Python 3.11+:
   ```bash
   python -m venv venv
   ```
3. Activate the environment.
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Planned Pipeline
1. **Data Preparation & Time Series Construction**: Loading, cleaning, validating, and aggregating data.
2. **Anomaly Detection**: Applying rolling statistics, EWMA, and Z-scores to identify temporal anomalies.
3. **Spatial Analysis**: Identifying geographical hotspots.
4. **Risk Scoring**: Evaluating the severity of detected anomalies.
5. **Visualization**: Presenting findings in an interactive dashboard.

## Future Roadmap
- Integration with real-time data feeds.
- Advanced machine learning models for outbreak forecasting.
- Automated alert generation and distribution.
