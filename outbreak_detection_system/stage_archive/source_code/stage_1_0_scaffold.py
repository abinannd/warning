import os

base_dir = r"c:\BRAIN-STORM\HT\outbreak_detection_system"

dirs = [
    "data/raw", "data/interim", "data/processed",
    "src/preprocessing", "src/detection", "src/spatial", "src/visualization", "src/utils",
    "tests",
    "outputs/reports", "outputs/figures", "outputs/logs",
    "notebooks", "docs"
]

files = {
    "src/preprocessing/__init__.py": "",
    "src/preprocessing/loader.py": '"""\nModule for loading raw data into the pipeline.\n"""\n',
    "src/preprocessing/cleaner.py": '"""\nModule for cleaning data (handling missing values, outliers, etc.).\n"""\n',
    "src/preprocessing/aggregator.py": '"""\nModule for aggregating data at various temporal and spatial scales.\n"""\n',
    "src/preprocessing/calendar.py": '"""\nModule for handling calendar features, holidays, and temporal alignments.\n"""\n',
    "src/preprocessing/validator.py": '"""\nModule for validating data schema and integrity before processing.\n"""\n',
    "src/preprocessing/splitter.py": '"""\nModule for splitting data into training, validation, and test sets.\n"""\n',
    "src/detection/__init__.py": "",
    "src/detection/rolling_stats.py": '"""\nModule for calculating rolling statistics for time series.\n"""\n',
    "src/detection/ewma.py": '"""\nModule for Exponentially Weighted Moving Average calculations.\n"""\n',
    "src/detection/zscore_detector.py": '"""\nModule for detecting anomalies using rolling Z-scores.\n"""\n',
    "src/detection/anomaly_detector.py": '"""\nGeneral module for advanced anomaly detection algorithms.\n"""\n',
    "src/detection/risk_scorer.py": '"""\nModule for calculating and assigning risk scores to detected anomalies.\n"""\n',
    "src/spatial/__init__.py": "",
    "src/spatial/hotspot_detector.py": '"""\nModule for detecting spatial hotspots and clusters of outbreaks.\n"""\n',
    "src/visualization/__init__.py": "",
    "src/visualization/dashboard.py": '"""\nModule for setting up and running the interactive dashboard.\n"""\n',
    "src/visualization/plots.py": '"""\nModule for generating static and interactive plots.\n"""\n',
    "src/utils/__init__.py": "",
    "src/utils/config.py": '"""\nModule for loading and managing project configurations.\n"""\n',
    "src/utils/logger.py": '"""\nModule for setting up project logging.\n"""\n',
    "src/utils/helpers.py": '"""\nModule containing miscellaneous helper functions.\n"""\n',
    "tests/__init__.py": "",
    "tests/test_loader.py": '"""\nTests for the loader module.\n"""\n',
    "tests/test_cleaner.py": '"""\nTests for the cleaner module.\n"""\n',
    "tests/test_aggregator.py": '"""\nTests for the aggregator module.\n"""\n',
    "tests/test_calendar.py": '"""\nTests for the calendar module.\n"""\n',
    "tests/test_validator.py": '"""\nTests for the validator module.\n"""\n',
    "tests/test_splitter.py": '"""\nTests for the splitter module.\n"""\n',
    "main.py": '"""\nMain entry point for the Early Outbreak Detection System.\n"""\n',
    "requirements.txt": "# Project Dependencies\n# Add packages as needed\n",
    ".gitignore": "__pycache__/\n*.pyc\n.venv/\nvenv/\n.env\n.idea/\n.vscode/\noutputs/\ndata/interim/\ndata/processed/\nlogs/\n",
    "pyproject.toml": "[project]\nname = \"outbreak_detection_system\"\nversion = \"0.1.0\"\ndescription = \"AI-Based Early Outbreak Detection System\"\nauthors = [\n    { name = \"Author Name\", email = \"author@example.com\" }\n]\nrequires-python = \">=3.11\"\ndependencies = []\n\n[build-system]\nrequires = [\"setuptools>=61.0\"]\nbuild-backend = \"setuptools.build_meta\"\n",
    "README.md": "# AI-Based Early Outbreak Detection System\n\n## Project Overview\nThe project aims to detect infectious disease outbreaks using Rolling Statistics, EWMA (Exponentially Weighted Moving Average), Rolling Z-score Anomaly Detection, Risk Scoring, Spatial Hotspot Detection, and an Interactive Dashboard.\n\n## Folder Structure\n```text\noutbreak_detection_system/\n├── data/\n│   ├── raw/\n│   ├── interim/\n│   └── processed/\n├── src/\n│   ├── preprocessing/\n│   ├── detection/\n│   ├── spatial/\n│   ├── visualization/\n│   └── utils/\n├── tests/\n├── outputs/\n├── notebooks/\n├── docs/\n├── main.py\n├── requirements.txt\n├── README.md\n├── LICENSE\n├── .gitignore\n└── pyproject.toml\n```\n\n## Installation Instructions\n1. Clone the repository.\n2. Create a virtual environment using Python 3.11+:\n   ```bash\n   python -m venv venv\n   ```\n3. Activate the environment.\n4. Install dependencies:\n   ```bash\n   pip install -r requirements.txt\n   ```\n\n## Planned Pipeline\n1. **Data Preparation & Time Series Construction**: Loading, cleaning, validating, and aggregating data.\n2. **Anomaly Detection**: Applying rolling statistics, EWMA, and Z-scores to identify temporal anomalies.\n3. **Spatial Analysis**: Identifying geographical hotspots.\n4. **Risk Scoring**: Evaluating the severity of detected anomalies.\n5. **Visualization**: Presenting findings in an interactive dashboard.\n\n## Future Roadmap\n- Integration with real-time data feeds.\n- Advanced machine learning models for outbreak forecasting.\n- Automated alert generation and distribution.\n",
    "LICENSE": "MIT License\n\nCopyright (c) 2026\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\nof this software and associated documentation files (the \"Software\"), to deal\nin the Software without restriction, including without limitation the rights\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the Software is\nfurnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all\ncopies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\nIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\nFITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\nAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\nLIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\nOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\nSOFTWARE.\n"
}

for d in dirs:
    os.makedirs(os.path.join(base_dir, d), exist_ok=True)

for f, content in files.items():
    with open(os.path.join(base_dir, f), "w", encoding="utf-8") as file:
        file.write(content)

print("Scaffold created successfully.")
