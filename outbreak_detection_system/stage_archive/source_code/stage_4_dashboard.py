"""
Stage 4 Dashboard Builder
Generates a self-contained HTML dashboard for early outbreak detection.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import timedelta

base_dir = r"C:\BRAIN-STORM\HT\warning\outbreak_detection_system"
data_dir = os.path.join(base_dir, "data", "processed")
reports_dir = os.path.join(base_dir, "reports")
out_dir = os.path.join(base_dir, "outputs")
os.makedirs(out_dir, exist_ok=True)

# -------------------------------------------------------------------------
# 1. Load Data
# -------------------------------------------------------------------------
print("Loading data...")
# Test detection results contains daily gap-corrected z-scores and tier flags
df_test = pd.read_pickle(os.path.join(data_dir, "test_detection_results.pkl"))
df_test['diagnosis_date'] = pd.to_datetime(df_test['diagnosis_date'])
df_test['week'] = df_test['diagnosis_date'].dt.isocalendar().week

# -------------------------------------------------------------------------
# 2. Build Weekly Warning Table
# -------------------------------------------------------------------------
print("Building weekly warning table...")
# Group by Week, District, Disease
# Priorities: Emergency (4) > Watch (3) > Advisory (2) > Normal (1)

def get_status(grp):
    if (grp['tier'] == 'Confirmed-Tier Event').any():
        return 4, 'Emergency Warning'
    elif (grp['tier'] == 'Watch-Tier Event').any():
        return 3, 'Watch-Status Warning'
    elif (grp['risk_level'] != 'Low').any():
        return 2, 'Advisory'
    else:
        return 1, 'Normal'

weekly_warnings = {}
weeks = sorted(df_test['week'].unique())
districts = sorted(df_test['district'].unique())

for w in weeks:
    week_str = f"Week {w}"
    weekly_warnings[week_str] = {}
    week_data = df_test[df_test['week'] == w]
    
    for dist in districts:
        dist_data = week_data[week_data['district'] == dist]
        highest_prio = 0
        best_status = "Normal"
        best_dis = "None"
        
        for dis, grp in dist_data.groupby('disease_name'):
            p, s = get_status(grp)
            if p > highest_prio:
                highest_prio = p
                best_status = s
                best_dis = dis
                
        # Color mapping
        color = "green"
        if highest_prio == 4: color = "red"
        elif highest_prio >= 2: color = "yellow"
        
        # Recommendations
        rec = "Routine surveillance."
        if highest_prio == 4:
            rec = "Immediate public health intervention recommended. Deploy rapid response team."
        elif highest_prio == 3:
            rec = "High sensitivity signal detected. Escalate local monitoring and testing."
        elif highest_prio == 2:
            rec = "Elevated statistical activity. Review local clinic logs."
            
        weekly_warnings[week_str][dist] = {
            "status": best_status,
            "disease": best_dis if highest_prio > 1 else "-",
            "color": color,
            "recommendation": rec
        }

# -------------------------------------------------------------------------
# 3. Build Seasonal Risk Profiles (2018-2024)
# -------------------------------------------------------------------------
print("Building seasonal risk profiles...")
df_train = pd.read_pickle(os.path.join(data_dir, "train_timeseries.pkl"))
df_train['diagnosis_date'] = pd.to_datetime(df_train['diagnosis_date'])
df_train['month'] = df_train['diagnosis_date'].dt.month

seasonal_history = {}
for dist in districts:
    seasonal_history[dist] = {}
    dist_train = df_train[df_train['district'] == dist]
    for dis, grp in dist_train.groupby('disease_name'):
        monthly_avg = grp.groupby('month')['case_count'].mean().reindex(range(1,13), fill_value=0)
        seasonal_history[dist][dis] = monthly_avg.tolist()

# -------------------------------------------------------------------------
# 4. Load Prophet Predictions (Palakkad-Chikungunya)
# -------------------------------------------------------------------------
print("Loading Prophet predictions...")
prophet_csv = os.path.join(reports_dir, "prophet_predictions_palakkad_chikungunya.csv")
prophet_data = []
if os.path.exists(prophet_csv):
    df_pro = pd.read_csv(prophet_csv)
    # Just take every 3rd day to reduce JSON bloat for the chart
    df_pro = df_pro.iloc[::3].copy()
    for _, r in df_pro.iterrows():
        prophet_data.append({
            "date": r['date'],
            "actual": r['actual'],
            "predicted": r['predicted'],
            "upper": r['upper_95'],
            "anomaly": r['anomaly_high']
        })

# -------------------------------------------------------------------------
# 5. Build HTML
# -------------------------------------------------------------------------
print("Generating HTML dashboard...")

json_payload = {
    "weeks": [f"Week {w}" for w in weeks],
    "districts": districts,
    "warnings": weekly_warnings,
    "seasonal": seasonal_history,
    "prophet": prophet_data
}

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Early Outbreak Warning Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ background-color: #f8fafc; font-family: 'Inter', sans-serif; }}
        .card-green {{ border-top: 4px solid #10b981; background: white; }}
        .card-yellow {{ border-top: 4px solid #f59e0b; background: #fffbeb; }}
        .card-red {{ border-top: 4px solid #ef4444; background: #fef2f2; }}
        .text-green {{ color: #059669; }}
        .text-yellow {{ color: #d97706; }}
        .text-red {{ color: #dc2626; }}
        .active-card {{ outline: 2px solid #3b82f6; box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.5); }}
    </style>
</head>
<body class="text-slate-800">

<div class="max-w-7xl mx-auto px-4 py-6">
    <!-- Header -->
    <div class="mb-8">
        <h1 class="text-3xl font-bold text-slate-900 mb-2">Early Outbreak Warning Dashboard</h1>
        <p class="text-slate-600 mb-4">Statistical surveillance and forward-looking risk assessment for the Malabar region.</p>
        
        <div class="bg-white p-4 rounded-lg shadow-sm border border-slate-200 flex flex-wrap gap-6 items-center">
            <div class="flex items-center gap-2">
                <div class="w-4 h-4 rounded-full bg-red-500"></div>
                <span class="text-sm font-medium">Emergency (Confirmed Event)</span>
            </div>
            <div class="flex items-center gap-2">
                <div class="w-4 h-4 rounded-full bg-yellow-500"></div>
                <span class="text-sm font-medium">Watch / Advisory</span>
            </div>
            <div class="flex items-center gap-2">
                <div class="w-4 h-4 rounded-full bg-green-500"></div>
                <span class="text-sm font-medium">Normal</span>
            </div>
            <div class="flex-grow"></div>
            <div class="text-xs text-slate-500 italic max-w-lg text-right">
                Disclaimer: This system provides statistical early warnings based on recent surveillance data trends. It does not predict outbreaks with certainty. For official guidance, consult local health authorities.
            </div>
        </div>
    </div>

    <!-- Controls -->
    <div class="mb-6 flex items-center gap-4 bg-white p-4 rounded-lg shadow-sm border border-slate-200">
        <label for="weekSelect" class="font-semibold text-slate-700">Select Time Period:</label>
        <select id="weekSelect" class="bg-slate-50 border border-slate-300 text-slate-900 rounded-md focus:ring-blue-500 focus:border-blue-500 block p-2.5">
            <!-- Populated by JS -->
        </select>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        <!-- Region Grid -->
        <div class="lg:col-span-1">
            <h2 class="text-xl font-bold mb-4">Regional Status Map</h2>
            <div class="grid grid-cols-2 gap-4" id="districtGrid">
                <!-- Populated by JS -->
            </div>
        </div>

        <!-- Detail Panel -->
        <div class="lg:col-span-2 space-y-6">
            
            <!-- Status Detail -->
            <div class="bg-white rounded-lg shadow-sm border border-slate-200 p-6">
                <div class="flex justify-between items-start mb-4">
                    <h2 class="text-2xl font-bold" id="detailTitle">Select a District</h2>
                    <span id="detailBadge" class="px-3 py-1 rounded-full text-sm font-bold bg-slate-100 text-slate-600">No Data</span>
                </div>
                
                <div class="grid grid-cols-2 gap-6 mb-6">
                    <div>
                        <p class="text-sm text-slate-500 font-medium uppercase tracking-wider">Triggering Disease</p>
                        <p class="text-lg font-semibold" id="detailDisease">-</p>
                    </div>
                    <div>
                        <p class="text-sm text-slate-500 font-medium uppercase tracking-wider">Public Health Action</p>
                        <p class="text-md" id="detailAction">-</p>
                    </div>
                </div>
            </div>

            <!-- Seasonal Risk Panel -->
            <div class="bg-white rounded-lg shadow-sm border border-slate-200 p-6">
                <h3 class="text-xl font-bold mb-2">Upcoming Seasonal Risk</h3>
                <p class="text-sm text-slate-600 mb-4" id="seasonalWarning">
                    Comparing historical 2018-2024 patterns to identify upcoming risk windows.
                </p>
                <div class="h-64">
                    <canvas id="seasonalChart"></canvas>
                </div>
            </div>

        </div>
    </div>
</div>

<script>
    const DATA = {json.dumps(json_payload)};
    
    let currentWeek = DATA.weeks[0];
    let currentDistrict = "Palakkad"; // Default
    let chartInstance = null;

    // Init Dropdown
    const select = document.getElementById('weekSelect');
    DATA.weeks.forEach(w => {{
        const opt = document.createElement('option');
        opt.value = w;
        opt.textContent = w;
        select.appendChild(opt);
    }});
    // Default to a week with activity if possible
    if(DATA.weeks.includes("Week 23")) {{ currentWeek = "Week 23"; select.value = "Week 23"; }}
    
    select.addEventListener('change', (e) => {{
        currentWeek = e.target.value;
        renderGrid();
        renderDetail();
    }});

    function renderGrid() {{
        const grid = document.getElementById('districtGrid');
        grid.innerHTML = '';
        const weekData = DATA.warnings[currentWeek];
        
        DATA.districts.forEach(dist => {{
            const dData = weekData[dist];
            const div = document.createElement('div');
            
            let cardClass = "card-green";
            let textColor = "text-green";
            if(dData.color === 'red') {{ cardClass = "card-red"; textColor = "text-red"; }}
            else if(dData.color === 'yellow') {{ cardClass = "card-yellow"; textColor = "text-yellow"; }}
            
            if(dist === currentDistrict) cardClass += " active-card";
            
            div.className = `p-4 rounded shadow-sm cursor-pointer transition-all hover:shadow-md ${{cardClass}}`;
            div.onclick = () => {{
                currentDistrict = dist;
                renderGrid();
                renderDetail();
            }};
            
            div.innerHTML = `
                <h3 class="font-bold text-slate-800">${{dist}}</h3>
                <p class="text-sm font-medium ${{textColor}}">${{dData.status}}</p>
                ${{dData.disease !== '-' ? `<p class="text-xs text-slate-500 mt-1">${{dData.disease}}</p>` : ''}}
            `;
            grid.appendChild(div);
        }});
    }}

    function renderDetail() {{
        const dData = DATA.warnings[currentWeek][currentDistrict];
        
        document.getElementById('detailTitle').textContent = currentDistrict;
        const badge = document.getElementById('detailBadge');
        badge.textContent = dData.status;
        
        if(dData.color === 'red') {{
            badge.className = "px-3 py-1 rounded-full text-sm font-bold bg-red-100 text-red-700";
        }} else if(dData.color === 'yellow') {{
            badge.className = "px-3 py-1 rounded-full text-sm font-bold bg-yellow-100 text-yellow-700";
        }} else {{
            badge.className = "px-3 py-1 rounded-full text-sm font-bold bg-green-100 text-green-700";
        }}
        
        document.getElementById('detailDisease').textContent = dData.disease;
        document.getElementById('detailAction').textContent = dData.recommendation;

        renderChart();
    }}

    function renderChart() {{
        const ctx = document.getElementById('seasonalChart').getContext('2d');
        if(chartInstance) chartInstance.destroy();

        // If Palakkad, show Prophet Forecast for Chikungunya
        if(currentDistrict === "Palakkad") {{
            document.getElementById('seasonalWarning').innerHTML = `
                <strong>AI Forecast Warning:</strong> Prophet ML model indicates an upward trajectory 
                for <strong>Chikungunya</strong>. Historical seasonal peaks align with this forecast.
            `;
            
            const pData = DATA.prophet;
            chartInstance = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: pData.map(d => d.date),
                    datasets: [
                        {{
                            label: 'Predicted Trend (Prophet)',
                            data: pData.map(d => d.predicted),
                            borderColor: '#3b82f6',
                            borderWidth: 2,
                            tension: 0.1,
                            pointRadius: 0
                        }},
                        {{
                            label: 'Actual Cases',
                            data: pData.map(d => d.actual),
                            borderColor: '#94a3b8',
                            backgroundColor: '#94a3b8',
                            borderWidth: 0,
                            pointRadius: 2,
                            showLine: false
                        }},
                        {{
                            label: 'Upper 95% CI',
                            data: pData.map(d => d.upper),
                            borderColor: 'rgba(59, 130, 246, 0.2)',
                            backgroundColor: 'rgba(59, 130, 246, 0.1)',
                            borderWidth: 1,
                            pointRadius: 0,
                            fill: '-1' // Fills to previous dataset (which we'll make a hidden lower bound if needed, but for simplicity just fill to bottom)
                        }}
                    ]
                }},
                options: {{
                    responsive: true, maintainAspectRatio: false,
                    plugins: {{ legend: {{ position: 'bottom' }} }},
                    scales: {{
                        x: {{ ticks: {{ maxTicksLimit: 12 }} }}
                    }}
                }}
            }});
        }} else {{
            // Other districts: Show Historical Monthly Averages for top diseases
            document.getElementById('seasonalWarning').textContent = "Showing historical monthly averages (2018-2024) across all monitored diseases to anticipate upcoming seasonal spikes.";
            
            const sData = DATA.seasonal[currentDistrict];
            const datasets = [];
            const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#64748b'];
            let i = 0;
            
            for(const [dis, averages] of Object.entries(sData)) {{
                // Only plot diseases with notable historical counts to reduce noise
                if(Math.max(...averages) > 0.5) {{
                    datasets.push({{
                        label: dis,
                        data: averages,
                        borderColor: colors[i % colors.length],
                        tension: 0.3,
                        borderWidth: 2
                    }});
                    i++;
                }}
            }}
            
            chartInstance = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                    datasets: datasets
                }},
                options: {{
                    responsive: true, maintainAspectRatio: false,
                    plugins: {{ legend: {{ position: 'bottom' }} }}
                }}
            }});
        }}
    }}

    // Initial Render
    renderGrid();
    renderDetail();
</script>

</body>
</html>
"""

out_file = os.path.join(out_dir, "outbreak_dashboard.html")
with open(out_file, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Dashboard successfully generated at {out_file}")
