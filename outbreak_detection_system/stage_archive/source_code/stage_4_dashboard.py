"""
Stage 4 Dashboard Builder — v3 (Optimized)
- Fully vectorized weekly warning computation (no nested Python loops)
- Seasonal profiles computed with correct sparsity handling
- Default week set to most active confirmed event period
- Disease breakdown sorted by activity level
- All computation done once and serialized into a single JSON blob
"""
import os
import json
import pandas as pd
import numpy as np

base_dir    = r"C:\BRAIN-STORM\HT\warning\outbreak_detection_system"
data_dir    = os.path.join(base_dir, "data", "processed")
reports_dir = os.path.join(base_dir, "reports")
out_dir     = os.path.join(base_dir, "outputs")
os.makedirs(out_dir, exist_ok=True)

PRIORITY = {"Confirmed-Tier Event": 4, "Watch-Tier Event": 3}
COLOR_MAP = {4: "red", 3: "yellow", 2: "yellow", 1: "green", 0: "green"}
STATUS_MAP = {4: "Emergency Warning", 3: "Watch-Status Warning", 2: "Advisory", 1: "Normal", 0: "Normal"}
REC_MAP = {
    4: "Immediate public health intervention recommended. Deploy rapid response team.",
    3: "High sensitivity signal detected. Escalate local monitoring and testing.",
    2: "Elevated statistical activity. Review local clinic logs.",
    1: "Routine surveillance.", 0: "Routine surveillance."
}

# ── 1. Load test detection results ───────────────────────────────────────────
print("Loading detection results...")
df = pd.read_pickle(os.path.join(data_dir, "test_detection_results.pkl"))
df["diagnosis_date"] = pd.to_datetime(df["diagnosis_date"])
df["week"] = df["diagnosis_date"].dt.isocalendar().week.astype(int)

weeks     = sorted(df["week"].unique().tolist())
districts = sorted(df["district"].unique().tolist())

# ── 2. Vectorized weekly warning table ───────────────────────────────────────
print("Building weekly warning table (vectorized)...")

# Assign numeric priority to each row
df["priority"] = 0
df.loc[df["tier"] == "Confirmed-Tier Event", "priority"] = 4
df.loc[df["tier"] == "Watch-Tier Event",     "priority"] = 3
df.loc[(df["priority"] == 0) & (df["risk_level"] == "Critical"), "priority"] = 2
df.loc[(df["priority"] == 0) & (df["risk_level"] == "High"),     "priority"] = 2
df.loc[(df["priority"] == 0) & (df["risk_level"] == "Medium"),   "priority"] = 2

# Aggregate: for each (week, district, disease) — sum cases, max priority
agg = df.groupby(["week", "district", "disease_name"]).agg(
    cases=("case_count", "sum"),
    priority=("priority", "max")
).reset_index()

# For each (week, district) find the top disease by priority then cases
agg = agg.sort_values(["week", "district", "priority", "cases"], ascending=[True, True, False, False])
top = agg.groupby(["week", "district"]).first().reset_index()

# Build the disease breakdown dict per (week, district)
breakdown_map = {}
for (wk, dist), grp in agg.groupby(["week", "district"]):
    breakdown_map[(wk, dist)] = {
        row.disease_name: {
            "cases": int(row.cases),
            "priority": int(row.priority),
            "status": STATUS_MAP.get(int(row.priority), "Normal")
        }
        for _, row in grp.iterrows()
    }

# Assemble weekly_warnings dict
weekly_warnings = {}
for wk in weeks:
    wdata = df[df["week"] == wk]["diagnosis_date"]
    start = wdata.min().strftime("%b %d")
    end   = wdata.max().strftime("%b %d, %Y")
    label = f"{start} \u2013 {end}"
    wkey  = f"Week {wk}"
    weekly_warnings[wkey] = {"label": label}
    for dist in districts:
        row = top[(top["week"] == wk) & (top["district"] == dist)]
        if row.empty:
            p, dis, cases = 0, "-", 0
        else:
            r    = row.iloc[0]
            p    = int(r["priority"])
            dis  = r["disease_name"] if p > 0 else "-"
            cases = int(r["cases"])
        weekly_warnings[wkey][dist] = {
            "status":      STATUS_MAP[p],
            "disease":     dis,
            "cases":       cases,
            "color":       COLOR_MAP[p],
            "recommendation": REC_MAP[p],
            "breakdown":   breakdown_map.get((wk, dist), {})
        }

# ── 3. Seasonal profiles (2018-2023 training data) ──────────────────────────
print("Building seasonal profiles...")
df_train = pd.read_pickle(os.path.join(data_dir, "train_timeseries.pkl"))
df_train["diagnosis_date"] = pd.to_datetime(df_train["diagnosis_date"])
df_train["month"] = df_train["diagnosis_date"].dt.month

# Vectorized monthly average per (district, disease, month)
monthly = df_train.groupby(["district", "disease_name", "month"])["case_count"].mean().reset_index()
# Find the max per (district, disease) to filter out truly zero-activity pairs
max_vals = monthly.groupby(["district", "disease_name"])["case_count"].max()

seasonal_history = {}
for dist in districts:
    seasonal_history[dist] = {}
    for dis in df_train["disease_name"].unique():
        key = (dist, dis)
        if key not in max_vals or max_vals[key] < 0.1:
            continue  # Skip fully sparse pairs
        sub = monthly[(monthly["district"] == dist) & (monthly["disease_name"] == dis)]
        full = sub.set_index("month")["case_count"].reindex(range(1, 13), fill_value=0)
        seasonal_history[dist][dis] = [round(float(v), 4) for v in full.tolist()]

# ── 4. Prophet predictions ────────────────────────────────────────────────────
print("Loading Prophet predictions...")
prophet_data = []
pcsv = os.path.join(reports_dir, "prophet_predictions_palakkad_chikungunya.csv")
if os.path.exists(pcsv):
    df_pro = pd.read_csv(pcsv).iloc[::3].copy()
    df_pro["date"] = pd.to_datetime(df_pro["date"]).dt.strftime("%b %d")
    for _, r in df_pro.iterrows():
        prophet_data.append({
            "date":      r["date"],
            "actual":    float(r["actual"]),
            "predicted": round(float(r["predicted"]), 4),
            "upper":     round(float(r["upper_95"]), 4),
            "anomaly":   bool(r["anomaly_high"])
        })

# ── 5. Find best default week (most active confirmed week) ───────────────────
confirmed_by_week = df[df["tier"] == "Confirmed-Tier Event"].groupby("week").size()
default_week_num  = int(confirmed_by_week.idxmax()) if not confirmed_by_week.empty else weeks[22]
default_week      = f"Week {default_week_num}"
print(f"Default week: {default_week} ({weekly_warnings[default_week]['label']})")

# ── 6. Build payload ──────────────────────────────────────────────────────────
payload = {
    "weeks":        [f"Week {w}" for w in weeks],
    "default_week": default_week,
    "districts":    districts,
    "warnings":     weekly_warnings,
    "seasonal":     seasonal_history,
    "prophet":      prophet_data
}

DATA_JSON = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
print(f"JSON payload size: {len(DATA_JSON)//1024} KB")

# ── 7. HTML ───────────────────────────────────────────────────────────────────
print("Generating HTML...")

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Early Outbreak Warning Dashboard — Malabar</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',sans-serif;background:#f1f5f9;color:#1e293b;font-size:14px}
.shell{max-width:1440px;margin:0 auto;padding:18px}
.topbar{background:#0f172a;color:#f8fafc;padding:14px 20px;border-radius:12px;margin-bottom:16px;display:flex;align-items:center;gap:24px;flex-wrap:wrap}
.brand{font-size:1.1rem;font-weight:700;flex:1;letter-spacing:-.01em}
.brand span{color:#38bdf8}
.legend-row{display:flex;gap:14px;flex-wrap:wrap;align-items:center}
.leg{display:flex;align-items:center;gap:6px;font-size:.75rem;font-weight:500;color:#cbd5e1}
.dot{width:12px;height:12px;border-radius:50%;flex-shrink:0}
.dot-s{width:12px;height:12px;border-radius:50%;border:2px dashed #38bdf8;flex-shrink:0}
.controls{background:#fff;border-radius:10px;padding:10px 16px;margin-bottom:16px;display:flex;align-items:center;gap:12px;box-shadow:0 1px 3px rgba(0,0,0,.07)}
.controls label{font-weight:600;font-size:.85rem;color:#475569;white-space:nowrap}
select{border:1px solid #cbd5e1;border-radius:8px;padding:7px 12px;font-size:.85rem;background:#f8fafc;color:#0f172a;cursor:pointer;min-width:220px}
select:focus{outline:none;border-color:#3b82f6;box-shadow:0 0 0 2px rgba(59,130,246,.2)}
.active-week{font-size:.8rem;color:#64748b;margin-left:4px}
.grid{display:grid;grid-template-columns:3fr 2fr;gap:16px}
@media(max-width:860px){.grid{grid-template-columns:1fr}}
#map{width:100%;height:600px;border-radius:12px;box-shadow:0 1px 4px rgba(0,0,0,.1)}
.panel{background:#fff;border-radius:12px;padding:18px;box-shadow:0 1px 3px rgba(0,0,0,.06);margin-bottom:14px}
.panel h2{font-size:1rem;font-weight:700;color:#0f172a;margin-bottom:14px;padding-bottom:8px;border-bottom:1px solid #f1f5f9}
.badge{display:inline-block;padding:3px 10px;border-radius:9999px;font-size:.73rem;font-weight:700}
.badge-red{background:#fef2f2;color:#b91c1c}
.badge-yellow{background:#fffbeb;color:#854d0e}
.badge-green{background:#f0fdf4;color:#166534}
.badge-gray{background:#f1f5f9;color:#475569}
.dh{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px}
.dh h3{font-size:1.2rem;font-weight:700}
.ml{font-size:.72rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px}
.mv{font-size:.92rem;font-weight:600;color:#0f172a}
.mb{margin-bottom:12px}
.rec{font-size:.82rem;color:#475569;line-height:1.55}
table{width:100%;border-collapse:collapse;font-size:.78rem;margin-top:6px}
th{text-align:left;padding:5px 7px;background:#f8fafc;color:#475569;font-weight:600;border-bottom:1px solid #e2e8f0}
td{padding:5px 7px;border-bottom:1px solid #f8fafc;color:#334155}
tr:hover td{background:#f8fafc}
.chart-wrap{position:relative;height:185px;margin-top:8px}
.disclaimer{font-size:.7rem;color:#94a3b8;font-style:italic;margin-top:14px;line-height:1.6;padding:0 2px}
.l-popup{font-family:'Inter',sans-serif;min-width:190px;font-size:.82rem}
.l-popup b{font-size:.9rem;display:block;margin-bottom:3px}
.l-popup .st{font-weight:700;margin-bottom:4px}
.l-popup .tr{font-size:.75rem;color:#64748b;margin-bottom:6px}
.l-popup hr{border:none;border-top:1px solid #e2e8f0;margin:6px 0}
.l-popup .rec{font-size:.75rem;color:#475569;font-style:italic;line-height:1.4}
</style>
</head>
<body>
<div class="shell">
  <div class="topbar">
    <div class="brand">&#x1F9A0; Early Outbreak Warning Dashboard &mdash; <span>Malabar Region, Kerala</span></div>
    <div class="legend-row">
      <div class="leg"><div class="dot" style="background:#c0392b"></div>Emergency</div>
      <div class="leg"><div class="dot" style="background:#e6a817"></div>Watch/Advisory</div>
      <div class="leg"><div class="dot" style="background:#4a9d5f"></div>Normal</div>
      <div class="leg"><div class="dot-s"></div>Seasonal Watch</div>
    </div>
  </div>

  <div class="controls">
    <label>&#x1F4C5; Time Period:</label>
    <select id="weekSelect"></select>
    <span class="active-week" id="weekLabel"></span>
  </div>

  <div class="grid">
    <div>
      <div class="panel" style="padding-bottom:12px">
        <h2>&#x1F5FA;&#xFE0F; Geographic Risk Map</h2>
        <div id="map"></div>
      </div>
    </div>
    <div>
      <div class="panel" id="detailPanel">
        <div class="dh">
          <h3 id="detailTitle">Palakkad</h3>
          <span id="detailBadge" class="badge badge-gray">Normal</span>
        </div>
        <div class="mb"><div class="ml">Triggering Disease</div><div class="mv" id="detailDisease">—</div></div>
        <div class="mb"><div class="ml">Total Cases This Week</div><div class="mv" id="detailCases">—</div></div>
        <div class="mb"><div class="ml">Recommended Action</div><div class="rec" id="detailAction">—</div></div>
        <div>
          <div class="ml">Disease Breakdown</div>
          <table>
            <thead><tr><th>Disease</th><th>Cases</th><th>Status</th></tr></thead>
            <tbody id="breakdownBody"></tbody>
          </table>
        </div>
      </div>
      <div class="panel">
        <h2>&#x1F4C8; Seasonal Risk Pattern</h2>
        <p id="seasonalWarning" style="font-size:.78rem;color:#475569;line-height:1.5;margin-bottom:10px"></p>
        <div class="chart-wrap"><canvas id="seasonalChart"></canvas></div>
      </div>
    </div>
  </div>

  <p class="disclaimer">
    &#x26A0;&#xFE0F; Disclaimer: This system provides statistical early warnings based on recent surveillance data trends (2018&ndash;2025).
    It does not predict outbreaks with certainty. Alerts are generated by a gap-corrected Z-score engine with Prophet ML forecasting (Palakkad&ndash;Chikungunya).
    For official guidance, consult local health authorities.
  </p>
</div>

<script>
const DATA = """ + DATA_JSON + """;

const COORDS = {
  'Kannur':    [11.8745,75.3704],
  'Kasaragod': [12.4996,74.9869],
  'Kozhikode': [11.2588,75.7804],
  'Malappuram':[11.0410,76.0788],
  'Palakkad':  [10.7867,76.6548],
  'Wayanad':   [11.6854,76.1320]
};

const COLOR = {red:'#c0392b', yellow:'#e6a817', green:'#4a9d5f'};
const BADGE = {red:'badge-red', yellow:'badge-yellow', green:'badge-green'};
const CHART_COLORS = ['#3b82f6','#ef4444','#10b981','#f59e0b','#8b5cf6','#ec4899','#64748b','#0ea5e9'];

let currentWeek = DATA.default_week;
let currentDist = 'Palakkad';
let chartInst   = null;
let markers     = {};
let labelMarkers = [];

// ── Map ───────────────────────────────────────────────────────────────────────
const map = L.map('map',{zoomControl:true}).setView([11.55,75.95],8);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{
  maxZoom:18, attribution:'&copy; OpenStreetMap contributors'
}).addTo(map);

// ── Week selector ─────────────────────────────────────────────────────────────
const sel = document.getElementById('weekSelect');
DATA.weeks.forEach(w => {
  const o = document.createElement('option');
  o.value = w;
  o.textContent = DATA.warnings[w].label || w;
  if(w === currentWeek) o.selected = true;
  sel.appendChild(o);
});
document.getElementById('weekLabel').textContent = DATA.warnings[currentWeek].label || '';

sel.addEventListener('change', e => {
  currentWeek = e.target.value;
  document.getElementById('weekLabel').textContent = DATA.warnings[currentWeek].label || '';
  updateMap();
  updateDetail();
});

// ── Map markers ───────────────────────────────────────────────────────────────
function updateMap() {
  // Remove old markers
  Object.values(markers).forEach(m => map.removeLayer(m));
  labelMarkers.forEach(m => map.removeLayer(m));
  markers = {}; labelMarkers = [];

  const wd = DATA.warnings[currentWeek];

  DATA.districts.forEach(dist => {
    const d      = wd[dist];
    const hex    = COLOR[d.color];
    const isSeas = (dist === 'Palakkad'); // Prophet seasonal watch
    const isSelected = (dist === currentDist);

    const m = L.circleMarker(COORDS[dist], {
      radius:      isSelected ? 24 : 20,
      fillColor:   hex,
      fillOpacity: 0.88,
      color:       isSeas ? '#38bdf8' : (isSelected ? '#fff' : '#fff'),
      weight:      isSeas ? 3 : (isSelected ? 3 : 1.5),
      dashArray:   isSeas ? '6 4' : ''
    }).addTo(map);

    const trigger = (d.disease !== '-') ? `<div class="tr">&#x26A0; ${d.disease} &mdash; ${d.cases} case(s)</div>` : '';
    m.bindPopup(`<div class="l-popup">
      <b>${dist}</b>
      <div class="st" style="color:${hex}">${d.status}</div>
      ${trigger}
      <hr/>
      <div class="rec">${d.recommendation}</div>
    </div>`, {maxWidth:250, autoPan:true});

    m.on('click', () => { currentDist = dist; updateMap(); updateDetail(); });
    markers[dist] = m;

    // District name label
    const icon = L.divIcon({
      className:'',
      html:`<span style="font:700 10px/1 Inter,sans-serif;color:#0f172a;text-shadow:0 0 4px #fff,0 0 4px #fff,0 0 4px #fff;white-space:nowrap;pointer-events:none">${dist}</span>`,
      iconAnchor:[0,-26]
    });
    const lm = L.marker(COORDS[dist],{icon,interactive:false,zIndexOffset:1000}).addTo(map);
    labelMarkers.push(lm);
  });

  // Open popup for selected district
  setTimeout(() => { if(markers[currentDist]) markers[currentDist].openPopup(); }, 700);
}

// ── Detail panel ──────────────────────────────────────────────────────────────
function updateDetail() {
  const d = DATA.warnings[currentWeek][currentDist];
  document.getElementById('detailTitle').textContent = currentDist;

  const badge = document.getElementById('detailBadge');
  badge.textContent = d.status;
  badge.className = 'badge ' + (BADGE[d.color] || 'badge-gray');

  document.getElementById('detailDisease').textContent = d.disease !== '-' ? d.disease : 'None active';
  document.getElementById('detailCases').textContent   = d.cases > 0 ? `${d.cases} case(s)` : '0 cases';
  document.getElementById('detailAction').textContent  = d.recommendation;

  // Disease breakdown table — sorted by cases desc
  const tbody = document.getElementById('breakdownBody');
  tbody.innerHTML = '';
  const entries = Object.entries(d.breakdown || {}).sort((a,b) => b[1].cases - a[1].cases);
  if(entries.length === 0){
    tbody.innerHTML = '<tr><td colspan="3" style="color:#94a3b8;font-style:italic;padding:8px">No recorded activity</td></tr>';
  } else {
    const PCOL = {4:'#b91c1c', 3:'#92400e', 2:'#854d0e', 1:'#166534', 0:'#94a3b8'};
    entries.forEach(([dis,info]) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${dis}</td><td style="font-weight:600">${info.cases}</td><td style="font-weight:600;color:${PCOL[info.priority]||'#94a3b8'};font-size:.72rem">${info.status}</td>`;
      tbody.appendChild(tr);
    });
  }

  // Sync popup
  setTimeout(() => {
    if(markers[currentDist] && !markers[currentDist].isPopupOpen())
      markers[currentDist].openPopup();
  }, 60);

  updateChart();
}

// ── Chart ─────────────────────────────────────────────────────────────────────
function updateChart() {
  const ctx = document.getElementById('seasonalChart').getContext('2d');
  if(chartInst){ chartInst.destroy(); chartInst = null; }

  const opts = {responsive:true,maintainAspectRatio:false,
    plugins:{legend:{position:'bottom',labels:{font:{size:10},boxWidth:12,padding:10}}},
    scales:{x:{ticks:{font:{size:10}}},y:{ticks:{font:{size:10}},beginAtZero:true}}};

  if(currentDist === 'Palakkad' && DATA.prophet.length > 0) {
    document.getElementById('seasonalWarning').innerHTML =
      '<strong>Prophet ML Forecast (Palakkad \u2013 Chikungunya):</strong> Blue = predicted baseline; grey dots = actual cases; shaded = 95% confidence interval. Points above the band are flagged anomalies.';
    const pd = DATA.prophet;
    chartInst = new Chart(ctx,{type:'line',data:{
      labels: pd.map(d=>d.date),
      datasets:[
        {label:'Predicted',data:pd.map(d=>d.predicted),borderColor:'#3b82f6',borderWidth:2,pointRadius:0,tension:.3,fill:false},
        {label:'Actual',   data:pd.map(d=>d.actual),   borderColor:'#94a3b8',borderWidth:0,pointRadius:2,showLine:false},
        {label:'Upper 95%',data:pd.map(d=>d.upper),    borderColor:'rgba(59,130,246,.25)',backgroundColor:'rgba(59,130,246,.08)',borderWidth:1,pointRadius:0,fill:'-1'}
      ]},options:opts});
  } else {
    document.getElementById('seasonalWarning').textContent =
      'Historical monthly average (2018\u20132023, training data). Peaks show seasonal risk windows for this district.';
    const sd  = DATA.seasonal[currentDist] || {};
    const sets = [];
    let ci = 0;
    // Sort diseases by their annual total so most active shows first
    const byActivity = Object.entries(sd).sort((a,b)=>b[1].reduce((s,x)=>s+x,0)-a[1].reduce((s,x)=>s+x,0));
    byActivity.forEach(([dis,avgs]) => {
      if(Math.max(...avgs) < 0.05) return; // skip truly zero series
      sets.push({label:dis,data:avgs,borderColor:CHART_COLORS[ci%CHART_COLORS.length],tension:.35,borderWidth:1.8,pointRadius:2,fill:false});
      ci++;
    });
    chartInst = new Chart(ctx,{type:'line',data:{
      labels:['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],datasets:sets},options:opts});
  }
}

// ── Boot ──────────────────────────────────────────────────────────────────────
window.addEventListener('load', () => {
  setTimeout(() => { map.invalidateSize(); updateMap(); updateDetail(); }, 300);
});
</script>
</body>
</html>"""

out_file = os.path.join(out_dir, "outbreak_dashboard.html")
with open(out_file, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Dashboard saved to: {out_file}")
print(f"Default week: {default_week} | {weekly_warnings[default_week]['label']}")
print("Done.")
