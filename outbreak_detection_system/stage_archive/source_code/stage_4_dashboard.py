"""
Stage 4 Dashboard Builder — v2 (Fixed + Enhanced)
Generates a self-contained HTML dashboard with Leaflet map.
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

# ── 1. Load test data ────────────────────────────────────────────────────────
print("Loading data...")
df_test = pd.read_pickle(os.path.join(data_dir, "test_detection_results.pkl"))
df_test["diagnosis_date"] = pd.to_datetime(df_test["diagnosis_date"])
df_test["week"] = df_test["diagnosis_date"].dt.isocalendar().week.astype(int)

weeks     = sorted(df_test["week"].unique().tolist())
districts = sorted(df_test["district"].unique().tolist())

# ── 2. Weekly warning table ───────────────────────────────────────────────────
print("Building weekly warning table...")

def get_status(grp):
    if (grp["tier"] == "Confirmed-Tier Event").any():
        return 4, "Emergency Warning"
    elif (grp["tier"] == "Watch-Tier Event").any():
        return 3, "Watch-Status Warning"
    elif (grp["risk_level"] != "Low").any():
        return 2, "Advisory"
    return 1, "Normal"

weekly_warnings = {}
for w in weeks:
    wk = f"Week {w}"
    weekly_warnings[wk] = {}
    wd = df_test[df_test["week"] == w]
    for dist in districts:
        dd = wd[wd["district"] == dist]
        best_p, best_s, best_dis, best_cases = 0, "Normal", "-", 0
        disease_breakdown = {}
        for dis, grp in dd.groupby("disease_name"):
            p, s = get_status(grp)
            total = int(grp["case_count"].sum())
            disease_breakdown[dis] = {"cases": total, "status": s, "priority": p}
            if p > best_p:
                best_p, best_s, best_dis = p, s, dis
                best_cases = total
        color = "green"
        if best_p == 4:   color = "red"
        elif best_p >= 2: color = "yellow"
        rec = "Routine surveillance."
        if best_p == 4: rec = "Immediate public health intervention recommended. Deploy rapid response team."
        elif best_p == 3: rec = "High sensitivity signal detected. Escalate local monitoring and testing."
        elif best_p == 2: rec = "Elevated statistical activity. Review local clinic logs."
        weekly_warnings[wk][dist] = {
            "status": best_s,
            "disease": best_dis,
            "cases": best_cases,
            "color": color,
            "recommendation": rec,
            "breakdown": disease_breakdown
        }

# ── 3. Seasonal profiles (2018-2024) ─────────────────────────────────────────
print("Building seasonal risk profiles...")
df_train = pd.read_pickle(os.path.join(data_dir, "train_timeseries.pkl"))
df_train["diagnosis_date"] = pd.to_datetime(df_train["diagnosis_date"])
df_train["month"] = df_train["diagnosis_date"].dt.month
seasonal_history = {}
for dist in districts:
    seasonal_history[dist] = {}
    dt = df_train[df_train["district"] == dist]
    for dis, grp in dt.groupby("disease_name"):
        ma = grp.groupby("month")["case_count"].mean().reindex(range(1, 13), fill_value=0)
        seasonal_history[dist][dis] = [round(v, 3) for v in ma.tolist()]

# ── 4. Prophet predictions ────────────────────────────────────────────────────
print("Loading Prophet predictions...")
prophet_data = []
pcsv = os.path.join(reports_dir, "prophet_predictions_palakkad_chikungunya.csv")
if os.path.exists(pcsv):
    df_pro = pd.read_csv(pcsv).iloc[::3]
    for _, r in df_pro.iterrows():
        prophet_data.append({
            "date": r["date"], "actual": r["actual"],
            "predicted": round(float(r["predicted"]), 4),
            "upper": round(float(r["upper_95"]), 4),
            "anomaly": bool(r["anomaly_high"])
        })

# ── 5. Build payload and HTML ─────────────────────────────────────────────────
print("Generating HTML dashboard...")

payload = {
    "weeks":     [f"Week {w}" for w in weeks],
    "districts": districts,
    "warnings":  weekly_warnings,
    "seasonal":  seasonal_history,
    "prophet":   prophet_data
}

DATA_JSON = json.dumps(payload, ensure_ascii=False)

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Early Outbreak Warning Dashboard — Malabar</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Inter',sans-serif;background:#f1f5f9;color:#1e293b}
  h1{font-size:1.75rem;font-weight:700;color:#0f172a}
  h2{font-size:1.1rem;font-weight:700;color:#0f172a;margin-bottom:12px}
  h3{font-size:.9rem;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px}
  .shell{max-width:1400px;margin:0 auto;padding:20px}
  /* header */
  .topbar{background:#fff;border-bottom:1px solid #e2e8f0;padding:16px 24px;display:flex;align-items:center;gap:24px;flex-wrap:wrap;margin-bottom:20px;border-radius:12px;box-shadow:0 1px 3px rgba(0,0,0,.06)}
  .brand{font-size:1.25rem;font-weight:700;color:#0f172a;flex:1}
  .legend-row{display:flex;gap:16px;flex-wrap:wrap;align-items:center}
  .leg{display:flex;align-items:center;gap:6px;font-size:.8rem;font-weight:500}
  .dot{width:14px;height:14px;border-radius:50%}
  .dot-dashed{width:14px;height:14px;border-radius:50%;border:2px dashed #3b82f6;background:transparent}
  /* controls */
  .controls{display:flex;align-items:center;gap:12px;background:#fff;padding:12px 16px;border-radius:10px;box-shadow:0 1px 3px rgba(0,0,0,.06);margin-bottom:20px}
  select{border:1px solid #cbd5e1;border-radius:8px;padding:8px 12px;font-size:.9rem;background:#f8fafc;color:#0f172a;cursor:pointer}
  select:focus{outline:none;border-color:#3b82f6;box-shadow:0 0 0 2px rgba(59,130,246,.2)}
  /* layout */
  .grid{display:grid;grid-template-columns:2fr 1fr;gap:20px}
  @media(max-width:900px){.grid{grid-template-columns:1fr}}
  /* map */
  #map{width:100%;height:580px;border-radius:12px;box-shadow:0 1px 4px rgba(0,0,0,.1)}
  /* panel */
  .panel{background:#fff;border-radius:12px;padding:20px;box-shadow:0 1px 3px rgba(0,0,0,.06);margin-bottom:16px}
  .badge{display:inline-block;padding:4px 12px;border-radius:9999px;font-size:.78rem;font-weight:700}
  .badge-red{background:#fef2f2;color:#b91c1c}
  .badge-yellow{background:#fffbeb;color:#92400e}
  .badge-green{background:#f0fdf4;color:#166534}
  .badge-gray{background:#f1f5f9;color:#475569}
  .district-header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:16px}
  .meta-label{font-size:.75rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px}
  .meta-value{font-size:.95rem;font-weight:600;color:#0f172a}
  .meta-block{margin-bottom:14px}
  /* disease breakdown table */
  .dis-table{width:100%;border-collapse:collapse;font-size:.82rem;margin-top:10px}
  .dis-table th{text-align:left;padding:5px 8px;background:#f8fafc;color:#475569;font-weight:600;border-bottom:1px solid #e2e8f0}
  .dis-table td{padding:5px 8px;border-bottom:1px solid #f1f5f9}
  .dis-table tr:hover td{background:#f8fafc}
  /* chart */
  .chart-wrap{position:relative;height:200px}
  /* disclaimer */
  .disclaimer{font-size:.72rem;color:#94a3b8;font-style:italic;margin-top:20px;line-height:1.5}
  /* leaflet popup */
  .l-popup{font-family:'Inter',sans-serif;min-width:180px}
  .l-popup .dist-name{font-size:.95rem;font-weight:700;margin-bottom:4px}
  .l-popup .status-txt{font-weight:600;font-size:.85rem;margin-bottom:4px}
  .l-popup .trigger{font-size:.78rem;color:#64748b}
  .l-popup hr{border:none;border-top:1px solid #e2e8f0;margin:8px 0}
  .l-popup .rec-txt{font-size:.75rem;color:#475569;font-style:italic;line-height:1.4}
</style>
</head>
<body>
<div class="shell">
  <!-- Topbar -->
  <div class="topbar">
    <div class="brand">🦠 Early Outbreak Warning Dashboard &mdash; Malabar Region</div>
    <div class="legend-row">
      <div class="leg"><div class="dot" style="background:#c0392b"></div>Emergency</div>
      <div class="leg"><div class="dot" style="background:#e6a817"></div>Watch / Advisory</div>
      <div class="leg"><div class="dot" style="background:#4a9d5f"></div>Normal</div>
      <div class="leg"><div class="dot-dashed"></div>Seasonal Watch</div>
    </div>
  </div>
  <!-- Controls -->
  <div class="controls">
    <label style="font-weight:600;font-size:.9rem">Time Period:</label>
    <select id="weekSelect"></select>
    <span style="font-size:.8rem;color:#64748b;margin-left:8px" id="weekLabel"></span>
  </div>
  <!-- Main Grid -->
  <div class="grid">
    <!-- Map Column -->
    <div>
      <h2>Geographic Risk Map</h2>
      <div id="map"></div>
    </div>
    <!-- Detail Column -->
    <div>
      <!-- Status card -->
      <div class="panel" id="detailPanel">
        <div class="district-header">
          <div>
            <h2 id="detailTitle" style="font-size:1.3rem">Palakkad</h2>
            <span id="detailBadge" class="badge badge-gray">Normal</span>
          </div>
        </div>
        <div class="meta-block">
          <div class="meta-label">Triggering Disease</div>
          <div class="meta-value" id="detailDisease">—</div>
        </div>
        <div class="meta-block">
          <div class="meta-label">Cases This Week</div>
          <div class="meta-value" id="detailCases">—</div>
        </div>
        <div class="meta-block">
          <div class="meta-label">Public Health Action</div>
          <div style="font-size:.85rem;color:#475569;line-height:1.5" id="detailAction">—</div>
        </div>
        <div class="meta-block">
          <div class="meta-label">Disease Breakdown (this week)</div>
          <table class="dis-table">
            <thead><tr><th>Disease</th><th>Cases</th><th>Status</th></tr></thead>
            <tbody id="breakdownBody"></tbody>
          </table>
        </div>
      </div>
      <!-- Seasonal panel -->
      <div class="panel">
        <h2>Upcoming Seasonal Risk</h2>
        <p id="seasonalWarning" style="font-size:.82rem;color:#475569;margin-bottom:12px;line-height:1.5"></p>
        <div class="chart-wrap"><canvas id="seasonalChart"></canvas></div>
      </div>
    </div>
  </div>
  <p class="disclaimer">Disclaimer: This system provides statistical early warnings based on recent surveillance data trends. It does not predict outbreaks with certainty. For official guidance, consult local health authorities.</p>
</div>

<script>
const DATA = """ + DATA_JSON + """;

const COORDS = {
  'Kannur':    [11.8745, 75.3704],
  'Kasaragod': [12.4996, 74.9869],
  'Kozhikode': [11.2588, 75.7804],
  'Malappuram':[11.0410, 76.0788],
  'Palakkad':  [10.7867, 76.6548],
  'Wayanad':   [11.6854, 76.1320]
};

let currentWeek     = DATA.weeks.includes('Week 23') ? 'Week 23' : DATA.weeks[0];
let currentDistrict = 'Palakkad';
let chartInst       = null;
let markers         = {};

// ── Map init ─────────────────────────────────────────────────────────────────
const map = L.map('map', {zoomControl:true}).setView([11.55, 75.95], 8);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 18,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
}).addTo(map);

// ── Week selector ─────────────────────────────────────────────────────────────
const sel = document.getElementById('weekSelect');
DATA.weeks.forEach(w => {
  const o = document.createElement('option');
  o.value = w; o.textContent = w;
  if(w === currentWeek) o.selected = true;
  sel.appendChild(o);
});
sel.addEventListener('change', e => {
  currentWeek = e.target.value;
  updateMap();
  updateDetail();
});

// ── Render markers ─────────────────────────────────────────────────────────────
function colorFor(c){ return c==='red'?'#c0392b': c==='yellow'?'#e6a817':'#4a9d5f'; }

function updateMap() {
  Object.values(markers).forEach(m => map.removeLayer(m));
  markers = {};
  const wd = DATA.warnings[currentWeek];
  DATA.districts.forEach(dist => {
    const d   = wd[dist];
    const hex = colorFor(d.color);
    const isSeasonal = (dist === 'Palakkad');
    const m = L.circleMarker(COORDS[dist], {
      radius: 20, fillColor: hex, fillOpacity: 0.88,
      color: isSeasonal ? '#3b82f6' : '#fff',
      weight: isSeasonal ? 3 : 1.5,
      dashArray: isSeasonal ? '6 4' : ''
    }).addTo(map);

    // District label
    const lbl = L.divIcon({
      className:'',
      html:`<div style="font-size:10px;font-weight:700;color:#0f172a;text-shadow:0 0 3px #fff,0 0 3px #fff;white-space:nowrap;pointer-events:none">${dist}</div>`,
      iconAnchor:[0,-22]
    });
    L.marker(COORDS[dist], {icon:lbl, interactive:false, zIndexOffset:1000}).addTo(map);

    const trigger = d.disease !== '-' ? `<div class="trigger">&#x26A0;&#xFE0F; ${d.disease}</div>` : '';
    m.bindPopup(`
      <div class="l-popup">
        <div class="dist-name">${dist}</div>
        <div class="status-txt" style="color:${hex}">${d.status}</div>
        ${trigger}
        <hr/>
        <div class="rec-txt">${d.recommendation}</div>
      </div>
    `, {maxWidth:240});

    m.on('click', () => { currentDistrict = dist; updateDetail(); });
    markers[dist] = m;
  });
  // Open popup for current district after map settles
  setTimeout(() => {
    if(markers[currentDistrict]) markers[currentDistrict].openPopup();
  }, 800);
}

// ── Detail panel ──────────────────────────────────────────────────────────────
function updateDetail() {
  const d = DATA.warnings[currentWeek][currentDistrict];
  document.getElementById('detailTitle').textContent = currentDistrict;
  const badge = document.getElementById('detailBadge');
  badge.textContent = d.status;
  badge.className = 'badge ' + (d.color==='red'?'badge-red': d.color==='yellow'?'badge-yellow':'badge-green');
  document.getElementById('detailDisease').textContent = d.disease !== '-' ? d.disease : 'None active';
  document.getElementById('detailCases').textContent   = d.cases > 0 ? d.cases + ' cases' : '0 cases';
  document.getElementById('detailAction').textContent  = d.recommendation;

  // Disease breakdown table
  const tbody = document.getElementById('breakdownBody');
  tbody.innerHTML = '';
  const bk = d.breakdown || {};
  const sorted = Object.entries(bk).sort((a,b) => b[1].cases - a[1].cases);
  sorted.forEach(([dis, info]) => {
    const tr = document.createElement('tr');
    const statusColor = info.priority>=4?'#c0392b': info.priority>=2?'#e6a817':'#4a9d5f';
    tr.innerHTML = `<td>${dis}</td><td>${info.cases}</td><td style="font-weight:600;color:${statusColor};font-size:.75rem">${info.status}</td>`;
    tbody.appendChild(tr);
  });
  if(sorted.length === 0) tbody.innerHTML = '<tr><td colspan="3" style="color:#94a3b8;font-style:italic">No activity</td></tr>';

  // Sync popup on map
  setTimeout(() => {
    if(markers[currentDistrict] && !markers[currentDistrict].isPopupOpen())
      markers[currentDistrict].openPopup();
  }, 50);

  updateChart();
}

// ── Chart ─────────────────────────────────────────────────────────────────────
function updateChart() {
  const ctx = document.getElementById('seasonalChart').getContext('2d');
  if(chartInst){ chartInst.destroy(); chartInst = null; }

  if(currentDistrict === 'Palakkad' && DATA.prophet.length > 0) {
    document.getElementById('seasonalWarning').innerHTML =
      '<strong>AI Forecast (Prophet ML):</strong> Palakkad–Chikungunya predicted trend with 95% CI. Points above the upper band are anomalies.';
    const pd = DATA.prophet;
    chartInst = new Chart(ctx, {
      type:'line',
      data:{ labels: pd.map(d=>d.date), datasets:[
        {label:'Predicted',data:pd.map(d=>d.predicted),borderColor:'#3b82f6',borderWidth:2,pointRadius:0,tension:.3},
        {label:'Actual',data:pd.map(d=>d.actual),borderColor:'#94a3b8',borderWidth:0,pointRadius:2,showLine:false},
        {label:'Upper 95%',data:pd.map(d=>d.upper),borderColor:'rgba(59,130,246,.2)',backgroundColor:'rgba(59,130,246,.08)',borderWidth:1,pointRadius:0,fill:'-1'}
      ]},
      options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'bottom',labels:{font:{size:10}}}},scales:{x:{ticks:{maxTicksLimit:10,font:{size:10}}},y:{ticks:{font:{size:10}}}}}
    });
  } else {
    document.getElementById('seasonalWarning').textContent =
      'Historical monthly avg. (2018–2024). Peaks indicate seasonal risk windows for this district.';
    const sd = DATA.seasonal[currentDistrict] || {};
    const COLORS = ['#3b82f6','#ef4444','#10b981','#f59e0b','#8b5cf6','#ec4899','#64748b'];
    const datasets = [];
    let ci = 0;
    Object.entries(sd).forEach(([dis, avgs]) => {
      if(Math.max(...avgs) > 0.3) {
        datasets.push({label:dis, data:avgs, borderColor:COLORS[ci%COLORS.length], tension:.35, borderWidth:1.5, pointRadius:2});
        ci++;
      }
    });
    chartInst = new Chart(ctx, {
      type:'line',
      data:{ labels:['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], datasets },
      options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'bottom',labels:{font:{size:10}}}},scales:{x:{ticks:{font:{size:10}}},y:{ticks:{font:{size:10}}}}}
    });
  }
}

// ── Boot ──────────────────────────────────────────────────────────────────────
window.addEventListener('load', () => {
  // Force map to recalculate size after layout is stable
  setTimeout(() => {
    map.invalidateSize();
    updateMap();
    updateDetail();
  }, 300);
});
</script>
</body>
</html>"""

out_file = os.path.join(out_dir, "outbreak_dashboard.html")
with open(out_file, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Dashboard generated at {out_file}")
