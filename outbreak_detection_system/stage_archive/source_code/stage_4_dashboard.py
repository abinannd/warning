"""
Stage 4 Dashboard Builder — v4 (Full Kerala: 12 Districts)
Uses the integrated all-Kerala dataset from Stage 4.3.
Covers: 6 Malabar (North) + 6 South Kerala districts, 21 diseases, 2018-2025.
"""
import os, json, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

base_dir    = r"C:\BRAIN-STORM\HT\warning\outbreak_detection_system"
data_dir    = os.path.join(base_dir, "data", "processed")
int_dir     = os.path.join(base_dir, "reports", "kerala_integrated")
reports_dir = os.path.join(base_dir, "reports")
out_dir     = os.path.join(base_dir, "outputs")
os.makedirs(out_dir, exist_ok=True)

EPSILON      = 1e-6
MALABAR_DIST = {'Kannur','Kasaragod','Kozhikode','Malappuram','Palakkad','Wayanad'}
SOUTH_DIST   = {'Alappuzha','Idukki','Kollam','Kottayam','Pathanamthitta','Thiruvananthapuram'}

# ── 1. Load integrated daily dataset ─────────────────────────────────────────
print("Loading integrated dataset...")
df_all = pd.read_csv(os.path.join(int_dir, "kerala_integrated_daily.csv"),
                     parse_dates=["diagnosis_date"])
df_all["week"] = df_all["diagnosis_date"].dt.isocalendar().week.astype(int)
df_all["year"] = df_all["diagnosis_date"].dt.year

# Only use 2025 data for the weekly dashboard (test period)
df_2025 = df_all[df_all["year"] == 2025].copy()
# Fall back: if district has no 2025 data (South Kerala ends 2024), use 2024
df_2024 = df_all[df_all["year"] == 2024].copy()

# Unified districts and diseases
districts = sorted(df_all["district"].unique().tolist())
diseases  = sorted(df_all["disease_name"].unique().tolist())

print(f"  Districts ({len(districts)}): {districts}")
print(f"  Diseases  ({len(diseases)}): {diseases}")

# ── 2. Z-score on the full integrated series ──────────────────────────────────
print("Computing Z-scores...")

def classify_risk(z):
    return np.select(
        [z < 2.0, (z >= 2.0) & (z < 2.5), (z >= 2.5) & (z < 3.0), z >= 3.0],
        ["Low", "Medium", "High", "Critical"], default="Low"
    )

PRIORITY = {"None": 0, "Medium": 2, "High": 2, "Critical": 2}
stat_results = []
for (dist, dis), g in df_all.groupby(["district","disease_name"]):
    g = g.copy().sort_values("diagnosis_date").reset_index(drop=True)
    b_mean = g["case_count"].rolling(30, min_periods=15).mean().shift(8).fillna(0)
    b_std  = g["case_count"].rolling(30, min_periods=15).std().shift(8).fillna(0)
    r_mean = g["case_count"].rolling(7, min_periods=1).mean()
    std_safe = b_std.clip(lower=EPSILON)
    std_safe[b_std == 0] = np.nan
    g["z_score"]    = ((r_mean - b_mean) / std_safe).fillna(0)
    g["risk_level"] = classify_risk(g["z_score"])
    stat_results.append(g)

df_stat = pd.concat(stat_results, ignore_index=True)

# Assign tiers
df_stat["priority"] = 0
df_stat.loc[df_stat["risk_level"].isin(["Medium","High","Critical"]), "priority"] = 2

# Two-tier bundling on the stat results
ctr = 1
df_stat["tier"] = "None"
for (dist, dis), grp in df_stat.groupby(["district","disease_name"]):
    grp = grp.sort_values("diagnosis_date")
    in_ev, cur = False, None
    for idx, row in grp.iterrows():
        if row["risk_level"] != "Low":
            if not in_ev:
                in_ev = True
                cur = {"idxs": [idx], "peak": row["case_count"]}
            else:
                cur["idxs"].append(idx)
                cur["peak"] = max(cur["peak"], row["case_count"])
        else:
            if in_ev:
                in_ev = False
                if len(cur["idxs"]) >= 2:
                    tier = "Confirmed-Tier Event" if cur["peak"] >= 2 else "Watch-Tier Event"
                    df_stat.loc[cur["idxs"], "tier"] = tier
                cur = None
    if in_ev and cur and len(cur["idxs"]) >= 2:
        tier = "Confirmed-Tier Event" if cur["peak"] >= 2 else "Watch-Tier Event"
        df_stat.loc[cur["idxs"], "tier"] = tier

# ── 3. Build weekly warning table (2025 for Malabar, 2024 for South Kerala) ──
print("Building weekly warning table...")

# Combine: use 2025 for Malabar, 2024 for South Kerala (their latest year)
df_north = df_stat[(df_stat["district"].isin(MALABAR_DIST)) & (df_stat["year"] == 2025)].copy()
df_south = df_stat[(df_stat["district"].isin(SOUTH_DIST))   & (df_stat["year"] == 2024)].copy()
df_active = pd.concat([df_north, df_south], ignore_index=True)
df_active["week"] = df_active["diagnosis_date"].dt.isocalendar().week.astype(int)

# Use Malabar weeks as the timeline backbone
weeks = sorted(df_north["week"].unique().tolist())

STATUS_MAP = {4:"Emergency Warning",3:"Watch-Status Warning",2:"Advisory",1:"Normal",0:"Normal"}
COLOR_MAP  = {4:"red",3:"yellow",2:"yellow",1:"green",0:"green"}
REC_MAP    = {
    4:"Immediate public health intervention recommended. Deploy rapid response team.",
    3:"High sensitivity signal detected. Escalate local monitoring and testing.",
    2:"Elevated statistical activity. Review local clinic logs.",
    1:"Routine surveillance.",0:"Routine surveillance."
}

df_active["priority2"] = 0
df_active.loc[df_active["tier"] == "Confirmed-Tier Event", "priority2"] = 4
df_active.loc[df_active["tier"] == "Watch-Tier Event",     "priority2"] = 3
df_active.loc[(df_active["priority2"]==0) & (df_active["risk_level"]=="Critical"),"priority2"] = 2
df_active.loc[(df_active["priority2"]==0) & (df_active["risk_level"]=="High"),    "priority2"] = 2
df_active.loc[(df_active["priority2"]==0) & (df_active["risk_level"]=="Medium"),  "priority2"] = 2

agg = df_active.groupby(["week","district","disease_name"]).agg(
    cases=("case_count","sum"), priority=("priority2","max")
).reset_index()
agg = agg.sort_values(["week","district","priority","cases"], ascending=[True,True,False,False])
top = agg.groupby(["week","district"]).first().reset_index()

breakdown_map = {}
for (wk, dist), grp in agg.groupby(["week","district"]):
    breakdown_map[(wk,dist)] = {
        row.disease_name: {"cases":int(row.cases),"priority":int(row.priority),
                           "status":STATUS_MAP.get(int(row.priority),"Normal")}
        for _, row in grp.iterrows()
    }

weekly_warnings = {}
for wk in weeks:
    # Get date label from Malabar data
    wdata = df_north[df_north["week"] == wk]["diagnosis_date"]
    if wdata.empty:
        wdata = df_active[df_active["week"] == wk]["diagnosis_date"]
    start = wdata.min().strftime("%b %d")
    end   = wdata.max().strftime("%b %d, %Y") if not wdata.empty else ""
    label = f"{start} – {end}"
    wkey  = f"Week {wk}"
    weekly_warnings[wkey] = {"label": label}

    for dist in districts:
        # For South Kerala, find the matching week by month/week number in 2024
        if dist in SOUTH_DIST:
            row = top[(top["week"] == wk) & (top["district"] == dist)]
        else:
            row = top[(top["week"] == wk) & (top["district"] == dist)]

        if row.empty:
            p, dis, cases = 0, "-", 0
        else:
            r = row.iloc[0]
            p, dis, cases = int(r["priority"]), r["disease_name"] if int(r["priority"]) > 0 else "-", int(r["cases"])

        weekly_warnings[wkey][dist] = {
            "status": STATUS_MAP[p], "disease": dis, "cases": cases,
            "color": COLOR_MAP[p], "recommendation": REC_MAP[p],
            "region": "North Kerala" if dist in MALABAR_DIST else "South Kerala",
            "breakdown": breakdown_map.get((wk, dist), {})
        }

# ── 4. Seasonal profiles (all districts, all diseases) ────────────────────────
print("Building seasonal profiles...")
df_train_raw = df_all[df_all["year"] < 2025].copy()
df_train_raw["month"] = df_train_raw["diagnosis_date"].dt.month
monthly = df_train_raw.groupby(["district","disease_name","month"])["case_count"].mean().reset_index()
max_vals = monthly.groupby(["district","disease_name"])["case_count"].max()

seasonal_history = {}
for dist in districts:
    seasonal_history[dist] = {}
    for dis in diseases:
        key = (dist, dis)
        if key not in max_vals or max_vals[key] < 0.05:
            continue
        sub = monthly[(monthly["district"]==dist) & (monthly["disease_name"]==dis)]
        full = sub.set_index("month")["case_count"].reindex(range(1,13), fill_value=0)
        seasonal_history[dist][dis] = [round(float(v),4) for v in full.tolist()]

# ── 5. Prophet predictions (Palakkad-Chikungunya) ────────────────────────────
print("Loading Prophet predictions...")
prophet_data = []
pcsv = os.path.join(reports_dir, "prophet_predictions_palakkad_chikungunya.csv")
if os.path.exists(pcsv):
    df_pro = pd.read_csv(pcsv).iloc[::3].copy()
    df_pro["date"] = pd.to_datetime(df_pro["date"]).dt.strftime("%b %d")
    for _, r in df_pro.iterrows():
        prophet_data.append({
            "date": r["date"], "actual": float(r["actual"]),
            "predicted": round(float(r["predicted"]),4),
            "upper": round(float(r["upper_95"]),4),
            "anomaly": bool(r["anomaly_high"])
        })

# ── 6. Best default week ──────────────────────────────────────────────────────
confirmed_by_week = df_north[df_north["tier"]=="Confirmed-Tier Event"].groupby("week").size()
default_week_num  = int(confirmed_by_week.idxmax()) if not confirmed_by_week.empty else weeks[0]
default_week      = f"Week {default_week_num}"
print(f"Default week: {default_week} ({weekly_warnings[default_week]['label']})")

# ── 7. Assemble payload ───────────────────────────────────────────────────────
payload = {
    "weeks":        [f"Week {w}" for w in weeks],
    "default_week": default_week,
    "districts":    districts,
    "malabar_districts": sorted(MALABAR_DIST),
    "south_districts":   sorted(SOUTH_DIST),
    "warnings":     weekly_warnings,
    "seasonal":     seasonal_history,
    "prophet":      prophet_data
}

DATA_JSON = json.dumps(payload, ensure_ascii=True, separators=(",",":"))
print(f"JSON payload size: {len(DATA_JSON)//1024} KB")

# ── 8. HTML ────────────────────────────────────────────────────────────────────
print("Generating HTML dashboard...")

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Kerala Outbreak Early Warning Dashboard</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',sans-serif;background:#f1f5f9;color:#1e293b;font-size:14px}
.shell{max-width:1500px;margin:0 auto;padding:16px}
/* Topbar */
.topbar{background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%);color:#f8fafc;padding:14px 22px;border-radius:12px;margin-bottom:14px;display:flex;align-items:center;gap:20px;flex-wrap:wrap}
.brand{font-size:1.1rem;font-weight:700;flex:1;letter-spacing:-.01em}
.brand span{color:#38bdf8}
.tag{font-size:.7rem;padding:2px 8px;border-radius:4px;font-weight:600;margin-left:8px;vertical-align:middle}
.tag-north{background:rgba(56,189,248,.2);color:#38bdf8}
.tag-south{background:rgba(251,191,36,.2);color:#fbbf24}
.legend-row{display:flex;gap:14px;flex-wrap:wrap;align-items:center}
.leg{display:flex;align-items:center;gap:5px;font-size:.73rem;font-weight:500;color:#cbd5e1}
.dot{width:11px;height:11px;border-radius:50%;flex-shrink:0}
.dot-s{width:11px;height:11px;border-radius:50%;border:2px dashed #38bdf8;flex-shrink:0}
/* Controls */
.controls{background:#fff;border-radius:10px;padding:10px 16px;margin-bottom:14px;display:flex;align-items:center;gap:12px;flex-wrap:wrap;box-shadow:0 1px 3px rgba(0,0,0,.07)}
.controls label{font-weight:600;font-size:.85rem;color:#475569;white-space:nowrap}
select{border:1px solid #cbd5e1;border-radius:8px;padding:7px 12px;font-size:.85rem;background:#f8fafc;color:#0f172a;cursor:pointer;min-width:200px}
select:focus{outline:none;border-color:#3b82f6}
.stat-pills{display:flex;gap:10px;flex-wrap:wrap;margin-left:auto}
.pill{padding:5px 12px;border-radius:8px;font-size:.75rem;font-weight:600}
.pill-red{background:#fef2f2;color:#b91c1c}
.pill-yellow{background:#fffbeb;color:#854d0e}
.pill-green{background:#f0fdf4;color:#166534}
/* Grid */
.grid{display:grid;grid-template-columns:3fr 2fr;gap:14px}
@media(max-width:860px){.grid{grid-template-columns:1fr}}
#map{width:100%;height:620px;border-radius:12px;box-shadow:0 1px 4px rgba(0,0,0,.1)}
/* Panels */
.panel{background:#fff;border-radius:12px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,.06);margin-bottom:12px}
.panel-h{font-size:.9rem;font-weight:700;color:#0f172a;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid #f1f5f9;display:flex;align-items:center;gap:8px}
.badge{display:inline-block;padding:3px 10px;border-radius:9999px;font-size:.72rem;font-weight:700}
.badge-red{background:#fef2f2;color:#b91c1c}
.badge-yellow{background:#fffbeb;color:#854d0e}
.badge-green{background:#f0fdf4;color:#166534}
.badge-gray{background:#f1f5f9;color:#475569}
.badge-blue{background:#eff6ff;color:#1d4ed8}
.dh{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;gap:8px}
.dh-left h3{font-size:1.15rem;font-weight:700}
.dh-left .region-tag{font-size:.7rem;color:#64748b;font-weight:500;margin-top:2px}
.ml{font-size:.7rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px}
.mv{font-size:.9rem;font-weight:600;color:#0f172a}
.mb12{margin-bottom:12px}
.rec{font-size:.8rem;color:#475569;line-height:1.55}
table{width:100%;border-collapse:collapse;font-size:.76rem}
th{text-align:left;padding:5px 7px;background:#f8fafc;color:#475569;font-weight:600;border-bottom:1px solid #e2e8f0}
td{padding:4px 7px;border-bottom:1px solid #f8fafc;color:#334155}
tr:hover td{background:#f8fafc}
.chart-wrap{position:relative;height:180px;margin-top:8px}
.disclaimer{font-size:.68rem;color:#94a3b8;font-style:italic;margin-top:12px;line-height:1.6;padding:0 2px}

/* Alert Previews */
.preview-container { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 8px; }
.wa-preview, .email-preview { flex: 1; min-width: 200px; }
.wa-header, .email-header { font-size: 0.72rem; color: #64748b; font-weight: 600; text-transform: uppercase; margin-bottom: 6px; }
.wa-bubble { background: #dcf8c6; border-radius: 8px 8px 8px 0; padding: 10px; color: #1e293b; font-size: 0.75rem; line-height: 1.4; box-shadow: 0 1px 2px rgba(0,0,0,0.1); position: relative; white-space: pre-wrap; }
.wa-bubble::before { content: ""; position: absolute; left: -6px; bottom: 0; width: 0; height: 0; border: 6px solid transparent; border-right-color: #dcf8c6; border-bottom-color: #dcf8c6; }
.email-box { background: #fff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px; color: #334155; font-size: 0.75rem; line-height: 1.5; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
.email-subject { font-weight: 600; margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid #f1f5f9; color: #0f172a; }
/* Popup */
.lp{font-family:'Inter',sans-serif;min-width:180px;font-size:.8rem;line-height:1.4}
.lp b{font-size:.88rem;display:block;margin-bottom:2px}
.lp .rg{font-size:.68rem;color:#94a3b8;margin-bottom:4px}
.lp .st{font-weight:700;margin-bottom:4px}
.lp .tr{font-size:.74rem;color:#64748b;margin-bottom:5px}
.lp hr{border:none;border-top:1px solid #e2e8f0;margin:5px 0}
.lp .rc{font-size:.72rem;color:#475569;font-style:italic}
/* Disclaimer Banner */
#disclaimerBanner { background-color: #fef5d9; border: 1px solid #fce895; border-radius: 8px; margin-bottom: 14px; color: #856404; font-size: 0.82rem; line-height: 1.4; transition: all 0.2s; overflow: hidden; }
.banner-header { display: flex; align-items: center; padding: 10px 15px; font-weight: 600; cursor: pointer; }
.banner-title { flex-grow: 1; margin-left: 6px; }
.banner-toggle { background: none; border: none; color: #856404; font-size: 1.1rem; cursor: pointer; font-weight: bold; }
.banner-body { padding: 0 15px 12px 15px; }
.banner-collapsed .banner-body { display: none; }
.banner-collapsed .banner-header { padding: 8px 15px; }
</style>
</head>
<body>
<div class="shell">

  <!-- Topbar -->
  <div class="topbar">
    <div class="brand">
      &#x1F9A0; Kerala Outbreak Early Warning Dashboard
      <span class="tag tag-north">North Kerala</span>
      <span class="tag tag-south">South Kerala</span>
    </div>
    <div class="legend-row">
      <div class="leg"><div class="dot" style="background:#c0392b"></div>Emergency</div>
      <div class="leg"><div class="dot" style="background:#e6a817"></div>Watch/Advisory</div>
      <div class="leg"><div class="dot" style="background:#4a9d5f"></div>Normal</div>
      <div class="leg"><div class="dot-s"></div>AI Seasonal Watch</div>
    </div>
  </div>

  <!-- Warning Banner -->
  <div id="disclaimerBanner" class="">
    <div class="banner-header" onclick="document.getElementById('disclaimerBanner').classList.toggle('banner-collapsed')">
      <span style="font-size: 1.1rem;">&#9888;</span>
      <span class="banner-title">Important: Dashboard Usage Disclaimer</span>
      <button class="banner-toggle">&#10005;</button>
    </div>
    <div class="banner-body">
      This dashboard provides statistical early-warning signals based on recent surveillance data trends. It does not predict outbreaks with certainty and should not be used as the sole basis for medical or public health decisions. Risk levels reflect deviations from historical statistical baselines, not confirmed epidemiological investigations. For official guidance, contact your local health authority.
    </div>
  </div>

  <!-- Controls + Stats Bar -->
  <div class="controls">
    <label>&#x1F4C5;</label>
    <select id="weekSelect"></select>
    <div class="stat-pills">
      <span class="pill pill-red" id="statRed">&#x1F534; 0 Emergency</span>
      <span class="pill pill-yellow" id="statYellow">&#x1F7E1; 0 Watch</span>
      <span class="pill pill-green" id="statGreen">&#x1F7E2; 0 Normal</span>
    </div>
  </div>

  <!-- Main Grid -->
  <div class="grid">
    <div>
      <div class="panel" style="padding-bottom:10px">
        <div class="panel-h">&#x1F5FA;&#xFE0F; Geographic Risk Map &mdash; Kerala (12 Districts)</div>
        <div id="map"></div>
      </div>
    </div>
    <div>
      <div class="panel">
        <div class="dh">
          <div class="dh-left">
            <h3 id="detailTitle">Palakkad</h3>
            <div class="region-tag" id="detailRegion">North Kerala (Malabar)</div>
          </div>
          <span id="detailBadge" class="badge badge-gray">Normal</span>
        </div>
        <div class="mb12"><div class="ml">Triggering Disease</div><div class="mv" id="detailDisease">—</div></div>
        <div class="mb12"><div class="ml">Cases This Period</div><div class="mv" id="detailCases">—</div></div>
        <div class="mb12"><div class="ml">Recommended Action</div><div class="rec" id="detailAction">—</div></div>
        <div>
          <div class="ml">Disease Breakdown</div>
          <table><thead><tr><th>Disease</th><th>Cases</th><th>Status</th></tr></thead>
          <tbody id="breakdownBody"></tbody></table>
        </div>
      </div>
      <div class="panel">
        <div class="panel-h">&#x1F4C8; Seasonal Risk Pattern</div>
        <p id="seasonalWarning" style="font-size:.76rem;color:#475569;line-height:1.5;margin-bottom:8px"></p>
        <div class="chart-wrap"><canvas id="seasonalChart"></canvas></div>
      </div>
    </div>
  </div>

  <p class="disclaimer">
    &#x26A0;&#xFE0F; Disclaimer: Statistical early warnings based on surveillance data (2018&ndash;2025 Malabar / 2020&ndash;2024 South Kerala).
    Alerts are generated by a gap-corrected Z-score engine with Prophet ML forecasting (Palakkad&ndash;Chikungunya).
    North Kerala weekly period shown; South Kerala reflects most recent available period (2024).
    Not a substitute for official health authority guidance.
  </p>
</div>

<script>
const DATA = """ + DATA_JSON + """;

const COORDS = {
  // North Kerala (Malabar)
  'Kannur':             [11.8745, 75.3704],
  'Kasaragod':          [12.4996, 74.9869],
  'Kozhikode':          [11.2588, 75.7804],
  'Malappuram':         [11.0410, 76.0788],
  'Palakkad':           [10.7867, 76.6548],
  'Wayanad':            [11.6854, 76.1320],
  // South Kerala
  'Alappuzha':          [ 9.4981, 76.3388],
  'Idukki':             [ 9.8560, 76.9744],
  'Kollam':             [ 8.8932, 76.6141],
  'Kottayam':           [ 9.5916, 76.5222],
  'Pathanamthitta':     [ 9.2648, 76.7870],
  'Thiruvananthapuram': [ 8.5241, 76.9366]
};

const COLOR = {red:'#c0392b', yellow:'#e6a817', green:'#4a9d5f'};
const BADGE = {red:'badge-red', yellow:'badge-yellow', green:'badge-green'};
const PCOL  = {4:'#b91c1c',3:'#92400e',2:'#854d0e',1:'#166534',0:'#94a3b8'};
const CHART_COLORS = ['#3b82f6','#ef4444','#10b981','#f59e0b','#8b5cf6','#ec4899','#14b8a6','#64748b'];

let currentWeek = DATA.default_week;
let currentDist = 'Palakkad';
let chartInst   = null;
let markers     = {};
let lblMarkers  = [];

// ── Map init ──────────────────────────────────────────────────────────────────
// Center on all of Kerala
const map = L.map('map',{zoomControl:true}).setView([10.5, 76.5], 7);
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
sel.addEventListener('change', e => {
  currentWeek = e.target.value;
  updateMap(); updateDetail();
});

// ── Stats pills ───────────────────────────────────────────────────────────────
function updatePills() {
  const wd = DATA.warnings[currentWeek];
  let r=0, y=0, g=0;
  DATA.districts.forEach(d => {
    const c = wd[d].color;
    if(c==='red') r++; else if(c==='yellow') y++; else g++;
  });
  document.getElementById('statRed').textContent    = `\uD83D\uDD34 ${r} Emergency`;
  document.getElementById('statYellow').textContent = `\uD83D\uDFE1 ${y} Watch`;
  document.getElementById('statGreen').textContent  = `\uD83D\uDFE2 ${g} Normal`;
}

// ── Map markers ───────────────────────────────────────────────────────────────
function updateMap() {
  Object.values(markers).forEach(m => map.removeLayer(m));
  lblMarkers.forEach(m => map.removeLayer(m));
  markers = {}; lblMarkers = [];

  const wd = DATA.warnings[currentWeek];

  DATA.districts.forEach(dist => {
    const d   = wd[dist];
    const hex = COLOR[d.color];
    const isNorth    = DATA.malabar_districts.includes(dist);
    const isSeasonal = (dist === 'Palakkad');
    const isSel      = (dist === currentDist);

    const m = L.circleMarker(COORDS[dist], {
      radius:      isSel ? 22 : 17,
      fillColor:   hex,
      fillOpacity: 0.88,
      color:       isSeasonal ? '#38bdf8' : (isSel ? '#fff' : isNorth ? '#fff' : '#fbbf24'),
      weight:      isSeasonal ? 3 : (isSel ? 3 : isNorth ? 1.5 : 2),
      dashArray:   isSeasonal ? '6 4' : ''
    }).addTo(map);

    const trigger = (d.disease !== '-') ? `<div class="tr">&#x26A0; ${d.disease} &mdash; ${d.cases} case(s)</div>` : '';
    const regionLbl = isNorth ? 'North Kerala (Malabar)' : 'South Kerala';
    m.bindPopup(`<div class="lp">
      <b>${dist}</b><div class="rg">${regionLbl}</div>
      <div class="st" style="color:${hex}">${d.status}</div>
      ${trigger}
      <hr/><div class="rc">${d.recommendation}</div>
    </div>`, {maxWidth:250, autoPan:true});

    m.on('click', () => { currentDist = dist; updateMap(); updateDetail(); });
    markers[dist] = m;

    // District label
    const icon = L.divIcon({
      className:'',
      html:`<span style="font:700 9.5px/1 Inter,sans-serif;color:#0f172a;text-shadow:0 0 3px #fff,0 0 3px #fff,0 0 3px #fff;white-space:nowrap;pointer-events:none">${dist}</span>`,
      iconAnchor:[0,-24]
    });
    const lm = L.marker(COORDS[dist],{icon,interactive:false,zIndexOffset:1000}).addTo(map);
    lblMarkers.push(lm);
  });

  updatePills();
  setTimeout(() => { if(markers[currentDist]) markers[currentDist].openPopup(); }, 700);
}

// ── Detail panel ──────────────────────────────────────────────────────────────
function updateDetail() {
  const d = DATA.warnings[currentWeek][currentDist];

  document.getElementById('detailTitle').textContent  = currentDist;
  document.getElementById('detailRegion').textContent = d.region;

  const badge = document.getElementById('detailBadge');
  badge.textContent = d.status;
  badge.className   = 'badge ' + (BADGE[d.color] || 'badge-gray');

  document.getElementById('detailDisease').textContent = d.disease !== '-' ? d.disease : 'None active';
  const isSouth = DATA.south_districts.includes(currentDist);
  document.getElementById('detailCases').textContent  =
    `${d.cases > 0 ? d.cases : 0} case(s)` + (isSouth ? ' (2024 data)' : '');
  document.getElementById('detailAction').textContent = d.recommendation;

  // Breakdown table
  const tbody = document.getElementById('breakdownBody');
  tbody.innerHTML = '';
  const entries = Object.entries(d.breakdown||{}).sort((a,b)=>b[1].cases-a[1].cases);
  if(!entries.length) {
    tbody.innerHTML = '<tr><td colspan="3" style="color:#94a3b8;font-style:italic;padding:8px">No recorded activity</td></tr>';
  } else {
    entries.forEach(([dis,info]) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${dis}</td><td style="font-weight:600">${info.cases}</td><td style="font-weight:600;color:${PCOL[info.priority]||'#94a3b8'};font-size:.7rem">${info.status}</td>`;
      tbody.appendChild(tr);
    });
  }

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

  const opts = {
    responsive:true, maintainAspectRatio:false,
    plugins:{legend:{position:'bottom',labels:{font:{size:9},boxWidth:10,padding:8}}},
    scales:{x:{ticks:{font:{size:9}}},y:{ticks:{font:{size:9}},beginAtZero:true}}
  };

  if(currentDist === 'Palakkad' && DATA.prophet.length > 0) {
    document.getElementById('seasonalWarning').innerHTML =
      '<strong>Prophet ML Forecast (Palakkad &mdash; Chikungunya):</strong> Predicted baseline (blue) with 95% CI (shaded). Grey dots = actual cases.';
    const pd = DATA.prophet;
    chartInst = new Chart(ctx, {type:'line', data:{
      labels: pd.map(d=>d.date),
      datasets:[
        {label:'Predicted',data:pd.map(d=>d.predicted),borderColor:'#3b82f6',borderWidth:2,pointRadius:0,tension:.3,fill:false},
        {label:'Actual',   data:pd.map(d=>d.actual),   borderColor:'#94a3b8',borderWidth:0,pointRadius:2,showLine:false},
        {label:'Upper 95%',data:pd.map(d=>d.upper),    borderColor:'rgba(59,130,246,.25)',backgroundColor:'rgba(59,130,246,.08)',borderWidth:1,pointRadius:0,fill:'-1'}
      ]}, options:opts});
  } else {
    const isSouth = DATA.south_districts.includes(currentDist);
    document.getElementById('seasonalWarning').textContent =
      isSouth
        ? `Historical monthly avg. (2020\u20132024, South Kerala). Peaks show seasonal disease risk windows for ${currentDist}.`
        : `Historical monthly avg. (2018\u20132024, Malabar). Peaks show seasonal disease risk windows for ${currentDist}.`;

    const sd = DATA.seasonal[currentDist] || {};
    const sets = [];
    let ci = 0;
    const byAct = Object.entries(sd).sort((a,b)=>b[1].reduce((s,x)=>s+x,0)-a[1].reduce((s,x)=>s+x,0));
    byAct.forEach(([dis,avgs]) => {
      if(Math.max(...avgs) < 0.05) return;
      sets.push({label:dis,data:avgs,borderColor:CHART_COLORS[ci%CHART_COLORS.length],tension:.35,borderWidth:1.8,pointRadius:2,fill:false});
      ci++;
    });
    chartInst = new Chart(ctx, {type:'line', data:{
      labels:['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
      datasets:sets}, options:opts});
  }

  // Alert Previews
  const wa = document.getElementById('waMessage');
  const eSub = document.getElementById('emailSubject');
  const eBod = document.getElementById('emailBody');
  const d = DATA.warnings[currentWeek][currentDist];
  if (d.status === 'Normal') {
    wa.innerText = `No unusual disease activity detected in ${currentDist} this period. Routine surveillance continues.`;
    eSub.innerText = `Health Update \u2014 ${currentDist} District, Kerala`;
    eBod.innerHTML = `Dear Resident,<br><br>No unusual disease activity detected in ${currentDist} this period. Routine surveillance continues.<br><br>Kerala Health Department \u2014 Early Warning System`;
  } else {
    const dis = d.disease !== '-' ? d.disease : 'Unknown';
    const rec = d.recommendation;
    const cases = `${d.cases > 0 ? d.cases : 0}`;
    wa.innerText = `[Kerala Health Alert] ${currentDist}: ${d.status} for ${dis}.\n${cases} case(s) reported this period.\nAction: ${rec.split('.')[0]}.\nStay alert. For guidance, contact your local health center.`;
    eSub.innerText = `Health Advisory \u2014 ${currentDist} District, Kerala`;
    eBod.innerHTML = `Dear Resident,<br><br>The Kerala Disease Surveillance System has issued a <strong>${d.status}</strong> for <strong>${dis}</strong> in ${currentDist} district.<br><br>Cases reported this period: ${cases}<br>Recommended action: ${rec}<br><br>This is an automated statistical advisory based on recent surveillance trends. For medical guidance, please consult your local health authority or nearest primary health center.<br><br>Kerala Health Department \u2014 Early Warning System`;
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
# Encode to bytes, replacing any lone surrogates, then decode back to a clean string
html_clean = html.encode("utf-8", errors="replace").decode("utf-8")
with open(out_file, "w", encoding="utf-8") as f:
    f.write(html_clean)

print(f"Dashboard saved: {out_file}")
print(f"Default week: {default_week} ({weekly_warnings[default_week]['label']})")
print("Done.")
