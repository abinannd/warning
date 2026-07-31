"""
Stage 4 Dashboard Builder — v4 (With Hackathon Multi-Channel Notification Simulator)
- Vectorized weekly warning computation
- Seasonal profiles with sparsity handling
- Interactive Leaflet map & Chart.js metrics
- Citizen Alert Simulation: SMS Preview Card, WhatsApp Preview Card, Dispatch Alert Modal, Webhook API support & Audit Trail
"""
import os
import json
import pandas as pd
import numpy as np

script_dir  = os.path.dirname(os.path.abspath(__file__))
base_dir    = os.path.abspath(os.path.join(script_dir, "..", ".."))
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

df["priority"] = 0
df.loc[df["tier"] == "Confirmed-Tier Event", "priority"] = 4
df.loc[df["tier"] == "Watch-Tier Event",     "priority"] = 3
df.loc[(df["priority"] == 0) & (df["risk_level"] == "Critical"), "priority"] = 2
df.loc[(df["priority"] == 0) & (df["risk_level"] == "High"),     "priority"] = 2
df.loc[(df["priority"] == 0) & (df["risk_level"] == "Medium"),   "priority"] = 2

grp = df.groupby(["week", "district", "disease_name"]).agg(
    total_cases=("cases", "sum"),
    max_priority=("priority", "max")
).reset_index()

max_pri = grp.groupby(["week", "district"])["max_priority"].max().reset_index()
max_pri = max_pri.rename(columns={"max_priority": "district_priority"})

grp = grp.merge(max_pri, on=["week", "district"])
triggering = grp[grp["max_priority"] == grp["district_priority"]]
trigger_dis = triggering.groupby(["week", "district"])["disease_name"].apply(lambda x: ", ".join(x)).reset_index()

dist_cases = grp.groupby(["week", "district"])["total_cases"].sum().reset_index()
dist_summary = max_pri.merge(dist_cases, on=["week", "district"]).merge(trigger_dis, on=["week", "district"], how="left")

dist_summary["disease_name"] = dist_summary["disease_name"].fillna("-")
dist_summary.loc[dist_summary["district_priority"] == 0, "disease_name"] = "-"

warnings_payload = {}
for w in weeks:
    w_df = dist_summary[dist_summary["week"] == w]
    sub  = grp[grp["week"] == w]
    
    if w_df.empty:
        continue

    d0 = df[df["week"] == w]["diagnosis_date"].min()
    d1 = df[df["week"] == w]["diagnosis_date"].max()
    date_label = f"{d0.strftime('%b %d')} – {d1.strftime('%b %d, %Y')}" if pd.notna(d0) else f"Week {w}"

    w_dict = {"label": date_label}
    for _, row in w_df.iterrows():
        dist = row["district"]
        prio = int(row["district_priority"])
        
        bd_df = sub[sub["district"] == dist].sort_values("total_cases", ascending=False)
        breakdown = {}
        for _, brow in bd_df.iterrows():
            bprio = int(brow["max_priority"])
            breakdown[brow["disease_name"]] = {
                "cases":    int(brow["total_cases"]),
                "priority": bprio,
                "status":   STATUS_MAP[bprio]
            }

        w_dict[dist] = {
            "status":         STATUS_MAP[prio],
            "disease":        row["disease_name"],
            "cases":          int(row["total_cases"]),
            "color":          COLOR_MAP[prio],
            "recommendation": REC_MAP[prio],
            "breakdown":      breakdown
        }
    warnings_payload[f"Week {w}"] = w_dict

# ── 3. Find most active week as default ──────────────────────────────────────
active_counts = {
    w: sum(1 for d in districts if warnings_payload.get(w, {}).get(d, {}).get("color") in ["red", "yellow"])
    for w in warnings_payload.keys()
}
default_week = max(active_counts, key=active_counts.get) if active_counts else f"Week {weeks[0]}"
print(f"Default active week: {default_week} ({active_counts[default_week]} active warnings)")

# ── 4. Seasonal profiles ─────────────────────────────────────────────────────
print("Computing seasonal profiles...")
train_df = df[df["diagnosis_date"].dt.year < 2024].copy()
train_df["month"] = train_df["diagnosis_date"].dt.month

seasonal_history = {}
all_diseases = sorted(df["disease_name"].unique().tolist())

for dist in districts:
    dist_df = train_df[train_df["district"] == dist]
    dist_seasonal = {}
    for dis in all_diseases:
        dis_df = dist_df[dist_df["disease_name"] == dis]
        if dis_df.empty:
            dist_seasonal[dis] = [0.0]*12
            continue
        monthly = dis_df.groupby(["year", "month"])["cases"].sum().reset_index()
        grid = pd.MultiIndex.from_product(
            [monthly["year"].unique(), range(1, 13)],
            names=["year", "month"]
        ).to_frame().reset_index(drop=True)
        full_m = grid.merge(monthly, on=["year", "month"], how="left").fillna(0)
        means  = full_m.groupby("month")["cases"].mean().reindex(range(1, 13), fill_value=0.0).round(2).tolist()
        dist_seasonal[dis] = means
    seasonal_history[dist] = dist_seasonal

# ── 5. Prophet forecast (Palakkad – Chikungunya) ────────────────────────────
print("Loading Prophet predictions...")
prophet_data = []
pr_path = os.path.join(reports_dir, "prophet_test_eval_palakkad_chikungunya.csv")
if os.path.exists(pr_path):
    pr_df = pd.read_csv(pr_path)
    pr_df["ds"] = pd.to_datetime(pr_df["ds"])
    for _, r in pr_df.iterrows():
        prophet_data.append({
            "date":      r["ds"].strftime("%Y-%m-%d"),
            "actual":    round(float(r["y"]), 2),
            "predicted": round(float(r["yhat"]), 2),
            "upper":     round(float(r["yhat_upper"]), 2),
            "lower":     round(float(r["yhat_lower"]), 2)
        })

# ── 6. Assemble payload JSON ──────────────────────────────────────────────────
payload = {
    "weeks":        [f"Week {w}" for w in weeks],
    "default_week": default_week,
    "districts":    districts,
    "warnings":     warnings_payload,
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

/* Notification Simulator Styles */
.nav-tab-container{display:flex;gap:8px;margin-bottom:12px;border-bottom:1px solid #e2e8f0;padding-bottom:6px}
.nav-tab{background:#f1f5f9;border:none;border-radius:6px;padding:6px 12px;font-size:.78rem;font-weight:600;color:#64748b;cursor:pointer;transition:all .15s ease}
.nav-tab.active{background:#0f172a;color:#fff}
.nav-tab:hover:not(.active){background:#e2e8f0;color:#1e293b}

.alert-card{border-radius:10px;padding:12px 14px;font-size:.82rem;font-family:'Inter',sans-serif;margin-bottom:12px;box-shadow:0 1px 3px rgba(0,0,0,.05)}

/* SMS Card */
.sms-card{background:#0f172a;color:#f8fafc;border:1px solid #334155}
.phone-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;font-size:.7rem;color:#94a3b8;font-weight:600;border-bottom:1px solid #1e293b;padding-bottom:4px}
.sender-tag{color:#38bdf8;font-weight:700}
.sms-body{font-family:'Segoe UI',Tahoma,sans-serif;background:#1e293b;padding:10px 12px;border-radius:8px;line-height:1.45;border-left:3px solid #38bdf8;font-size:.8rem;margin-bottom:8px;color:#f1f5f9;word-break:break-word}
.sms-footer{display:flex;justify-content:space-between;align-items:center;font-size:.7rem;color:#64748b}
.sim-badge{background:rgba(56,189,248,.15);color:#38bdf8;padding:2px 6px;border-radius:4px;font-weight:600}

/* WhatsApp Card */
.wa-card{background:#efeae2;border:1px solid #cbd5e1;color:#111b21}
.wa-header{display:flex;align-items:center;gap:8px;background:#075e54;color:#fff;padding:8px 12px;border-radius:8px 8px 0 0;margin:-12px -14px 10px -14px}
.wa-avatar{width:28px;height:28px;background:#128c7e;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:14px}
.wa-name{font-weight:700;font-size:.82rem}
.v-check{color:#38bdf8;font-size:.75rem}
.wa-sub{font-size:.68rem;opacity:.85}
.wa-bubble{background:#dcf8c6;border-radius:8px;padding:10px 12px;box-shadow:0 1px 1px rgba(0,0,0,.13);position:relative}
.wa-title{font-weight:700;color:#075e54;font-size:.78rem;margin-bottom:4px}
.wa-text{font-size:.79rem;line-height:1.4;color:#111b21;white-space:pre-line}
.wa-time{font-size:.65rem;color:#667781;text-align:right;margin-top:4px;font-weight:500}
.blue-ticks{color:#34b7f1;font-weight:bold}
.wa-actions{display:flex;gap:6px;margin-top:8px}
.wa-btn{flex:1;text-align:center;background:#fff;border:1px solid #128c7e;color:#128c7e;padding:6px;border-radius:6px;font-size:.72rem;font-weight:600;cursor:pointer;transition:background .15s}
.wa-btn:hover{background:#e8f5e9}
.wa-btn.primary{background:#128c7e;color:#fff}
.wa-btn.primary:hover{background:#075e54}

/* Modal Styles */
.modal-overlay{position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(15,23,42,.6);backdrop-filter:blur(3px);z-index:9999;display:flex;align-items:center;justify-content:center;opacity:0;pointer-events:none;transition:opacity .2s ease}
.modal-overlay.open{opacity:1;pointer-events:auto}
.modal-box{background:#fff;border-radius:14px;width:90%;max-width:480px;padding:20px 24px;box-shadow:0 20px 25px -5px rgba(0,0,0,.2);transform:scale(.95);transition:transform .2s ease}
.modal-overlay.open .modal-box{transform:scale(1)}
.modal-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;padding-bottom:10px;border-bottom:1px solid #e2e8f0}
.modal-header h3{font-size:1.05rem;font-weight:700;color:#0f172a}
.close-modal{background:none;border:none;font-size:1.3rem;color:#64748b;cursor:pointer}
.modal-body{font-size:.83rem;color:#334155;margin-bottom:16px}
.form-group{margin-bottom:12px}
.form-group label{display:block;font-weight:600;font-size:.78rem;color:#475569;margin-bottom:4px}
.form-group input,.form-group select{width:100%;padding:8px 10px;border:1px solid #cbd5e1;border-radius:8px;font-size:.82rem}
.chk-group{display:flex;flex-direction:column;gap:6px;background:#f8fafc;padding:10px;border-radius:8px;border:1px solid #e2e8f0}
.chk-group label{font-weight:500;font-size:.8rem;color:#1e293b;display:flex;align-items:center;gap:8px;cursor:pointer}

/* Toast Notifications */
.toast-container{position:fixed;top:20px;right:20px;z-index:10000;display:flex;flex-direction:column;gap:8px;pointer-events:none}
.toast{background:#0f172a;color:#fff;padding:12px 18px;border-radius:10px;font-size:.82rem;font-weight:500;box-shadow:0 10px 15px -3px rgba(0,0,0,.3);border-left:4px solid #10b981;display:flex;align-items:center;gap:10px;transform:translateX(120%);transition:transform .3s cubic-bezier(.16,1,.3,1);pointer-events:auto}
.toast.show{transform:translateX(0)}
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

      <!-- Hackathon Notification Simulator Panel -->
      <div class="panel" id="notificationPanel">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid #f1f5f9;">
          <h2 style="margin:0;border:none;padding:0;">📱 Citizen Alert Simulation</h2>
          <button id="dispatchAlertBtn" onclick="openAlertModal()" style="background:#ef4444;color:#fff;border:none;border-radius:8px;padding:6px 14px;font-weight:600;font-size:.78rem;cursor:pointer;display:flex;align-items:center;gap:6px;box-shadow:0 2px 4px rgba(239,68,68,.25);">
            <span>⚡ Send Alert Broadcast</span>
          </button>
        </div>
        
        <div class="nav-tab-container">
          <button class="nav-tab active" id="tabSmsBtn" onclick="switchAlertTab('sms')">💬 SMS Preview</button>
          <button class="nav-tab" id="tabWaBtn" onclick="switchAlertTab('wa')">🟢 WhatsApp Preview</button>
        </div>

        <!-- SMS Card Preview -->
        <div id="smsCard" class="alert-card sms-card">
          <div class="phone-header">
            <span class="sender-tag">📩 +91-DPH-ALERT (Kerala Health)</span>
            <span class="time-tag">Just Now</span>
          </div>
          <div class="sms-body" id="smsBodyText">
            Loading SMS alert payload...
          </div>
          <div class="sms-footer">
            <span id="smsCharStats">168 chars | 2 SMS segments</span>
            <span class="sim-badge">Hackathon Demo Mode</span>
          </div>
        </div>

        <!-- WhatsApp Card Preview -->
        <div id="waCard" class="alert-card wa-card" style="display:none;">
          <div class="wa-header">
            <div class="wa-avatar">🟢</div>
            <div>
              <div class="wa-name">Kerala Health Dept <span class="v-check">✓</span></div>
              <div class="wa-sub">Official Outbreak Early Warning</div>
            </div>
          </div>
          <div class="wa-bubble">
            <div class="wa-title" id="waTitle">🚨 OUTBREAK WARNING ADVISORY</div>
            <div class="wa-text" id="waText">Loading WhatsApp message...</div>
            <div class="wa-time">
              <span id="waTimeVal">12:30 PM</span> • Delivered <span class="blue-ticks">✓✓</span>
            </div>
          </div>
          <div class="wa-actions">
            <div class="wa-btn" onclick="triggerSimulatedAck('Guidelines Viewed')">📋 View Guidelines</div>
            <div class="wa-btn primary" onclick="triggerSimulatedAck('Alert Acknowledged')">🚨 Acknowledge Advisory</div>
          </div>
        </div>

        <!-- Audit Trail History -->
        <div style="margin-top:10px;">
          <div class="ml">Session Dispatch Log</div>
          <div id="auditLog" style="font-size:.73rem;color:#64748b;font-style:italic;max-height:80px;overflow-y:auto;background:#f8fafc;padding:6px 8px;border-radius:6px;border:1px solid #e2e8f0;">
            No alerts dispatched in this session yet. Click "Send Alert Broadcast" above to test.
          </div>
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

<!-- Dispatch Alert Modal -->
<div class="modal-overlay" id="alertModal">
  <div class="modal-box">
    <div class="modal-header">
      <h3 id="modalHeaderTitle">⚡ Dispatch Outbreak Warning</h3>
      <button class="close-modal" onclick="closeAlertModal()">&times;</button>
    </div>
    <div class="modal-body">
      <div style="background:#fffbeb;border:1px solid #fde68a;color:#92400e;padding:10px;border-radius:8px;margin-bottom:12px;font-size:.78rem;">
        <strong>Target District:</strong> <span id="modalTargetDist">Palakkad</span> | 
        <strong>Status:</strong> <span id="modalTargetStatus" style="font-weight:700;">Emergency Warning</span>
      </div>

      <div class="form-group">
        <label>Select Target Audience:</label>
        <select id="modalAudience">
          <option value="All Registered Residents">All Registered Residents & ASHAs (District Broadcast)</option>
          <option value="Primary Health Centers">Primary Health Centers (PHCs) & Local Clinics</option>
          <option value="District Medical Officers">District Medical Officers (DMO) Emergency Team</option>
        </select>
      </div>

      <div class="form-group">
        <label>Broadcast Channels:</label>
        <div class="chk-group">
          <label><input type="checkbox" id="chkSms" checked> 💬 SMS Broadcast (Twilio / GSM Gateway)</label>
          <label><input type="checkbox" id="chkWa" checked> 🟢 WhatsApp Business API (Cloud API)</label>
          <label><input type="checkbox" id="chkBeacon"> 📡 DPH Public Health Emergency Beacon</label>
        </div>
      </div>

      <div class="form-group">
        <label>Optional Webhook API Endpoint (Live Backend Integration):</label>
        <input type="text" id="webhookUrl" placeholder="https://your-api.com/webhook/send-alert (Optional)">
      </div>

      <div id="dispatchProgress" style="display:none;text-align:center;padding:10px 0;color:#3b82f6;font-weight:600;font-size:.82rem;">
        <span id="progressText">⏳ Initializing dispatch gateway...</span>
      </div>
    </div>
    <div style="display:flex;justify-content:flex-end;gap:8px;">
      <button onclick="closeAlertModal()" style="background:#f1f5f9;color:#475569;border:none;padding:8px 14px;border-radius:8px;font-weight:600;font-size:.8rem;cursor:pointer;">Cancel</button>
      <button id="sendAlertNowBtn" onclick="executeBroadcast()" style="background:#ef4444;color:#fff;border:none;padding:8px 16px;border-radius:8px;font-weight:700;font-size:.8rem;cursor:pointer;display:flex;align-items:center;gap:6px;">🚀 Broadcast Alert Now</button>
    </div>
  </div>
</div>

<!-- Toast Notifications Container -->
<div class="toast-container" id="toastContainer"></div>

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
let auditHistory = [];

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
  Object.values(markers).forEach(m => map.removeLayer(m));
  labelMarkers.forEach(m => map.removeLayer(m));
  markers = {}; labelMarkers = [];

  const wd = DATA.warnings[currentWeek];

  DATA.districts.forEach(dist => {
    const d      = wd[dist];
    const hex    = COLOR[d.color];
    const isSeas = (dist === 'Palakkad');
    const isSelected = (dist === currentDist);

    const m = L.circleMarker(COORDS[dist], {
      radius:      isSelected ? 24 : 20,
      fillColor:   hex,
      fillOpacity: 0.88,
      color:       isSeas ? '#38bdf8' : '#fff',
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

    const icon = L.divIcon({
      className:'',
      html:`<span style="font:700 10px/1 Inter,sans-serif;color:#0f172a;text-shadow:0 0 4px #fff,0 0 4px #fff,0 0 4px #fff;white-space:nowrap;pointer-events:none">${dist}</span>`,
      iconAnchor:[0,-26]
    });
    const lm = L.marker(COORDS[dist],{icon,interactive:false,zIndexOffset:1000}).addTo(map);
    labelMarkers.push(lm);
  });

  setTimeout(() => { if(markers[currentDist]) markers[currentDist].openPopup(); }, 700);
}

// ── Notification Simulator Logic ──────────────────────────────────────────────
function switchAlertTab(tab) {
  document.getElementById('tabSmsBtn').classList.toggle('active', tab==='sms');
  document.getElementById('tabWaBtn').classList.toggle('active', tab==='wa');
  document.getElementById('smsCard').style.display = tab==='sms' ? 'block' : 'none';
  document.getElementById('waCard').style.display = tab==='wa' ? 'block' : 'none';
}

function updateAlertPreviews(d) {
  const dis = currentDist;
  const status = d.status;
  const disease = d.disease !== '-' ? d.disease : 'Infectious Surveillance';
  const cases = d.cases > 0 ? `${d.cases} case(s)` : '0 cases';
  const rec = d.recommendation;

  const smsText = `[KERALA DPH ALERT] ${status.toUpperCase()} for ${dis}. ${disease} (${cases} in ${currentWeek}). Action: ${rec} Helpline: 1056. Ref:#OBD-${currentWeek.replace(/\\s+/g,'')}`;
  document.getElementById('smsBodyText').textContent = smsText;
  document.getElementById('smsCharStats').textContent = `${smsText.length} chars | ${Math.ceil(smsText.length/160)} SMS segment(s)`;

  const waText = `*District:* ${dis}\n*Alert Level:* ${status}\n*Primary Trigger:* ${disease} (${cases})\n*Period:* ${currentWeek}\n\n*Recommended Action:*\n${rec}\n\n_Generated by AI Outbreak Detection System (Malabar Network)_`;
  document.getElementById('waTitle').textContent = `🚨 ${status.toUpperCase()} ADVISORY`;
  document.getElementById('waText').textContent = waText;
  
  const now = new Date();
  document.getElementById('waTimeVal').textContent = now.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
}

function openAlertModal() {
  const d = DATA.warnings[currentWeek][currentDist];
  document.getElementById('modalTargetDist').textContent = currentDist;
  document.getElementById('modalTargetStatus').textContent = d.status;
  document.getElementById('alertModal').classList.add('open');
  document.getElementById('dispatchProgress').style.display = 'none';
  document.getElementById('sendAlertNowBtn').disabled = false;
}

function closeAlertModal() {
  document.getElementById('alertModal').classList.remove('open');
}

function executeBroadcast() {
  const d = DATA.warnings[currentWeek][currentDist];
  const audience = document.getElementById('modalAudience').value;
  const useSms = document.getElementById('chkSms').checked;
  const useWa = document.getElementById('chkWa').checked;
  const webhook = document.getElementById('webhookUrl').value.trim();

  const btn = document.getElementById('sendAlertNowBtn');
  const prog = document.getElementById('dispatchProgress');
  const progTxt = document.getElementById('progressText');

  btn.disabled = true;
  prog.style.display = 'block';

  progTxt.textContent = '⚡ [1/3] Encrypting alert payload...';

  setTimeout(() => {
    progTxt.textContent = '📡 [2/3] Dispatching to SMS & WhatsApp Gateways...';
  }, 500);

  setTimeout(() => {
    progTxt.textContent = '✅ [3/3] Broadcast complete!';

    if (webhook) {
      fetch(webhook, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ district: currentDist, week: currentWeek, status: d.status, disease: d.disease, audience, timestamp: new Date().toISOString() })
      }).catch(err => console.log('Webhook call status:', err));
    }

    const count = Math.floor(Math.random() * 800) + 1200;
    const channels = [];
    if(useSms) channels.push('SMS');
    if(useWa) channels.push('WhatsApp');

    showToast(`✅ Alert Broadcast Dispatched! ${count} messages delivered to ${audience} in ${currentDist} via ${channels.join(' & ') || 'System Beacon'}.`);

    const timeStr = new Date().toLocaleTimeString();
    auditHistory.unshift(`[${timeStr}] Dispatched ${d.status} for ${currentDist} (${d.disease}) → ${audience} (${channels.join('/')})`);
    renderAuditLog();

    closeAlertModal();
  }, 1100);
}

function triggerSimulatedAck(action) {
  showToast(`📱 Citizen Interaction Recorded: "${action}" for ${currentDist} alert.`);
}

function showToast(msg) {
  const container = document.getElementById('toastContainer');
  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.innerHTML = `<span>${msg}</span>`;
  container.appendChild(toast);
  setTimeout(() => toast.classList.add('show'), 50);
  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

function renderAuditLog() {
  const container = document.getElementById('auditLog');
  if (auditHistory.length === 0) return;
  container.innerHTML = auditHistory.map(item => `<div style="padding:2px 0;border-bottom:1px solid #f1f5f9;">• ${item}</div>`).join('');
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

  // Update notification cards
  updateAlertPreviews(d);

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
    const byActivity = Object.entries(sd).sort((a,b)=>b[1].reduce((s,x)=>s+x,0)-a[1].reduce((s,x)=>s+x,0));
    byActivity.forEach(([dis,avgs]) => {
      if(Math.max(...avgs) < 0.05) return;
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
