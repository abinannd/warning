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

  updateCitizenAlert();

  // ── Generate public advisory for selected district ──────────────────────────
  generateAdvisory(currentDist, d);

  // ── Show/hide emergency banner ──────────────────────────────────────────────
  if ((d.status === 'Emergency Warning' || d.status === 'High Risk') && d.disease && d.disease !== '-') {
    showEmergencyBanner(currentDist, d.disease, d.status, d.cases);
  } else {
    hideEmergencyBanner();
  }
  updateChart();
}

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
        ? `Historical monthly avg. (2020–2024, South Kerala). Peaks show seasonal disease risk windows for ${currentDist}.`
        : `Historical monthly avg. (2018–2024, Malabar). Peaks show seasonal disease risk windows for ${currentDist}.`;

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
}

// ══════════════════════════════════════════════════════════════════════════════
// PUBLIC OUTBREAK ADVISORY & ALERT SYSTEM
// ══════════════════════════════════════════════════════════════════════════════

// ── Disease-specific advisory recommendations ─────────────────────────────────
