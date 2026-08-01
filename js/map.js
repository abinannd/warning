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
  document.getElementById('statRed').textContent    = `?? ${r} Emergency`;
  document.getElementById('statYellow').textContent = `?? ${y} Watch`;
  document.getElementById('statGreen').textContent  = `?? ${g} Normal`;
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
