let bannerDismissedFor = null;

function showEmergencyBanner(district, disease, status, cases) {
  const key = district + '|' + currentWeek;
  if (bannerDismissedFor === key) return;

  let banner = document.getElementById('emergencyBanner');
  if (!banner) {
    banner = document.createElement('div');
    banner.id = 'emergencyBanner';
    document.body.appendChild(banner);
  }
  banner.className = '';
  banner.style.display = 'flex';
  banner.innerHTML = `
    <div class="eb-icon">&#x1F534;</div>
    <div class="eb-body">
      <div class="eb-title">Emergency Warning — ${district}</div>
      <div class="eb-sub">
        Disease: <strong>${disease}</strong> &bull;
        Cases: <strong>${cases}</strong> &bull;
        ${status}<br>
        <strong>Immediate Action Required.</strong> Deploy rapid response team and notify health authorities.
      </div>
    </div>
    <button class="eb-dismiss" onclick="dismissBanner('${district}|${currentWeek}')">✕ Dismiss</button>
  `;
}

function dismissBanner(key) {
  bannerDismissedFor = key;
  const banner = document.getElementById('emergencyBanner');
  if (banner) {
    banner.style.display = 'none';
    banner.classList.add('hidden');
  }
}

function hideEmergencyBanner() {
  const banner = document.getElementById('emergencyBanner');
  if (banner) banner.style.display = 'none';
}
