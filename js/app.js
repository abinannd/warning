// ── Boot ──────────────────────────────────────────────────────────────────────

async function loadDashboardData() {
    try {
        const response = await fetch('data/json/dashboard_data.json');
        DATA = await response.json();
        
        document.getElementById('dispatchLog').value = 'No alerts dispatched in this session yet. Click "Send Alert Broadcast" above to test.';
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => setActiveAlertTab(btn.getAttribute('data-tab')));
        });
        const sendBtn = document.getElementById('sendAlertBtn');
        if (sendBtn) sendBtn.addEventListener('click', appendDispatchLog);
        
        setTimeout(() => { map.invalidateSize(); updateMap(); updateDetail(); setActiveAlertTab('sms'); }, 300);
    } catch (e) {
        console.error("Failed to load dashboard data:", e);
    }
}
window.addEventListener('load', loadDashboardData);
