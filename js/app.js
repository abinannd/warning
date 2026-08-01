// ── Boot ──────────────────────────────────────────────────────────────────────

async function loadDashboardData() {
    try {
        const response = await fetch('data/json/dashboard_data.json');
        DATA = await response.json();

        // Set the default week now that DATA is available
        currentWeek = DATA.default_week;

        // Build the week dropdown
        initWeekSelector();

        // Set up event listeners
        document.getElementById('dispatchLog').value = 'No alerts dispatched in this session yet. Click "Send Alert Broadcast" above to test.';
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => setActiveAlertTab(btn.getAttribute('data-tab')));
        });
        const sendBtn = document.getElementById('sendAlertBtn');
        if (sendBtn) sendBtn.addEventListener('click', appendDispatchLog);

        // Initialize map and UI
        setTimeout(() => { map.invalidateSize(); updateMap(); updateDetail(); setActiveAlertTab('sms'); }, 300);
    } catch (e) {
        console.error("Failed to load dashboard data:", e);
    }
}
window.addEventListener('load', loadDashboardData);
