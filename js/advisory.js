const DISEASE_ACTIONS = {
  'Dengue': [
    'Remove all stagnant water from containers, flower pots, and coolers.',
    'Sleep under mosquito nets, especially during daytime.',
    'Wear long-sleeved clothing and use mosquito repellent.',
    'Stay hydrated and eat nutritious food.',
    'Visit a hospital immediately if fever persists beyond 2 days.',
    'Do not self-medicate with aspirin or ibuprofen — consult a doctor.'
  ],
  'Chikungunya': [
    'Prevent mosquito bites — use nets, repellents and protective clothing.',
    'Take adequate rest and avoid over-exertion.',
    'Drink plenty of fluids to stay hydrated.',
    'Apply cold compresses to reduce joint pain and swelling.',
    'Consult a doctor if symptoms worsen or fever exceeds 102°F.',
    'Eliminate mosquito breeding sites around your home.'
  ],
  'Influenza': [
    'Wear a well-fitted mask in crowded or enclosed spaces.',
    'Wash hands frequently with soap for at least 20 seconds.',
    'Avoid crowded places and gatherings if symptomatic.',
    'Stay home and rest if you develop fever, cough, or body ache.',
    'Ensure adequate ventilation in indoor spaces.',
    'Consult a doctor if symptoms are severe or persist beyond 5 days.'
  ],
  'COVID-19': [
    'Wear a mask in public spaces, especially indoors.',
    'Maintain at least 1-metre physical distance from others.',
    'Get tested if you develop fever, cough, or loss of smell/taste.',
    'Follow local health authority guidelines and quarantine if positive.',
    'Ensure good ventilation in your home and workspace.',
    'Keep up to date with vaccination as recommended.'
  ],
  'Malaria': [
    'Sleep under insecticide-treated mosquito nets every night.',
    'Apply insect repellent on exposed skin at dusk and dawn.',
    'Eliminate stagnant water sources near your home.',
    'Wear long-sleeved clothing in the evenings.',
    'Seek immediate medical attention if you have fever with chills.',
    'Take anti-malarial medication only as prescribed by a doctor.'
  ],
  'Typhoid': [
    'Drink only boiled or purified/bottled water.',
    'Maintain strict food hygiene — eat freshly cooked food.',
    'Wash hands thoroughly before eating and after using the toilet.',
    'Avoid street food and raw vegetables washed in tap water.',
    'Get vaccinated if travelling to high-risk areas.',
    'Consult a doctor if you have sustained high fever for more than 3 days.'
  ],
  'Leptospirosis': [
    'Avoid wading through floodwater or stagnant pools.',
    'Wear rubber boots and gloves when handling soil or water.',
    'Cover cuts and skin abrasions with waterproof dressings.',
    'Wash hands with soap after contact with animals or soil.',
    'Seek medical attention if fever develops after water exposure.',
    'Report any dead rodents near your home to local authorities.'
  ],
  'Hepatitis A': [
    'Drink only safe, boiled or purified water.',
    'Avoid raw or undercooked shellfish.',
    'Maintain good hand hygiene before eating and after using the toilet.',
    'Get vaccinated against Hepatitis A if not already immunised.',
    'Consult a doctor if you develop jaundice (yellowing of eyes/skin).'
  ],
  'Viral Fever': [
    'Rest adequately and stay well hydrated.',
    'Take paracetamol as directed for fever relief.',
    'Avoid self-medication with antibiotics.',
    'Use mosquito repellents to prevent vector-borne infections.',
    'Seek medical attention if fever is very high or persists beyond 3 days.',
    'Isolate at home if you suspect a contagious illness.'
  ],
  'Rabies': [
    'Avoid contact with stray or wild animals.',
    'If bitten or scratched by an animal, wash the wound immediately with soap and water for 15 minutes.',
    'Seek medical attention immediately after any animal bite.',
    'Ensure your pets are vaccinated against rabies.',
    'Do not attempt to handle or rescue stray animals without protection.'
  ],
  'Common Cold': [
    'Wash hands frequently to prevent spread.',
    'Cover your mouth when sneezing or coughing.',
    'Avoid close contact with infected individuals.',
    'Stay hydrated and rest adequately.',
    'Consult a doctor if symptoms are prolonged or severe.'
  ],
  'Chickenpox': [
    'Keep the infected person isolated until all blisters have crusted over.',
    'Avoid scratching blisters to prevent secondary infection.',
    'Trim fingernails short to reduce the risk of skin damage.',
    'Apply calamine lotion to relieve itching.',
    'Ensure good ventilation in the patient\'s room.',
    'Consult a doctor if blisters become infected or fever is very high.'
  ],
  'Scrub Typhus': [
    'Wear protective clothing (long sleeves, trousers, boots) in scrub areas.',
    'Apply insect repellent containing DEET on exposed skin.',
    'Check your body for mites after being outdoors.',
    'Avoid sitting or lying directly on the ground in vegetation areas.',
    'Seek immediate medical attention if fever develops after outdoor exposure.'
  ]
};

// ── Emergency-specific action items ──────────────────────────────────────────
const EMERGENCY_ACTIONS = [
  'Report all suspected cases immediately to the nearest PHC or District Health Office.',
  'Activate local rapid response teams and community health workers.',
  'Suspend or postpone non-essential public gatherings in the affected area.',
  'Ensure adequate supply of medicines, ORS, and diagnostic kits at health centres.',
  'Intensify vector control operations (fumigation, fogging, larval source reduction).',
  'Issue public announcements through local media and community leaders.',
  'Establish 24-hour surveillance hotline for case reporting.'
];

// ── AI Reason lookup (status → explanation) ──────────────────────────────────
const AI_REASON = {
  'Emergency Warning':    'AI surveillance detected a statistically significant spike (Z-score ≥ 3.0) in reported cases, far exceeding the historical baseline for this disease and season.',
  'High Risk':            'AI surveillance detected a statistically significant spike (Z-score ≥ 3.0) in reported cases, far exceeding the historical baseline for this disease and season.',
  'Watch-Status Warning': 'AI model detected a high-sensitivity signal (Z-score ≥ 2.0) suggesting elevated disease activity above the seasonal norm. Active monitoring escalated.',
  'Advisory':             'AI model identified elevated statistical activity (Z-score ≥ 1.5) above baseline. Precautionary advisory issued for increased vigilance.',
  'Normal':               'No significant deviation from baseline activity detected. Routine surveillance in progress.'
};

// ── Advisory generation logic ─────────────────────────────────────────────────
function generateAdvisory(district, districtData) {
  const status   = districtData.status;
  const disease  = districtData.disease !== '-' ? districtData.disease : null;
  const cases    = districtData.cases;
  const week     = DATA.warnings[currentWeek];
  const weekLabel = week.label || currentWeek;

  const isHighAlert = (status === 'Emergency Warning' || status === 'High Risk');
  const isWatch     = (status === 'Watch-Status Warning' || status === 'Advisory');
  const isNormal    = (status === 'Normal');

  // Card colour class
  const card = document.getElementById('advisoryCard');
  card.className = 'advisory-card ' + (
    isHighAlert ? 'advisory-red' :
    isWatch     ? 'advisory-yellow' :
    isNormal    ? 'advisory-green' : 'advisory-gray'
  );

  // Badge
  const badge = document.getElementById('advBadge');
  badge.textContent = status;
  badge.className = 'adv-header-badge badge ' + (
    isHighAlert ? 'badge-red' :
    isWatch     ? 'badge-yellow' :
    isNormal    ? 'badge-green' : 'badge-gray'
  );

  // Banner
  const banner = document.getElementById('advBanner');
  if (isHighAlert && disease) {
    banner.className = 'adv-banner adv-banner-red';
    banner.innerHTML = `⚠ PUBLIC HEALTH ALERT — <strong>${district}</strong> — <strong>${disease}</strong> — <strong>${status}</strong>`;
  } else if (isWatch && disease) {
    banner.className = 'adv-banner adv-banner-yellow';
    banner.innerHTML = `&#x1F7E1; HEALTH ADVISORY — <strong>${district}</strong> — <strong>${disease}</strong> — <strong>${status}</strong>`;
  } else if (isNormal) {
    banner.className = 'adv-banner adv-banner-green';
    banner.innerHTML = `&#x2705; No active public health advisory for <strong>${district}</strong> this period.`;
  } else {
    banner.className = 'adv-banner adv-banner-gray';
    banner.textContent = 'No active public health advisory for this district.';
  }

  // Public safety message
  const msgWrap = document.getElementById('advMessageWrap');
  const msgEl   = document.getElementById('advMessage');
  if ((isHighAlert || isWatch) && disease) {
    msgWrap.style.display = '';
    const dis = disease;
    const safetyMsg = isHighAlert
      ? `⚠ PUBLIC HEALTH ALERT\n\nDistrict: ${district}\nDisease: ${dis}\nRisk Level: ${status}\n\nThe AI surveillance system has detected an unusual increase in reported ${dis} cases. This alert requires immediate attention from residents and health authorities.\n\nResidents are strongly advised to follow the safety instructions below. Seek medical attention immediately if symptoms develop.\n\nThis alert is generated using AI-based outbreak detection and is precautionary in nature. Always follow guidance from official health authorities.`
      : `ℹ HEALTH ADVISORY\n\nDistrict: ${district}\nDisease: ${dis}\nRisk Level: ${status}\n\nThe AI surveillance system has identified elevated ${dis} activity in ${district}. This is a precautionary advisory to encourage vigilance.\n\nResidents are advised to take preventive measures and monitor for symptoms.\n\nThis advisory is generated using AI-based outbreak detection.`;
    msgEl.textContent = safetyMsg;
  } else {
    msgWrap.style.display = 'none';
  }

  // Recommended actions
  const actWrap = document.getElementById('advActionsWrap');
  const actList  = document.getElementById('advActions');
  if ((isHighAlert || isWatch) && disease) {
    actWrap.style.display = '';
    actList.innerHTML = '';
    const actions = DISEASE_ACTIONS[disease] || DISEASE_ACTIONS['Viral Fever'];
    actions.forEach(a => {
      const li = document.createElement('li');
      li.textContent = a;
      actList.appendChild(li);
    });
  } else {
    actWrap.style.display = 'none';
  }

  // Emergency recommendations
  const emgWrap = document.getElementById('advEmergencyWrap');
  const emgList  = document.getElementById('advEmergency');
  if (isHighAlert) {
    emgWrap.style.display = '';
    emgList.innerHTML = '';
    EMERGENCY_ACTIONS.forEach(a => {
      const li = document.createElement('li');
      li.textContent = a;
      emgList.appendChild(li);
    });
  } else {
    emgWrap.style.display = 'none';
  }

  // Metadata
  const metaWrap = document.getElementById('advMeta');
  if ((isHighAlert || isWatch) && disease) {
    metaWrap.style.display = '';
    document.getElementById('advMetaDistrict').textContent = '📍 ' + district;
    document.getElementById('advMetaDisease').textContent  = '🦠 ' + disease;
    document.getElementById('advMetaCases').textContent    = '📊 Cases: ' + cases;
    document.getElementById('advMetaDate').textContent     = '🕒 ' + weekLabel + ' — ' + new Date().toLocaleString('en-IN', {day:'2-digit',month:'short',year:'numeric',hour:'2-digit',minute:'2-digit'});
    document.getElementById('advMetaReason').textContent   = '🤖 AI: ' + (AI_REASON[status] || '');
  } else {
    metaWrap.style.display = 'none';
  }

  // Emergency contact
  document.getElementById('advContact').style.display = (isHighAlert || isWatch) ? '' : 'none';

  // Future-ready data hooks (for SMS / Email / WhatsApp / Push integration)
  const payload = JSON.stringify({ district, disease, status, cases, week: currentWeek, weekLabel, timestamp: new Date().toISOString() });
  ['sms','email','whatsapp','push'].forEach(type => {
    const el = document.getElementById('futureHook' + type.charAt(0).toUpperCase() + type.slice(1));
    if (el) el.setAttribute('data-payload', payload);
  });
}

// ── Emergency notification banner (dismissible) ───────────────────────────────
