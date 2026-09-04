// COSMIC OBSERVER — ACTIVE ALERT LAYER frontend.
// Real alert history (GET /api/alerts) + live push (WebSocket /ws/alerts,
// fed by app.py's background conjunction/NEO scanners and the live CV
// anomaly detector) + real browser notifications (Web Notifications API)
// + acknowledge workflow.

const CATEGORY_LABELS = {
  conjunction: 'CONJUNCTION', anomaly: 'CV ANOMALY',
  neo_risk: 'NEO RISK', neo_close_approach: 'NEO CLOSE APPROACH',
};

let currentAlerts = [];

async function fetchJson(url, options) {
  const resp = await fetch(url, options);
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${resp.status}`);
  }
  return resp.json();
}

function renderAlert(alert, { flash = false } = {}) {
  const card = document.createElement('div');
  card.className = `alert-card severity-${alert.severity}${alert.acknowledged ? ' acknowledged' : ''}${flash ? ' flash' : ''}`;
  card.dataset.id = alert.id;

  const ts = alert.timestamp.replace('T', ' ').slice(0, 19) + ' UTC';
  card.innerHTML = `
    <div class="alert-header">
      <span class="alert-title">${alert.title}</span>
      <span class="alert-badge">${CATEGORY_LABELS[alert.category] || alert.category} &middot; ${alert.severity.toUpperCase()}</span>
    </div>
    <div class="alert-meta">${ts}${alert.acknowledged ? ' &middot; ACKNOWLEDGED' : ''}</div>
    <div class="alert-desc">${alert.description}</div>
    ${alert.acknowledged ? '' : `<button class="co-btn alert-ack-btn" data-id="${alert.id}">ACKNOWLEDGE</button>`}
  `;

  const ackBtn = card.querySelector('.alert-ack-btn');
  if (ackBtn) {
    ackBtn.addEventListener('click', async () => {
      try {
        await fetchJson(`/api/alerts/${alert.id}/acknowledge`, { method: 'POST' });
        alert.acknowledged = true;
        await refreshAlerts();
      } catch (err) {
        ackBtn.textContent = `FAILED: ${err.message || err}`;
      }
    });
  }

  return card;
}

function renderStats() {
  const total = currentAlerts.length;
  const unacked = currentAlerts.filter((a) => !a.acknowledged).length;
  const critical = currentAlerts.filter((a) => a.severity === 'critical' && !a.acknowledged).length;
  const warning = currentAlerts.filter((a) => a.severity === 'warning' && !a.acknowledged).length;

  document.getElementById('stat-total').textContent = total;
  document.getElementById('stat-unacked').textContent = unacked;
  document.getElementById('stat-critical').textContent = critical;
  document.getElementById('stat-warning').textContent = warning;
}

async function refreshAlerts() {
  const category = document.getElementById('filter-category').value;
  const ackFilter = document.getElementById('filter-ack').value;

  const params = new URLSearchParams();
  if (category) params.set('category', category);
  if (ackFilter === 'unacked') params.set('unacknowledged_only', 'true');
  params.set('limit', '200');

  const data = await fetchJson(`/api/alerts?${params.toString()}`);
  currentAlerts = data.alerts;

  const listEl = document.getElementById('alert-list');
  const emptyEl = document.getElementById('alert-list-empty');
  listEl.innerHTML = '';
  if (currentAlerts.length === 0) {
    emptyEl.style.display = 'block';
  } else {
    emptyEl.style.display = 'none';
    for (const alert of currentAlerts) listEl.appendChild(renderAlert(alert));
  }
  renderStats();
}

document.getElementById('filter-category').addEventListener('change', refreshAlerts);
document.getElementById('filter-ack').addEventListener('change', refreshAlerts);
document.getElementById('refresh-btn').addEventListener('click', refreshAlerts);

// ---------------------------------------------------------------------
// Browser notifications
// ---------------------------------------------------------------------

function updateNotifButton() {
  const btn = document.getElementById('notif-btn');
  if (!('Notification' in window)) {
    btn.textContent = 'NOTIFICATIONS UNSUPPORTED';
    btn.disabled = true;
    return;
  }
  if (Notification.permission === 'granted') {
    btn.textContent = '\u{1F514} NOTIFICATIONS ENABLED';
    btn.classList.add('active');
  } else if (Notification.permission === 'denied') {
    btn.textContent = 'NOTIFICATIONS BLOCKED (check browser settings)';
    btn.disabled = true;
  } else {
    btn.textContent = '\u{1F514} ENABLE BROWSER NOTIFICATIONS';
  }
}

document.getElementById('notif-btn').addEventListener('click', async () => {
  if (!('Notification' in window)) return;
  await Notification.requestPermission();
  updateNotifButton();
});
updateNotifButton();

function showBrowserNotification(alert) {
  if (!('Notification' in window) || Notification.permission !== 'granted') return;
  try {
    new Notification(`[${alert.severity.toUpperCase()}] ${alert.title}`, {
      body: alert.description,
      tag: `cosmic-observer-alert-${alert.id}`,
    });
  } catch (err) {
    // Notification construction can throw in some contexts (e.g. service-worker-only browsers) — non-fatal.
  }
}

function flashKlaxon(alert) {
  if (alert.severity !== 'critical') return;
  const banner = document.getElementById('klaxon-banner');
  banner.textContent = `⚠ CRITICAL ALERT — ${alert.title}`;
  banner.style.display = 'block';
  setTimeout(() => { banner.style.display = 'none'; }, 6000);
}

// ---------------------------------------------------------------------
// Live push via WebSocket
// ---------------------------------------------------------------------

function connectAlertSocket() {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${proto}//${window.location.host}/ws/alerts`);
  const linkEl = document.getElementById('stat-link');

  ws.onopen = () => { linkEl.textContent = 'LIVE'; linkEl.style.color = 'var(--co-phosphor-green)'; };
  ws.onclose = () => {
    linkEl.textContent = 'RECONNECTING…'; linkEl.style.color = 'var(--co-amber)';
    setTimeout(connectAlertSocket, 3000);
  };
  ws.onerror = () => { linkEl.textContent = 'ERROR'; linkEl.style.color = '#ff3355'; };

  ws.onmessage = async (event) => {
    let alert;
    try { alert = JSON.parse(event.data); } catch (err) { return; }

    showBrowserNotification(alert);
    flashKlaxon(alert);
    await refreshAlerts();

    // Highlight the just-arrived card, if it's currently in view (filters may exclude it).
    const card = document.querySelector(`.alert-card[data-id="${alert.id}"]`);
    if (card) {
      card.classList.add('flash');
      setTimeout(() => card.classList.remove('flash'), 3000);
      card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  };
}

refreshAlerts();
connectAlertSocket();
setInterval(refreshAlerts, 30000);
