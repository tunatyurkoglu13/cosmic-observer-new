// COSMIC OBSERVER — shared alert badge widget, included on every page
// (dashboard, sensor feeds, trends) so an active alert is visible no
// matter where the user currently is, not just on /alerts itself.
// Injects a small fixed badge (unacknowledged count) linking to /alerts,
// connects to /ws/alerts for live updates, and shows a real browser
// notification for newly-arrived alerts (only if the user already
// granted permission on the /alerts page — this widget never itself
// prompts, to avoid an unsolicited permission dialog on every page load).

(function () {
  const badge = document.createElement('a');
  badge.href = '/alerts';
  badge.id = 'co-alert-badge';
  badge.style.cssText = `
    position: fixed; bottom: 16px; right: 16px; z-index: 9998;
    display: flex; align-items: center; gap: 6px;
    background: rgba(0,10,12,0.9); border: 1px solid var(--co-cyan, #00ffff);
    color: var(--co-cyan, #00ffff); font-family: 'JetBrains Mono', monospace;
    font-size: 11px; letter-spacing: 0.05em; padding: 7px 12px;
    text-decoration: none; box-shadow: 0 0 12px rgba(0,255,255,0.15);
  `;
  badge.innerHTML = `<span>&#128276; ALERTS</span><span id="co-alert-badge-count" style="display:none; background:#ff3355; color:#fff; border-radius:10px; padding:1px 7px; font-weight:700;"></span>`;
  document.body.appendChild(badge);

  const countEl = badge.querySelector('#co-alert-badge-count');

  function setCount(n) {
    if (n > 0) {
      countEl.textContent = n;
      countEl.style.display = 'inline-block';
      badge.style.borderColor = '#ff3355';
    } else {
      countEl.style.display = 'none';
      badge.style.borderColor = 'var(--co-cyan, #00ffff)';
    }
  }

  async function pollCount() {
    try {
      const resp = await fetch('/api/alerts/unacknowledged-count');
      if (!resp.ok) return;
      const data = await resp.json();
      setCount(data.count);
    } catch (err) {
      // silent — the badge just won't update this cycle, not worth surfacing an error UI for
    }
  }

  function connectSocket() {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${proto}//${window.location.host}/ws/alerts`);
    ws.onmessage = (event) => {
      let alert;
      try { alert = JSON.parse(event.data); } catch (err) { return; }
      pollCount();
      if ('Notification' in window && Notification.permission === 'granted') {
        try {
          new Notification(`[${alert.severity.toUpperCase()}] ${alert.title}`, {
            body: alert.description, tag: `cosmic-observer-alert-${alert.id}`,
          });
        } catch (err) { /* non-fatal */ }
      }
    };
    ws.onclose = () => setTimeout(connectSocket, 5000);
  }

  pollCount();
  connectSocket();
  setInterval(pollCount, 60000);
})();
