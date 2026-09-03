// COSMIC OBSERVER — SENSOR & TELEMETRY NETWORK frontend.
// Tab 1 (ISS CV) connects to /ws/cv exactly as before. Tabs 2-6 are real
// NASA/JPL/STScI/NOAA data feeds, each with its own small loader module
// below — kept intentionally modular (one function per feed, one small
// object describing its refresh behavior) so adding a new sensor later
// is "write a loader + register it," not a rewrite.

// ===========================================================================
// Shared tactical-HUD chrome: CRT noise overlay + SIGNAL status helpers.
// Purely decorative/UI-convention (like a camera app's REC dot) — SIGNAL
// itself reflects a real fetch outcome, not a fabricated claim.
// ===========================================================================

function initCRTNoise(container) {
  const canvas = document.createElement('canvas');
  canvas.className = 'crt-noise-canvas';
  // Low internal resolution, scaled up via CSS — a chunky, authentic
  // static texture that's cheap to redraw every frame.
  canvas.width = 64;
  canvas.height = 48;
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');
  const imageData = ctx.createImageData(canvas.width, canvas.height);

  function draw() {
    const buf = imageData.data;
    for (let i = 0; i < buf.length; i += 4) {
      const v = Math.random() * 255;
      buf[i] = v; buf[i + 1] = v; buf[i + 2] = v; buf[i + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
  }
  draw();
  setInterval(draw, 140);
  return canvas;
}

function setSignal(elId, status) {
  // status: 'optimal' | 'degraded' | 'none'
  const el = document.getElementById(elId);
  if (!el) return;
  el.classList.remove('hud-signal-optimal', 'hud-signal-degraded', 'hud-signal-none');
  if (status === 'optimal') {
    el.textContent = 'SIGNAL: OPTIMAL';
    el.classList.add('hud-signal-optimal');
  } else if (status === 'degraded') {
    el.textContent = 'SIGNAL: DEGRADED';
    el.classList.add('hud-signal-degraded');
  } else {
    el.textContent = 'NO SIGNAL';
    el.classList.add('hud-signal-none');
  }
}

function formatDataRate(bps) {
  if (!bps || bps <= 0) return 'IDLE';
  if (bps >= 1e6) return `${(bps / 1e6).toFixed(2)} Mb/s`;
  if (bps >= 1e3) return `${(bps / 1e3).toFixed(1)} Kb/s`;
  return `${bps.toFixed(0)} b/s`;
}

function formatRangeKm(km) {
  if (km == null) return '--';
  const au = km / 1.496e8;
  if (au >= 0.05) return `${km.toLocaleString()} km (${au.toFixed(3)} AU)`;
  return `${km.toLocaleString()} km`;
}

async function fetchJson(url) {
  const resp = await fetch(url);
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${resp.status}`);
  }
  return resp.json();
}

// ===========================================================================
// Tab switching
// ===========================================================================

const TAB_REFRESH_MS = {
  dsn: 15000,
  telescopes: 5 * 60 * 1000,
  solar: 5 * 60 * 1000,
  earth: 10 * 60 * 1000,
  mars: 10 * 60 * 1000,
};

const tabLoaders = {
  dsn: loadDSN,
  telescopes: loadTelescopes,
  solar: loadSolar,
  earth: loadEarth,
  mars: loadMarsRovers,
};

let activeRefreshTimer = null;

function activateTab(tabKey) {
  document.querySelectorAll('.tab-btn').forEach((btn) => btn.classList.toggle('active', btn.dataset.tab === tabKey));
  document.querySelectorAll('.tab-panel').forEach((panel) => panel.classList.toggle('active', panel.id === `panel-${tabKey}`));

  if (activeRefreshTimer) {
    clearInterval(activeRefreshTimer);
    activeRefreshTimer = null;
  }

  const loader = tabLoaders[tabKey];
  if (loader) {
    loader();
    const interval = TAB_REFRESH_MS[tabKey];
    if (interval) activeRefreshTimer = setInterval(loader, interval);
  }
}

document.querySelectorAll('.tab-btn').forEach((btn) => {
  btn.addEventListener('click', () => activateTab(btn.dataset.tab));
});

// ===========================================================================
// Tab: ISS CV — unchanged live-video WebSocket logic.
// Connects to /ws/cv, which alternates two message types per frame:
//   1. a binary WebSocket message: the JPEG-encoded, HUD-annotated frame
//   2. a text WebSocket message: JSON with that frame's detections + metrics
// Displayed via an <img> + object URL rather than <canvas>, since we
// receive an already-fully-rendered image (HUD baked in server-side by
// cv.hud) — no client-side drawing needed.
// ===========================================================================

let ws = null;
let currentObjectUrl = null;

const videoImg = document.getElementById('video-frame');
const placeholder = document.getElementById('video-placeholder');
const cvStatusEl = document.getElementById('cv-status');
const startBtn = document.getElementById('cv-start-btn');

function setStatus(text) {
  cvStatusEl.textContent = text;
}

function stopStream() {
  if (ws) {
    ws.close();
    ws = null;
  }
  startBtn.textContent = '▶ START STREAM';
  startBtn.classList.remove('active');
}

function startStream() {
  if (ws) {
    stopStream();
    return;
  }

  const source = document.getElementById('cv-source').value;
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = `${proto}//${window.location.host}/ws/cv?source=${encodeURIComponent(source)}`;

  ws = new WebSocket(url);
  ws.binaryType = 'blob';

  ws.onopen = () => {
    setStatus(`CONNECTED — SOURCE: ${source.toUpperCase()}`);
    startBtn.textContent = '■ STOP STREAM';
    startBtn.classList.add('active');
    placeholder.style.display = 'none';
    videoImg.style.display = 'block';
  };

  ws.onmessage = (event) => {
    if (typeof event.data === 'string') {
      handleMetadataMessage(event.data);
    } else {
      handleFrameMessage(event.data);
    }
  };

  ws.onerror = () => {
    setStatus('CONNECTION ERROR');
  };

  ws.onclose = () => {
    setStatus('DISCONNECTED');
    startBtn.textContent = '▶ START STREAM';
    startBtn.classList.remove('active');
    ws = null;
  };
}

function handleFrameMessage(blob) {
  const nextUrl = URL.createObjectURL(blob);
  videoImg.onload = () => {
    if (currentObjectUrl) URL.revokeObjectURL(currentObjectUrl);
    currentObjectUrl = nextUrl;
  };
  videoImg.src = nextUrl;
}

function handleMetadataMessage(jsonText) {
  let data;
  try {
    data = JSON.parse(jsonText);
  } catch (err) {
    return;
  }

  if (data.error) {
    setStatus(`ERROR: ${data.error}`);
    stopStream();
    return;
  }

  if (data.notice) {
    setStatus(`NOTICE: ${data.notice}`);
    return;
  }

  document.getElementById('stat-fps').textContent = data.fps.toFixed(1);
  document.getElementById('stat-frame').textContent = data.frame_index;
  document.getElementById('stat-targets').textContent = data.detections.length;
  document.getElementById('stat-conf').textContent = data.avg_confidence.toFixed(2);

  updateAnomalyDisplay(data.anomaly);

  const listEl = document.getElementById('detections-list');
  if (data.detections.length === 0) {
    listEl.innerHTML = '<div class="dim">NO TARGETS ACQUIRED</div>';
    return;
  }

  listEl.innerHTML = data.detections
    .map((d) => {
      const [x1, y1, x2, y2] = d.box_xyxy;
      return `<div class="det-row"><b>${d.class_name.toUpperCase()}</b> — ${(d.confidence * 100).toFixed(0)}%<br>
        <span class="dim">box: (${x1.toFixed(0)}, ${y1.toFixed(0)}) &rarr; (${x2.toFixed(0)}, ${y2.toFixed(0)})</span></div>`;
    })
    .join('');
}

async function uploadVideo() {
  const fileInput = document.getElementById('cv-file');
  if (!fileInput.files.length) {
    setStatus('NO FILE SELECTED');
    return;
  }

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);

  setStatus('UPLOADING…');
  try {
    const resp = await fetch('/api/cv/upload', { method: 'POST', body: formData });
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      throw new Error(body.detail || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    setStatus(`UPLOADED: ${data.filename} (${(data.size_bytes / 1e6).toFixed(1)} MB)`);
    document.getElementById('cv-source').value = 'upload';
  } catch (err) {
    setStatus(`UPLOAD FAILED: ${err.message || err}`);
  }
}

async function identifyImage() {
  const fileInput = document.getElementById('identify-file');
  const statusEl = document.getElementById('identify-status');
  if (!fileInput.files.length) {
    statusEl.textContent = 'NO FILE SELECTED';
    return;
  }

  stopStream();

  const queries = document.getElementById('identify-queries').value;
  const formData = new FormData();
  formData.append('file', fileInput.files[0]);
  formData.append('text_queries', queries);

  statusEl.textContent = 'ANALYZING… (zero-shot detection is slower than the live stream)';
  try {
    const resp = await fetch('/api/cv/identify', { method: 'POST', body: formData });
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      throw new Error(body.detail || `HTTP ${resp.status}`);
    }
    const data = await resp.json();

    placeholder.style.display = 'none';
    videoImg.style.display = 'block';
    videoImg.src = `data:image/jpeg;base64,${data.annotated_image_base64}`;

    document.getElementById('stat-targets').textContent = data.detections.length;
    document.getElementById('stat-streaks').textContent = data.streaks.length;
    document.getElementById('stat-fps').textContent = '--';
    document.getElementById('stat-frame').textContent = '--';
    document.getElementById('stat-conf').textContent = data.detections.length
      ? (data.detections.reduce((s, d) => s + d.confidence, 0) / data.detections.length).toFixed(2)
      : '--';

    const listEl = document.getElementById('detections-list');
    const rows = [];
    for (const d of data.detections) {
      rows.push(`<div class="det-row"><b>${d.class_name.toUpperCase()}</b> — ${(d.confidence * 100).toFixed(0)}% (zero-shot)</div>`);
    }
    for (const s of data.streaks) {
      const skyInfo = s.start_sky ? ` &middot; RA/Dec ${s.start_sky[0].toFixed(2)}, ${s.start_sky[1].toFixed(2)}` : '';
      rows.push(`<div class="det-row" style="border-left-color:var(--co-amber);"><b>STREAK</b> — ${s.length_px.toFixed(0)}px @ ${s.angle_deg.toFixed(0)}&deg;${skyInfo}</div>`);
    }
    listEl.innerHTML = rows.length ? rows.join('') : '<div class="dim">NO TARGETS ACQUIRED</div>';

    let statusMsg = `DONE — ${data.detections.length} ZERO-SHOT MATCH(ES), ${data.streaks.length} STREAK(S)`;
    if (data.has_wcs) statusMsg += ' — WCS PRESENT (sky coords available)';
    if (data.zero_shot_error) statusMsg += ` — ZERO-SHOT ERROR: ${data.zero_shot_error}`;
    statusEl.textContent = statusMsg;
  } catch (err) {
    statusEl.textContent = `FAILED: ${err.message || err}`;
  }
}

startBtn.addEventListener('click', startStream);
document.getElementById('cv-upload-btn').addEventListener('click', uploadVideo);
document.getElementById('identify-btn').addEventListener('click', identifyImage);
initCRTNoise(document.getElementById('cv-visual'));

// ---------------------------------------------------------------------
// Anomaly detection (ConvAutoencoder, cv/anomaly.py + cv/anomaly_train.py)
// ---------------------------------------------------------------------

function updateAnomalyDisplay(anomaly) {
  const errorEl = document.getElementById('anomaly-error');
  const thresholdEl = document.getElementById('anomaly-threshold');
  const statusEl = document.getElementById('anomaly-status');

  if (!anomaly) {
    errorEl.textContent = '--';
    thresholdEl.textContent = '--';
    statusEl.textContent = 'MODEL NOT LOADED';
    statusEl.className = 'hud-signal-degraded';
    return;
  }

  errorEl.textContent = anomaly.reconstruction_error.toFixed(4);
  thresholdEl.textContent = anomaly.threshold.toFixed(4);
  if (anomaly.is_anomaly) {
    statusEl.textContent = `ANOMALY (severity ${anomaly.severity.toFixed(2)})`;
    statusEl.className = 'hud-signal-none';
  } else {
    statusEl.textContent = 'NOMINAL';
    statusEl.className = 'hud-signal-optimal';
  }
}

async function loadAnomalyModelStatus() {
  const el = document.getElementById('anomaly-model-status');
  try {
    const data = await fetchJson('/api/cv/anomaly-status');
    if (data.model_loaded) {
      el.textContent = `LOADED (thr ${data.threshold.toFixed(4)})`;
      el.className = 'hud-signal-optimal';
    } else {
      el.textContent = 'NOT TRAINED';
      el.className = 'hud-signal-degraded';
    }
  } catch (err) {
    el.textContent = 'UNKNOWN';
  }
}

async function loadAnomalyEventLog() {
  const el = document.getElementById('anomaly-log');
  try {
    const data = await fetchJson('/api/cv/anomaly-log?limit=10');
    if (data.count === 0) {
      el.innerHTML = '<div class="dim">NO ANOMALIES LOGGED</div>';
      return;
    }
    el.innerHTML = data.events
      .map((e) => `<div class="det-row" style="border-left-color:#ff3355;">
        <b>${e.timestamp.replace('T', ' ').slice(0, 19)} UTC</b><br>
        <span class="dim">err ${e.reconstruction_error.toFixed(4)} / thr ${e.threshold.toFixed(4)} &middot; ${e.source}</span>
      </div>`)
      .join('');
  } catch (err) {
    el.innerHTML = `<div class="dim">LOG UNAVAILABLE</div>`;
  }
}

loadAnomalyModelStatus();
loadAnomalyEventLog();
setInterval(loadAnomalyEventLog, 10000);

// ===========================================================================
// Tab: DSN NOW — NASA Deep Space Network live dish status.
// ===========================================================================

async function loadDSN() {
  const dishesEl = document.getElementById('dsn-dishes');
  try {
    const status = await fetchJson('/api/sensors/dsn');

    const byStation = {};
    for (const dish of status.dishes) {
      (byStation[dish.station] = byStation[dish.station] || []).push(dish);
    }

    let html = '';
    for (const [station, dishes] of Object.entries(byStation)) {
      html += `<div class="dsn-station-head">${station.toUpperCase()}</div>`;
      for (const dish of dishes) {
        const isIdle = dish.target_name === 'DSN' || dish.target_name === 'DSS' || !dish.target_name;
        const activeSignal = dish.signals.find((s) => s.active && s.direction === 'down') || dish.signals.find((s) => s.active);
        const rateText = activeSignal ? formatDataRate(activeSignal.data_rate_bps) : 'IDLE';
        const bandText = activeSignal ? ` &middot; ${activeSignal.band}-BAND` : '';
        const rangeText = dish.downlink_range_km != null ? ` &middot; ${formatRangeKm(dish.downlink_range_km)}` : '';
        html += `<div class="dsn-dish-row ${isIdle ? 'idle' : ''}">
          <b>${dish.name}</b> &rarr; <span class="dish-target">${dish.target_name || 'IDLE'}</span>
          <span class="dish-rate" style="float:right;">${rateText}</span>
          <div class="dish-meta">${dish.activity}${bandText}${rangeText}</div>
        </div>`;
      }
    }
    dishesEl.innerHTML = html || '<div class="dim">NO DISH DATA</div>';

    document.getElementById('dsn-active-count').textContent =
      status.dishes.filter((d) => d.signals.some((s) => s.active)).length;
    document.getElementById('dsn-spacecraft-count').textContent = status.active_spacecraft.length;
    const totalDown = status.dishes
      .flatMap((d) => d.signals)
      .filter((s) => s.direction === 'down' && s.active)
      .reduce((sum, s) => sum + s.data_rate_bps, 0);
    document.getElementById('dsn-total-rate').textContent = formatDataRate(totalDown);

    document.getElementById('dsn-spacecraft-list').innerHTML = status.active_spacecraft.length
      ? status.active_spacecraft.map((s) => `<div class="dsn-dish-row">${s}</div>`).join('')
      : '<div class="dim">NONE CURRENTLY TRACKED</div>';

    setSignal('dsn-signal-status', 'optimal');
  } catch (err) {
    dishesEl.innerHTML = `<div class="dim">LINK DOWN: ${err.message || err}</div>`;
    setSignal('dsn-signal-status', 'none');
  }
}

// ===========================================================================
// Tab: TELESCOPES — most recently archived real Hubble/JWST observation (MAST).
// ===========================================================================

async function loadTelescopes() {
  const signalEl = document.getElementById('telescopes-signal');
  let anyFailed = false;

  for (const [key, prefix] of [['hubble', 'hst'], ['jwst', 'jwst']]) {
    try {
      const obs = await fetchJson(`/api/sensors/telescopes/${key}`);
      document.getElementById(`${prefix}-target`).textContent = obs.target_name;
      document.getElementById(`${prefix}-radec`).textContent = `${obs.ra_deg.toFixed(3)}°, ${obs.dec_deg.toFixed(3)}°`;
      document.getElementById(`${prefix}-instrument`).textContent = obs.instrument || '--';
      document.getElementById(`${prefix}-proposal`).textContent = obs.proposal_id ? `#${obs.proposal_id}` : '--';
      document.getElementById(`${prefix}-time`).textContent = obs.observed_at_utc
        ? obs.observed_at_utc.replace('T', ' ').slice(0, 19) + ' UTC'
        : '--';
    } catch (err) {
      anyFailed = true;
      document.getElementById(`${prefix}-target`).textContent = 'NO SIGNAL';
    }
  }

  signalEl.textContent = anyFailed ? 'LINK DEGRADED' : 'MAST ARCHIVE LINK ESTABLISHED';
  signalEl.className = anyFailed ? 'hud-signal-degraded' : 'hud-signal-optimal';
}

// ===========================================================================
// Tab: SOLAR — NASA SDO live imagery (direct public image URLs, no backend
// call needed — these are already publicly served static image endpoints).
// ===========================================================================

const SDO_CHANNEL_INFO = {
  '0171': '171Å (extreme UV) — the quiet corona and upper transition region, ~600,000 K plasma. Shows coronal loops well.',
  '0193': '193Å (extreme UV) — corona and hot flare plasma, ~1.25 million K. NASA\'s default "quiet Sun" view.',
  '0211': '211Å (extreme UV) — active regions, ~2 million K. Highlights magnetically active areas.',
  '0304': '304Å (extreme UV) — chromosphere and transition region, ~50,000 K. Shows prominences at the limb.',
  HMIIC: 'HMI Continuum — ordinary visible light. The photosphere as your eye would see it (through a solar filter), including sunspots.',
  HMIB: 'HMI Magnetogram — line-of-sight magnetic field strength. White/black = opposite magnetic polarity.',
};

function loadSolar() {
  const select = document.getElementById('solar-channel');
  const channel = select.value;
  const img = document.getElementById('solar-img');
  const cacheBust = Date.now();
  img.src = `https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_${channel}.jpg?_=${cacheBust}`;
  document.getElementById('solar-desc').textContent = SDO_CHANNEL_INFO[channel] || '';

  img.onload = () => setSignal('solar-signal', 'optimal');
  img.onerror = () => setSignal('solar-signal', 'none');
}

document.getElementById('solar-channel').addEventListener('change', loadSolar);
initCRTNoise(document.getElementById('solar-visual'));

// ===========================================================================
// Tab: EARTH — DSCOVR/EPIC latest real full-Earth photo.
// ===========================================================================

async function loadEarth() {
  const img = document.getElementById('earth-img');
  const ph = document.getElementById('earth-placeholder');
  try {
    const data = await fetchJson('/api/sensors/earth-epic');
    img.src = data.image_url;
    img.style.display = 'block';
    ph.style.display = 'none';
    document.getElementById('earth-centroid').textContent = `${data.centroid_lat.toFixed(1)}°, ${data.centroid_lon.toFixed(1)}°`;
    document.getElementById('earth-time').textContent = data.date + ' UTC';
    setSignal('earth-signal', 'optimal');
  } catch (err) {
    img.style.display = 'none';
    ph.style.display = 'block';
    ph.textContent = `NO SIGNAL: ${err.message || err}`;
    setSignal('earth-signal', 'none');
  }
}

initCRTNoise(document.getElementById('earth-visual'));

// ===========================================================================
// Tab: MARS ROVERS — latest real downlinked photos (Curiosity/Perseverance).
// ===========================================================================

let currentRover = 'curiosity';

async function loadMarsRovers() {
  const gridEl = document.getElementById('rover-grid');
  document.getElementById('mars-rover-name').textContent = currentRover.toUpperCase();
  try {
    const data = await fetchJson(`/api/sensors/mars-rover/${currentRover}`);
    if (data.count === 0) {
      gridEl.innerHTML = '<div class="dim">NO RECENT PHOTOS RETURNED</div>';
    } else {
      gridEl.innerHTML = data.photos
        .slice(0, 12)
        .map((p) => `
          <div class="rover-photo">
            <img src="${p.img_src}" alt="${p.camera_full_name}" loading="lazy" />
            <div class="rover-caption">${p.camera_name} &middot; SOL ${p.sol} &middot; ${p.earth_date}</div>
          </div>
        `)
        .join('');
      document.getElementById('mars-rover-status').textContent = data.photos[0].rover_status.toUpperCase();
      document.getElementById('mars-rover-sol').textContent = data.photos[0].sol;
    }
    setSignal('mars-signal', 'optimal');
  } catch (err) {
    gridEl.innerHTML = `<div class="dim">MARS UPLINK OFFLINE — ${err.message || err}<br><br>This is a real NASA API outage (this project's Mars Rover Photos client hit this during development too — see data/mars_rover_photos.py), not a simulated failure. It's cached and retried automatically once the service recovers.</div>`;
    document.getElementById('mars-rover-status').textContent = '--';
    document.getElementById('mars-rover-sol').textContent = '--';
    setSignal('mars-signal', 'none');
  }
}

document.querySelectorAll('#panel-mars .co-btn[data-rover]').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#panel-mars .co-btn[data-rover]').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    currentRover = btn.dataset.rover;
    loadMarsRovers();
  });
});
