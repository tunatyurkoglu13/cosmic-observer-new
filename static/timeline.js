// COSMIC OBSERVER — TEMPORAL TRENDS frontend.
// Hand-rolled canvas line charts (no external charting library — this
// project keeps zero external JS dependencies where it reasonably can,
// and a single line/area chart is simple enough to draw directly).
// Two draw paths share one core routine: a small "sparkline" per metric
// tile (no axes/grid/tooltip) and one large focused chart (full axes,
// grid, hover tooltip) — see drawChart()'s `compact` option.

let metricMeta = {};       // metric key -> {display_name, unit, color_hex, latest_value, latest_timestamp}
let focusedMetric = null;
let focusedHours = 6;
let refreshTimer = null;

async function fetchJson(url) {
  const resp = await fetch(url);
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${resp.status}`);
  }
  return resp.json();
}

// ===========================================================================
// Chart drawing — shared by tile sparklines and the main focused chart.
// ===========================================================================

function drawChart(canvas, samples, opts) {
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const w = rect.width, h = rect.height;
  ctx.clearRect(0, 0, w, h);

  if (!samples || samples.length === 0) return;

  const compact = !!opts.compact;
  const padding = compact
    ? { left: 2, right: 2, top: 2, bottom: 2 }
    : { left: 54, right: 16, top: 10, bottom: 26 };
  const plotW = Math.max(1, w - padding.left - padding.right);
  const plotH = Math.max(1, h - padding.top - padding.bottom);

  const values = samples.map((s) => s.value);
  const times = samples.map((s) => new Date(s.timestamp).getTime());
  let minV = Math.min(...values);
  let maxV = Math.max(...values);
  if (minV === maxV) { minV -= Math.max(0.5, Math.abs(minV) * 0.1); maxV += Math.max(0.5, Math.abs(maxV) * 0.1); }
  const vPad = (maxV - minV) * 0.12;
  minV -= vPad; maxV += vPad;
  const minT = times[0];
  const maxT = times[times.length - 1] || minT + 1;

  const xFor = (t) => padding.left + ((t - minT) / (maxT - minT || 1)) * plotW;
  const yFor = (v) => padding.top + plotH - ((v - minV) / (maxV - minV || 1)) * plotH;

  const color = opts.color || '#00ffff';

  if (!compact) {
    // Grid + Y-axis value labels.
    ctx.strokeStyle = 'rgba(0,255,255,0.1)';
    ctx.lineWidth = 1;
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.fillStyle = 'rgba(0,255,255,0.5)';
    const gridLines = 4;
    for (let i = 0; i <= gridLines; i++) {
      const y = padding.top + (plotH / gridLines) * i;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(w - padding.right, y);
      ctx.stroke();
      const v = maxV - (maxV - minV) * (i / gridLines);
      ctx.fillText(v.toFixed(v >= 100 ? 0 : 2), 4, y + 3);
    }
    // X-axis time labels (start / end).
    ctx.fillText(new Date(minT).toISOString().slice(11, 16) + 'Z', padding.left, h - 8);
    ctx.textAlign = 'right';
    ctx.fillText(new Date(maxT).toISOString().slice(11, 16) + 'Z', w - padding.right, h - 8);
    ctx.textAlign = 'left';
  }

  // Area fill under the line.
  const grad = ctx.createLinearGradient(0, padding.top, 0, padding.top + plotH);
  grad.addColorStop(0, color + (compact ? '33' : '44'));
  grad.addColorStop(1, color + '00');
  ctx.beginPath();
  ctx.moveTo(xFor(times[0]), yFor(values[0]));
  for (let i = 1; i < samples.length; i++) ctx.lineTo(xFor(times[i]), yFor(values[i]));
  ctx.lineTo(xFor(times[times.length - 1]), padding.top + plotH);
  ctx.lineTo(xFor(times[0]), padding.top + plotH);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // The line itself, with a soft glow.
  ctx.beginPath();
  ctx.moveTo(xFor(times[0]), yFor(values[0]));
  for (let i = 1; i < samples.length; i++) ctx.lineTo(xFor(times[i]), yFor(values[i]));
  ctx.strokeStyle = color;
  ctx.lineWidth = compact ? 1.2 : 1.6;
  ctx.shadowColor = color;
  ctx.shadowBlur = compact ? 3 : 7;
  ctx.stroke();
  ctx.shadowBlur = 0;

  if (!compact) {
    ctx.fillStyle = color;
    for (let i = 0; i < samples.length; i++) {
      ctx.beginPath();
      ctx.arc(xFor(times[i]), yFor(values[i]), 1.6, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Stash the mapping so the tooltip handler can hit-test without redoing this work.
  canvas._chartGeometry = { samples, xFor, minT, maxT, padding, plotW };
}

function attachTooltip(canvas, tooltipEl) {
  canvas.addEventListener('mousemove', (event) => {
    const geo = canvas._chartGeometry;
    if (!geo || geo.samples.length === 0) {
      tooltipEl.style.display = 'none';
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const t = geo.minT + ((mouseX - geo.padding.left) / geo.plotW) * (geo.maxT - geo.minT);

    let nearest = geo.samples[0];
    let nearestDist = Infinity;
    for (const s of geo.samples) {
      const st = new Date(s.timestamp).getTime();
      const d = Math.abs(st - t);
      if (d < nearestDist) { nearestDist = d; nearest = s; }
    }

    tooltipEl.style.display = 'block';
    tooltipEl.style.left = `${mouseX + 12}px`;
    tooltipEl.style.top = `${event.clientY - rect.top - 24}px`;
    tooltipEl.innerHTML = `<b>${nearest.value.toFixed(4)}</b><br><span style="opacity:0.6">${nearest.timestamp.replace('T', ' ').slice(0, 19)} UTC</span>`;
  });
  canvas.addEventListener('mouseleave', () => { tooltipEl.style.display = 'none'; });
}

// ===========================================================================
// Metric tiles
// ===========================================================================

async function buildTiles() {
  metricMeta = await fetchJson('/api/timeseries');
  const row = document.getElementById('tile-row');
  row.innerHTML = '';

  const keys = Object.keys(metricMeta);
  if (!focusedMetric && keys.length > 0) focusedMetric = keys[0];

  for (const key of keys) {
    const meta = metricMeta[key];
    const tile = document.createElement('div');
    tile.className = 'metric-tile' + (key === focusedMetric ? ' focused' : '');
    tile.dataset.metric = key;

    const valueText = meta.latest_value != null ? meta.latest_value.toFixed(meta.latest_value >= 100 ? 1 : 3) : '--';
    const timeText = meta.latest_timestamp ? meta.latest_timestamp.replace('T', ' ').slice(0, 19) + ' UTC' : 'no samples yet';

    tile.innerHTML = `
      <div class="tile-name">${meta.display_name}</div>
      <div class="tile-value">${valueText}<span class="tile-unit">${meta.unit || ''}</span></div>
      <canvas class="sparkline"></canvas>
      <div class="tile-time">${timeText}</div>
    `;
    tile.addEventListener('click', () => {
      focusedMetric = key;
      document.querySelectorAll('.metric-tile').forEach((t) => t.classList.toggle('focused', t.dataset.metric === key));
      loadFocusedChart();
    });
    row.appendChild(tile);

    // Sparkline: a short recent window, independent of the main chart's range.
    fetchJson(`/api/timeseries/${key}?hours=6&limit=200`)
      .then((data) => {
        const canvas = tile.querySelector('canvas.sparkline');
        drawChart(canvas, data.samples, { compact: true, color: meta.color_hex });
      })
      .catch(() => {});
  }
}

// ===========================================================================
// Main focused chart
// ===========================================================================

async function loadFocusedChart() {
  if (!focusedMetric) return;
  const meta = metricMeta[focusedMetric];
  const titleEl = document.getElementById('chart-title');
  const valueNowEl = document.getElementById('chart-value-now');
  const canvas = document.getElementById('chart-canvas');
  const emptyEl = document.getElementById('chart-empty');
  const tooltipEl = document.getElementById('chart-tooltip');

  titleEl.childNodes[0].textContent = meta.display_name.toUpperCase();
  valueNowEl.textContent = meta.latest_value != null ? `${meta.latest_value.toFixed(3)} ${meta.unit || ''}` : '';

  try {
    const data = await fetchJson(`/api/timeseries/${focusedMetric}?hours=${focusedHours}&limit=2000`);
    if (data.samples.length === 0) {
      emptyEl.style.display = 'flex';
      canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    emptyEl.style.display = 'none';
    drawChart(canvas, data.samples, { compact: false, color: meta.color_hex });
  } catch (err) {
    emptyEl.style.display = 'flex';
    emptyEl.textContent = `FAILED TO LOAD: ${err.message || err}`;
  }
}

document.querySelectorAll('.range-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.range-btn').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    focusedHours = parseFloat(btn.dataset.hours);
    loadFocusedChart();
  });
});

attachTooltip(document.getElementById('chart-canvas'), document.getElementById('chart-tooltip'));
window.addEventListener('resize', () => loadFocusedChart());

async function refreshAll() {
  await buildTiles();
  await loadFocusedChart();
}

refreshAll();
refreshTimer = setInterval(refreshAll, 60000);
