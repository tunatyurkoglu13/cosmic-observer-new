// COSMIC OBSERVER — static-site adapter (GitHub Pages build only).
//
// The hosted showcase has no backend. scripts/build_static_site.py runs the
// real FastAPI app in-process, saves every read-only response under api/,
// and injects this file as the first script on every page. It answers the
// frontend's own /api/* fetches and /ws/* sockets from those files, so
// dashboard.js, cv.js, timeline.js and alerts.js run unmodified — the same
// code that talks to the live server.
//
// What can't be precomputed (video upload, zero-shot image identification,
// acknowledging alerts) answers with a clear "backend only" message.

(function () {
  'use strict';

  // Site root, resolved from this script's own URL so the build works under
  // any GitHub Pages sub-path (user.github.io/<repo>/) or a custom domain.
  const ROOT = new URL('.', document.currentScript.src);
  const REPO_URL = document.querySelector('meta[name="co-source"]')?.content || '';

  const nativeFetch = window.fetch.bind(window);
  const NativeWebSocket = window.WebSocket;

  const BACKEND_ONLY =
    'Runs on the Python backend only — this static showcase serves precomputed data. ' +
    'Clone the repo and run `docker compose up` to try it live.';

  const PAGE_LINKS = { '/': '', '/cv': 'cv.html', '/timeline': 'timeline.html', '/alerts': 'alerts.html' };

  // ---------------------------------------------------------------------
  // Snapshot files
  // ---------------------------------------------------------------------

  function snapshotUrl(path, ext = '.json') {
    const safe = path.split('/').filter(Boolean).map(encodeURIComponent).join('/');
    return new URL(`api/${safe}${ext}`, ROOT);
  }

  async function loadJson(path) {
    try {
      const resp = await nativeFetch(snapshotUrl(path));
      return resp.ok ? await resp.json() : null;
    } catch (err) {
      return null;
    }
  }

  let buildInfoPromise = null;
  function buildInfo() {
    if (!buildInfoPromise) buildInfoPromise = loadJson('_build');
    return buildInfoPromise;
  }

  function jsonResponse(body, status = 200) {
    return new Response(JSON.stringify(body), {
      status,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  // The build stores the backend's 400 answers (e.g. "inclination not
  // reachable from this site") as {__status, detail}; replay them as such.
  function snapshotResponse(data, transform) {
    if (data && data.__status) return jsonResponse({ detail: data.detail }, data.__status);
    return transform ? Promise.resolve(transform(data)).then((d) => jsonResponse(d)) : jsonResponse(data);
  }

  function formatUtc(iso) {
    if (!iso) return '--';
    const d = new Date(iso);
    const date = d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric', timeZone: 'UTC' });
    return `${date} · ${d.toISOString().slice(11, 16)} UTC`;
  }

  // ---------------------------------------------------------------------
  // fetch('/api/...') routing
  // ---------------------------------------------------------------------

  // Query parameters the live endpoints apply server-side; a static host
  // ignores query strings, so the same filtering happens here instead.
  // Time windows are measured back from the build time (when the data was
  // captured), not from "now" — the snapshot is a picture of that moment.
  const QUERY_HANDLERS = {
    async timeseries(data, params) {
      const info = await buildInfo();
      const ref = info ? Date.parse(info.generated_at) : Date.now();
      const since = ref - parseFloat(params.get('hours') || '24') * 3600e3;
      const limit = parseInt(params.get('limit') || '2000', 10);
      const samples = data.samples.filter((s) => Date.parse(s.timestamp) >= since).slice(0, limit);
      return { ...data, count: samples.length, samples };
    },
    alerts(data, params) {
      const category = params.get('category');
      const unackedOnly = params.get('unacknowledged_only') === 'true';
      const limit = parseInt(params.get('limit') || '100', 10);
      const alerts = data.alerts
        .filter((a) => !category || a.category === category)
        .filter((a) => !unackedOnly || !a.acknowledged)
        .slice(0, limit);
      return { alerts };
    },
    'anomaly-log'(data, params) {
      const events = data.events.slice(0, parseInt(params.get('limit') || '50', 10));
      return { count: events.length, events };
    },
  };

  function queryHandlerFor(path) {
    if (path.startsWith('timeseries/')) return QUERY_HANDLERS.timeseries;
    if (path === 'alerts') return QUERY_HANDLERS.alerts;
    if (path === 'cv/anomaly-log') return QUERY_HANDLERS['anomaly-log'];
    return null;
  }

  async function routeGet(path, params) {
    const data = await loadJson(path);
    if (data === null) return jsonResponse({ detail: 'Not included in this static snapshot.' }, 404);
    const handler = queryHandlerFor(path);
    return snapshotResponse(data, handler && ((d) => handler(d, params)));
  }

  async function routePost(path, init) {
    let body = {};
    try { body = JSON.parse(init.body || '{}'); } catch (err) { /* FormData etc. */ }

    if (path === 'cola/scan') {
      const data = await loadJson(`cola/scan/${body.site}`);
      return data ? snapshotResponse(data) : jsonResponse({ detail: 'No precomputed scan for this launch site.' }, 404);
    }
    if (path === 'cola/trajectory') {
      // Same key the build uses: the candidate's launch_time with every
      // non-digit stripped — the dashboard sends it back verbatim.
      const key = String(body.launch_time || '').replace(/\D/g, '');
      const data = await loadJson(`cola/trajectory/${body.site}/${key}`);
      return data ? snapshotResponse(data) : jsonResponse({ detail: 'Trajectory not in this snapshot.' }, 404);
    }
    return jsonResponse({ detail: BACKEND_ONLY }, 501);
  }

  window.fetch = function (input, init) {
    const raw = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    let url;
    try { url = new URL(raw, window.location.href); } catch (err) { return nativeFetch(input, init); }
    if (url.origin !== window.location.origin || !url.pathname.startsWith('/api/')) {
      return nativeFetch(input, init);
    }

    const path = url.pathname.slice('/api/'.length).replace(/\/+$/, '');
    const method = ((init && init.method) || (input instanceof Request ? input.method : 'GET')).toUpperCase();
    return method === 'GET' ? routeGet(path, url.searchParams) : routePost(path, init || {});
  };

  // ---------------------------------------------------------------------
  // WebSockets
  // ---------------------------------------------------------------------

  // Minimal WebSocket stand-in: supports both on<event> handlers and
  // addEventListener, which is everything the frontend uses.
  class SnapshotSocket extends EventTarget {
    constructor(url) {
      super();
      this.url = url;
      this.readyState = 0;
      this.binaryType = 'blob';
      this.onopen = this.onmessage = this.onerror = this.onclose = null;
      setTimeout(() => this._open(), 0);
    }

    _emit(type, data) {
      const event = type === 'message' ? new MessageEvent('message', { data }) : new Event(type);
      this.dispatchEvent(event);
      const handler = this[`on${type}`];
      if (typeof handler === 'function') handler.call(this, event);
    }

    _open() {
      if (this.readyState !== 0) return;
      this.readyState = 1;
      this._emit('open');
      this._start();
    }

    _start() {}
    _stop() {}

    send() {}

    close() {
      if (this.readyState >= 2) return;
      this.readyState = 3;
      this._stop();
      this._emit('close');
    }
  }

  // /ws/alerts — alerts are refreshed per build, not pushed. The alerts
  // page labels an open socket "LIVE"; relabel it once that has run.
  class AlertSnapshotSocket extends SnapshotSocket {
    _start() {
      const link = document.getElementById('stat-link');
      if (!link) return;
      link.textContent = 'SNAPSHOT';
      link.style.color = 'var(--co-amber)';
      link.title = 'Alerts are re-scanned on every data refresh, not pushed live.';
    }
  }

  // /ws/cv — replays real YOLOv8 inference frames recorded by the build
  // (binary JPEG then JSON metadata per frame, exactly like the server).
  class CvReplaySocket extends SnapshotSocket {
    async _start() {
      const params = new URL(this.url).searchParams;
      const recording = await loadJson('cv/stream/meta');
      if (this.readyState !== 1) return;
      if (!recording || !recording.frames.length) {
        this._emit('message', JSON.stringify({ error: BACKEND_ONLY }));
        return;
      }

      const needsBackend = { upload: 'Uploaded video', iss_live: 'The ISS live feed' }[params.get('source')];
      const otherModel = recording.model && params.get('model') && params.get('model') !== recording.model;
      const notice = [
        `SNAPSHOT MODE — replaying real YOLOv8 inference recorded ${formatUtc(recording.recorded_at)}`,
        `(${recording.label}).`,
        otherModel ? `Recorded with the ${recording.model.toUpperCase()} model.` : '',
        needsBackend ? `${needsBackend} needs the live backend.` : '',
      ].filter(Boolean).join(' ');

      const blobs = await Promise.all(
        recording.frames.map((f) => nativeFetch(snapshotUrl(`cv/stream/${f.file}`, '')).then((r) => r.blob()))
      );
      if (this.readyState !== 1) return;
      this._emit('message', JSON.stringify({ notice }));

      let i = 0;
      const tick = async () => {
        if (this.readyState !== 1) return;
        const n = i % recording.frames.length;
        const frame = this.binaryType === 'arraybuffer' ? await blobs[n].arrayBuffer() : blobs[n];
        this._emit('message', frame);
        this._emit('message', JSON.stringify(recording.frames[n].meta));
        i += 1;
        this._timer = setTimeout(tick, 1000 / recording.fps);
      };
      tick();
    }

    _stop() {
      clearTimeout(this._timer);
    }
  }

  function SnapshotWebSocket(url, protocols) {
    const path = new URL(url, window.location.href).pathname;
    if (path.endsWith('/ws/cv')) return new CvReplaySocket(url);
    if (path.endsWith('/ws/alerts')) return new AlertSnapshotSocket(url);
    return new NativeWebSocket(url, protocols);
  }
  Object.assign(SnapshotWebSocket, { CONNECTING: 0, OPEN: 1, CLOSING: 2, CLOSED: 3 });
  window.WebSocket = SnapshotWebSocket;

  // ---------------------------------------------------------------------
  // Page touch-ups (run after every page's own scripts have executed)
  // ---------------------------------------------------------------------

  function rewriteRootLinks() {
    // Pages link to backend routes like "/cv"; on GitHub Pages the site
    // lives under /<repo>/, so point them at the built .html files.
    for (const a of document.querySelectorAll('a[href^="/"]:not([href^="//"])')) {
      const target = PAGE_LINKS[a.getAttribute('href')];
      if (target !== undefined) a.href = new URL(target, ROOT).href;
    }
  }

  function lockBackendOnlyControls() {
    const lock = (id, hint) => {
      const el = document.getElementById(id);
      if (!el) return;
      el.disabled = true;
      if (hint) el.title = hint;
    };

    // Mission planner: one scenario per launch site is precomputed with
    // the form's default parameters, so only the site stays selectable.
    for (const id of ['mp-incl', 'mp-alt', 'mp-hours', 'mp-bubble']) {
      lock(id, 'Fixed in snapshot mode — the scan is precomputed with these parameters.');
    }
    const mpButton = document.getElementById('mp-compute-btn');
    if (mpButton) {
      mpButton.insertAdjacentHTML(
        'afterend',
        '<div class="co-snapshot-hint">Scenario precomputed for every launch site on each data refresh — pick a site and run it.</div>'
      );
    }

    // Sensor feeds: upload + zero-shot identification need live inference.
    for (const id of ['cv-file', 'cv-upload-btn', 'identify-file', 'identify-queries', 'identify-btn']) {
      lock(id, BACKEND_ONLY);
    }
    const uploadOption = document.querySelector('#cv-source option[value="upload"]');
    if (uploadOption) uploadOption.disabled = true;
    const identifyPanel = document.getElementById('identify-panel');
    if (identifyPanel) {
      identifyPanel.insertAdjacentHTML(
        'beforeend',
        '<div class="co-snapshot-hint">Needs the live Python backend (OWL-ViT runs server-side).</div>'
      );
    }
  }

  async function addSnapshotChip() {
    const panel = document.getElementById('title-panel');
    if (!panel) return;

    const chip = document.createElement('details');
    chip.className = 'co-snapshot-chip';
    chip.innerHTML = `
      <summary><span class="co-snapshot-dot"></span><span>SNAPSHOT</span><span class="co-snapshot-time">--</span></summary>
      <div class="co-snapshot-body">
        <p>Static showcase hosted on GitHub Pages. There is no server behind this page:
        a scheduled GitHub Actions job runs the real Python backend headless, captures every
        read-only API response, and republishes the site.</p>
        <p class="co-snapshot-meta"></p>
        <p>Interactive inference (video upload, zero-shot identification) needs the live backend —
        see the README to run it locally with Docker.</p>
        ${REPO_URL ? `<a class="co-snapshot-link" href="${REPO_URL}" target="_blank" rel="noopener noreferrer">SOURCE CODE ↗</a>` : ''}
      </div>`;
    const anchor = panel.querySelector('.subtitle');
    if (anchor) anchor.after(chip); else panel.appendChild(chip);

    const info = await buildInfo();
    if (!info) return;
    chip.querySelector('.co-snapshot-time').textContent = formatUtc(info.generated_at);
    const meta = [`Data captured ${formatUtc(info.generated_at)}`, `refreshed every ${info.refresh_hours} h`];
    if (info.commit) meta.push(`build ${info.commit.slice(0, 7)}`);
    chip.querySelector('.co-snapshot-meta').textContent = meta.join(' · ');
  }

  document.addEventListener('DOMContentLoaded', () => {
    document.documentElement.classList.add('co-static');
    rewriteRootLinks();
    lockBackendOnlyControls();
    addSnapshotChip();
  });
})();
