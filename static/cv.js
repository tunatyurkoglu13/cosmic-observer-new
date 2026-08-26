// COSMIC OBSERVER — CV module frontend.
// Connects to /ws/cv, which alternates two message types per frame:
//   1. a binary WebSocket message: the JPEG-encoded, HUD-annotated frame
//   2. a text WebSocket message: JSON with that frame's detections + metrics
// Displayed via an <img> + object URL rather than <canvas>, since we
// receive an already-fully-rendered image (HUD baked in server-side by
// cv.hud) — no client-side drawing needed, which is both simpler and
// cheaper than re-rasterizing onto a canvas every frame.

let ws = null;
let currentObjectUrl = null;

const videoImg = document.getElementById('video-frame');
const placeholder = document.getElementById('video-placeholder');
const statusEl = document.getElementById('cv-status');
const startBtn = document.getElementById('cv-start-btn');

function setStatus(text) {
  statusEl.textContent = text;
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
    // A disclosed fallback (e.g. ISS live unreachable -> sample clip) —
    // informational, not fatal, so the stream keeps running.
    setStatus(`NOTICE: ${data.notice}`);
    return;
  }

  document.getElementById('stat-fps').textContent = data.fps.toFixed(1);
  document.getElementById('stat-frame').textContent = data.frame_index;
  document.getElementById('stat-targets').textContent = data.detections.length;
  document.getElementById('stat-conf').textContent = data.avg_confidence.toFixed(2);

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

startBtn.addEventListener('click', startStream);
document.getElementById('cv-upload-btn').addEventListener('click', uploadVideo);
