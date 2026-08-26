// COSMIC OBSERVER — retro-futuristic 3D dashboard
// Consumes static/snapshot.json (built by scripts/generate_dashboard_snapshot.py)
// and renders a holographic globe with satellite tracking overlays.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const R_EARTH_KM = 6378.137;
const GLOBE_RADIUS = 4.0; // scene units representing R_EARTH_KM

const PALETTE = {
  cyan: 0x00ffff,
  magenta: 0xff0066,
  amber: 0xffcc00,
  phosphorGreen: 0x00ff66,
};

const CLASS_COLOR = {
  stations: PALETTE.amber,
  visual: PALETTE.cyan,
  active: PALETTE.cyan,
  debris: PALETTE.magenta,
};

const ISS_NORAD_ID = 25544;

function latLonAltToVector3(latDeg, lonDeg, altKm, out) {
  const radius = GLOBE_RADIUS * ((R_EARTH_KM + (altKm || 0)) / R_EARTH_KM);
  const lat = THREE.MathUtils.degToRad(latDeg);
  const lon = THREE.MathUtils.degToRad(lonDeg);
  const x = radius * Math.cos(lat) * Math.cos(lon);
  const z = radius * Math.cos(lat) * Math.sin(lon);
  const y = radius * Math.sin(lat);
  return out ? out.set(x, y, z) : new THREE.Vector3(x, y, z);
}

async function extractErrorDetail(resp) {
  // FastAPI's HTTPException(status, "message") responses come back as
  // {"detail": "message"} — surface that directly instead of a bare
  // status code, since our backend now puts a genuinely useful
  // explanation there (e.g. "CelesTrak is unreachable" vs. a real bug).
  try {
    const body = await resp.json();
    if (body && body.detail) return body.detail;
  } catch (_) {
    // response wasn't JSON — fall through to the generic message
  }
  return `HTTP ${resp.status}`;
}

function makeCircleSprite(colorHex) {
  const size = 64;
  const canvas = document.createElement('canvas');
  canvas.width = size; canvas.height = size;
  const ctx = canvas.getContext('2d');
  const grad = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  const c = new THREE.Color(colorHex);
  const rgb = `${Math.round(c.r * 255)},${Math.round(c.g * 255)},${Math.round(c.b * 255)}`;
  grad.addColorStop(0, `rgba(${rgb},1)`);
  grad.addColorStop(0.4, `rgba(${rgb},0.8)`);
  grad.addColorStop(1, `rgba(${rgb},0)`);
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  return new THREE.CanvasTexture(canvas);
}

const EARTH_VERTEX_SHADER = `
varying vec3 vNormal;
void main() {
  // Deliberately OBJECT-space (not normalMatrix * normal, which is
  // view-space and rotates with the camera). The Earth mesh itself
  // never rotates or moves — only the camera orbits around it via
  // OrbitControls — so object space IS world space here, and it must
  // match the fixed Earth frame every satellite/launch-site/ground-track
  // position is placed in (see latLonAltToVector3). Using the view-space
  // normal made the lat/lon computed in the fragment shader silently
  // drift as the camera moved, desyncing the map from real positions.
  vNormal = normalize(normal);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const EARTH_FRAGMENT_SHADER = `
uniform vec3 uSunDir;
uniform sampler2D uDayMap;
uniform sampler2D uNightMap;
uniform vec3 uGridColor;
varying vec3 vNormal;

#define PI 3.14159265

void main() {
  vec3 n = normalize(vNormal);
  float ndotl = dot(n, normalize(uSunDir));

  // Same lat/lon this app computes everywhere else (core.propagator /
  // dashboard.js latLonAltToVector3), re-derived from the surface
  // normal so the day/night texture lookup lines up exactly with where
  // satellites, launch sites, and ground tracks are plotted.
  float lat = asin(clamp(n.y, -1.0, 1.0));
  float lon = atan(n.z, n.x);
  vec2 uv = vec2(lon / (2.0 * PI) + 0.5, 0.5 - lat / PI);

  vec3 dayColor = texture2D(uDayMap, uv).rgb;
  vec3 nightColor = texture2D(uNightMap, uv).rgb;
  vec3 base = mix(nightColor, dayColor, smoothstep(-0.15, 0.15, ndotl));

  // Faint reference grid every 30 deg, kept subtle so the real map reads
  // clearly underneath — a HUD overlay, not the primary visual anymore.
  float latGrid = abs(fract(lat / PI * 6.0 + 0.5) - 0.5) * 2.0;
  float lonGrid = abs(fract(lon / PI * 6.0 + 0.5) - 0.5) * 2.0;
  float gridLine = 1.0 - smoothstep(0.0, 0.02, min(latGrid, lonGrid));

  vec3 color = mix(base, uGridColor, gridLine * 0.15);
  gl_FragColor = vec4(color, 1.0);
}
`;

const ATMOSPHERE_VERTEX_SHADER = `
varying vec3 vNormal;
void main() {
  vNormal = normalize(normalMatrix * normal);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const ATMOSPHERE_FRAGMENT_SHADER = `
uniform vec3 uGlowColor;
varying vec3 vNormal;
void main() {
  float intensity = pow(0.65 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 3.0);
  gl_FragColor = vec4(uGlowColor, clamp(intensity, 0.0, 1.0));
}
`;

class Dashboard {
  constructor(snapshot) {
    this.snapshot = snapshot;
    this.frameIndex = 0;
    this.playing = false;

    this._initScene();
    this._buildEarth();
    this._buildSatellites();
    this._buildGroundTracks();
    this._buildTerminator();
    this._initHud();
    this._initPicking();
    this._buildLaunchSites();
    this._initMissionPlanner();

    this._updateFrame(0);
    this._animate();
  }

  _initScene() {
    const container = document.getElementById('scene-container');
    this.scene = new THREE.Scene();

    this.camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 1000);
    this.camera.position.set(0, 5, 12);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.minDistance = 6;
    this.controls.maxDistance = 40;
    // Slow idle spin so the globe reads as a living 3D world (continents
    // visibly turning) rather than a static image; OrbitControls pauses
    // this automatically while the user is dragging and resumes after.
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 0.35;

    window.addEventListener('resize', () => {
      this.camera.aspect = window.innerWidth / window.innerHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(window.innerWidth, window.innerHeight);
    });
  }

  _buildEarth() {
    // Real day/night Earth imagery (NASA-derived textures shipped with
    // three.js's own examples) so continents are actually recognizable —
    // sampled in the shader using this app's own lat/lon convention (see
    // EARTH_FRAGMENT_SHADER), so the map lines up exactly with where
    // satellites, launch sites, and ground tracks get plotted.
    const textureLoader = new THREE.TextureLoader();
    const TEXTURE_BASE = 'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r160/examples/textures/planets/';
    const dayMap = textureLoader.load(TEXTURE_BASE + 'earth_atmos_2048.jpg');
    const nightMap = textureLoader.load(TEXTURE_BASE + 'earth_lights_2048.png');
    dayMap.colorSpace = THREE.SRGBColorSpace;
    nightMap.colorSpace = THREE.SRGBColorSpace;
    // THREE.TextureLoader defaults flipY=true (auto-flips the image
    // vertically on upload, since WebGL's texture-space origin is
    // bottom-left while image files are stored top-down). Our fragment
    // shader computes UV directly from real lat/lon with row 0 = north
    // pole (verified pixel-by-pixel against the raw file), so that
    // automatic flip must be disabled or the map renders upside-down.
    dayMap.flipY = false;
    nightMap.flipY = false;

    const geometry = new THREE.SphereGeometry(GLOBE_RADIUS, 64, 64);
    this.earthMaterial = new THREE.ShaderMaterial({
      uniforms: {
        uSunDir: { value: new THREE.Vector3(1, 0, 0) },
        uDayMap: { value: dayMap },
        uNightMap: { value: nightMap },
        uGridColor: { value: new THREE.Color(PALETTE.cyan) },
      },
      vertexShader: EARTH_VERTEX_SHADER,
      fragmentShader: EARTH_FRAGMENT_SHADER,
    });
    this.earth = new THREE.Mesh(geometry, this.earthMaterial);
    this.scene.add(this.earth);

    const atmoGeometry = new THREE.SphereGeometry(GLOBE_RADIUS * 1.08, 64, 64);
    const atmoMaterial = new THREE.ShaderMaterial({
      uniforms: { uGlowColor: { value: new THREE.Color(PALETTE.cyan) } },
      vertexShader: ATMOSPHERE_VERTEX_SHADER,
      fragmentShader: ATMOSPHERE_FRAGMENT_SHADER,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
      transparent: true,
    });
    this.scene.add(new THREE.Mesh(atmoGeometry, atmoMaterial));
  }

  _buildSatellites() {
    // Group satellites by classification into separate Points objects (one draw call each).
    this.groups = {};
    const byClass = {};
    for (const sat of this.snapshot.satellites) {
      const cls = sat.classification in CLASS_COLOR ? sat.classification : 'active';
      (byClass[cls] = byClass[cls] || []).push(sat);
    }

    this.satelliteIndex = []; // flat list of {sat, groupKey, indexInGroup} for picking

    for (const [cls, sats] of Object.entries(byClass)) {
      const color = CLASS_COLOR[cls] || PALETTE.cyan;
      const positions = new Float32Array(sats.length * 3);
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

      const material = new THREE.PointsMaterial({
        size: cls === 'stations' ? 0.22 : 0.12,
        map: makeCircleSprite(color),
        transparent: true,
        depthWrite: false,
        sizeAttenuation: true,
      });

      const points = new THREE.Points(geometry, material);
      this.scene.add(points);
      this.groups[cls] = { points, sats };

      sats.forEach((sat, i) => this.satelliteIndex.push({ sat, cls, i }));
    }

    // Special highlighted marker for the ISS.
    const issSat = this.snapshot.satellites.find((s) => s.norad_id === ISS_NORAD_ID);
    if (issSat) {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3), 3));
      const material = new THREE.PointsMaterial({
        size: 0.35, map: makeCircleSprite(PALETTE.phosphorGreen),
        transparent: true, depthWrite: false, sizeAttenuation: true,
      });
      this.issPoint = new THREE.Points(geometry, material);
      this.scene.add(this.issPoint);
      this.issSat = issSat;
    }
  }

  _buildGroundTracks() {
    this.trackLines = [];
    const color = new THREE.Color(PALETTE.amber);
    for (const [noradId, segments] of Object.entries(this.snapshot.ground_tracks || {})) {
      for (const seg of segments) {
        if (seg.length < 2) continue;
        const points = seg.map(([lat, lon]) => latLonAltToVector3(lat, lon, 15));
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.45 });
        const line = new THREE.Line(geometry, material);
        this.scene.add(line);
        this.trackLines.push(line);
      }
    }
  }

  _buildTerminator() {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3));
    const material = new THREE.LineBasicMaterial({ color: PALETTE.cyan, transparent: true, opacity: 0.5 });
    this.terminatorLine = new THREE.LineLoop(geometry, material);
    this.scene.add(this.terminatorLine);
  }

  _initHud() {
    this.slider = document.getElementById('time-slider');
    this.slider.max = this.snapshot.frame_times_iso.length - 1;
    this.slider.addEventListener('input', () => this._updateFrame(parseInt(this.slider.value, 10)));

    this.playBtn = document.getElementById('play-btn');
    this.playBtn.addEventListener('click', () => {
      this.playing = !this.playing;
      this.playBtn.textContent = this.playing ? '⏸ PAUSE' : '▶ PLAY';
      this.playBtn.classList.toggle('active', this.playing);
    });

    document.getElementById('stat-count').textContent = this.snapshot.satellites.length;
  }

  _initPicking() {
    this.raycaster = new THREE.Raycaster();
    this.raycaster.params.Points.threshold = 0.12;
    this.mouse = new THREE.Vector2();

    this.renderer.domElement.addEventListener('click', (event) => {
      const rect = this.renderer.domElement.getBoundingClientRect();
      this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      this._pick();
    });
  }

  _pick() {
    this.raycaster.setFromCamera(this.mouse, this.camera);
    const pointObjects = Object.values(this.groups).map((g) => g.points);
    if (this.issPoint) pointObjects.push(this.issPoint);

    const intersects = this.raycaster.intersectObjects(pointObjects);
    if (intersects.length === 0) return;

    const hit = intersects[0];
    let sat = null;
    if (this.issPoint && hit.object === this.issPoint) {
      sat = this.issSat;
    } else {
      for (const [cls, group] of Object.entries(this.groups)) {
        if (group.points === hit.object) {
          sat = group.sats[hit.index];
          break;
        }
      }
    }
    if (sat) this._showInfo(sat);
  }

  _showInfo(sat) {
    const [lat, lon, alt] = sat.track[Math.min(this.frameIndex, sat.track.length - 1)];
    document.getElementById('info-panel').innerHTML = `
      <table>
        <tr><td>NAME</td><td>${sat.name}</td></tr>
        <tr><td>NORAD ID</td><td class="value">${sat.norad_id}</td></tr>
        <tr><td>CLASS</td><td class="value">${sat.classification.toUpperCase()}</td></tr>
        <tr><td>LAT</td><td class="value">${lat.toFixed(2)}&deg;</td></tr>
        <tr><td>LON</td><td class="value">${lon.toFixed(2)}&deg;</td></tr>
        <tr><td>ALT</td><td class="value">${alt.toFixed(1)} km</td></tr>
      </table>`;
  }

  _updateFrame(frameIndex) {
    this.frameIndex = frameIndex;
    this.slider.value = frameIndex;

    const tmp = new THREE.Vector3();

    for (const group of Object.values(this.groups)) {
      const posAttr = group.points.geometry.attributes.position;
      group.sats.forEach((sat, i) => {
        const idx = Math.min(frameIndex, sat.track.length - 1);
        const [lat, lon, alt] = sat.track[idx];
        latLonAltToVector3(lat, lon, alt, tmp);
        posAttr.setXYZ(i, tmp.x, tmp.y, tmp.z);
      });
      posAttr.needsUpdate = true;
    }

    if (this.issPoint && this.issSat) {
      const idx = Math.min(frameIndex, this.issSat.track.length - 1);
      const [lat, lon, alt] = this.issSat.track[idx];
      latLonAltToVector3(lat, lon, alt, tmp);
      this.issPoint.geometry.attributes.position.setXYZ(0, tmp.x, tmp.y, tmp.z);
      this.issPoint.geometry.attributes.position.needsUpdate = true;
    }

    const [subLat, subLon] = this.snapshot.subsolar_track[frameIndex];
    const sunDir = latLonAltToVector3(subLat, subLon, 0).normalize();
    this.earthMaterial.uniforms.uSunDir.value.copy(sunDir);

    const terminator = this.snapshot.terminator_tracks[frameIndex];
    const termPoints = terminator.map(([lat, lon]) => latLonAltToVector3(lat, lon, 5));
    this.terminatorLine.geometry.dispose();
    this.terminatorLine.geometry = new THREE.BufferGeometry().setFromPoints(termPoints);

    document.getElementById('stat-frame').textContent = `${frameIndex + 1} / ${this.snapshot.frame_times_iso.length}`;
    document.getElementById('stat-epoch').textContent = this.snapshot.frame_times_iso[frameIndex].replace('T', ' ').slice(0, 19);
    document.getElementById('stat-subsolar').textContent = `${subLat.toFixed(1)}°, ${subLon.toFixed(1)}°`;
    document.getElementById('time-readout').textContent = this.snapshot.frame_times_iso[frameIndex].replace('T', ' ').slice(0, 19) + ' UTC';
  }

  _animate() {
    requestAnimationFrame(() => this._animate());
    this.controls.update();

    if (this.playing) {
      this._playAccum = (this._playAccum || 0) + 1;
      if (this._playAccum > 20) {
        this._playAccum = 0;
        const next = (this.frameIndex + 1) % this.snapshot.frame_times_iso.length;
        this._updateFrame(next);
      }
    }

    this.renderer.render(this.scene, this.camera);
  }

  // -------------------------------------------------------------------
  // Mission Planner — COLA (Collision On Launch Assessment)
  // -------------------------------------------------------------------

  _buildLaunchSites() {
    this.launchSiteMarkers = {};
    fetch('/api/launch-sites')
      .then((r) => r.json())
      .then((sites) => {
        this.launchSites = sites;
        const select = document.getElementById('mp-site');
        select.innerHTML = '';
        for (const [key, site] of Object.entries(sites)) {
          const opt = document.createElement('option');
          opt.value = key;
          opt.textContent = site.name;
          select.appendChild(opt);

          const geometry = new THREE.SphereGeometry(0.045, 8, 8);
          const material = new THREE.MeshBasicMaterial({ color: 0xffffff });
          const marker = new THREE.Mesh(geometry, material);
          const pos = latLonAltToVector3(site.lat_deg, site.lon_deg, 5);
          marker.position.copy(pos);
          this.scene.add(marker);
          this.launchSiteMarkers[key] = marker;
        }
      })
      .catch((err) => console.error('Failed to load launch sites:', err));
  }

  _initMissionPlanner() {
    this.trajectoryLine = null;
    this.selectedCandidateEl = null;

    document.getElementById('mp-compute-btn').addEventListener('click', () => this._runColaScan());
  }

  async _runColaScan() {
    const statusEl = document.getElementById('mp-status');
    const resultsEl = document.getElementById('mp-results');
    const site = document.getElementById('mp-site').value;
    const inclination = parseFloat(document.getElementById('mp-incl').value);
    const altitude = parseFloat(document.getElementById('mp-alt').value);
    const hours = parseFloat(document.getElementById('mp-hours').value);
    const bubble = parseFloat(document.getElementById('mp-bubble').value);

    if (!site) return;

    statusEl.textContent = 'SCANNING CATALOG…';
    resultsEl.innerHTML = '';
    this._clearTrajectory();

    const now = new Date();
    const end = new Date(now.getTime() + hours * 3600 * 1000);

    let data;
    try {
      const resp = await fetch('/api/cola/scan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          site,
          target_inclination_deg: inclination,
          target_altitude_km: altitude,
          search_start: now.toISOString(),
          search_end: end.toISOString(),
          candidate_step_minutes: Math.max(1, Math.round(hours * 60 / 60)), // ~60 candidates across the window
          bubble_radius_km: bubble,
          catalog_groups: ['stations', 'visual'],
          catalog_limit_per_group: 150,
        }),
      });
      if (!resp.ok) throw new Error(await extractErrorDetail(resp));
      data = await resp.json();
    } catch (err) {
      statusEl.textContent = `HATA: ${err.message || err}`;
      return;
    }

    const clearCount = data.candidates.filter((c) => c.clear).length;
    statusEl.textContent = `${data.objects_screened} OBJECTS SCREENED — ${clearCount}/${data.candidates.length} WINDOWS CLEAR`;

    this._lastColaRequest = { site, inclination, altitude, bubble };

    for (const candidate of data.candidates) {
      const row = document.createElement('div');
      row.className = `mp-candidate ${candidate.clear ? 'clear' : 'blocked'}`;
      const t = new Date(candidate.launch_time);
      const timeStr = t.toISOString().slice(11, 19) + ' UTC';
      const statusStr = candidate.clear ? 'CLEAR' : `BLOCKED (${candidate.violations.length})`;

      let html = `<div><b>${timeStr}</b> — az ${candidate.azimuth_deg.toFixed(1)}° — ${statusStr}</div>`;
      html += `<div class="dim">closest approach: ${candidate.closest_approach_km.toFixed(1)} km</div>`;
      if (!candidate.clear) {
        const v = candidate.violations[0];
        html += `<div class="mp-violation">⚠ ${v.satellite_name} @ ${v.distance_km.toFixed(1)} km (T+${v.t_offset_s.toFixed(0)}s)</div>`;
      }
      row.innerHTML = html;

      row.addEventListener('click', () => {
        if (this.selectedCandidateEl) this.selectedCandidateEl.classList.remove('selected');
        row.classList.add('selected');
        this.selectedCandidateEl = row;
        this._drawTrajectory(candidate.launch_time);
      });

      resultsEl.appendChild(row);
    }

    if (data.candidates.length > 0) {
      resultsEl.firstChild.click();
    }
  }

  _clearTrajectory() {
    if (this.trajectoryLine) {
      this.scene.remove(this.trajectoryLine);
      this.trajectoryLine.geometry.dispose();
      this.trajectoryLine.material.dispose();
      this.trajectoryLine = null;
    }
  }

  async _drawTrajectory(launchTimeIso) {
    const req = this._lastColaRequest;
    if (!req) return;

    let data;
    try {
      const resp = await fetch('/api/cola/trajectory', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          site: req.site,
          target_inclination_deg: req.inclination,
          target_altitude_km: req.altitude,
          launch_time: launchTimeIso,
          bubble_radius_km: req.bubble,
          catalog_groups: ['stations', 'visual'],
          catalog_limit_per_group: 150,
        }),
      });
      if (!resp.ok) throw new Error(await extractErrorDetail(resp));
      data = await resp.json();
    } catch (err) {
      document.getElementById('mp-status').textContent = `HATA: ${err.message || err}`;
      return;
    }

    this._clearTrajectory();

    const positions = [];
    const colors = [];
    const bubbleRadius = data.bubble_radius_km;

    for (const sample of data.trajectory) {
      const p = latLonAltToVector3(sample.lat_deg, sample.lon_deg, sample.alt_km);
      positions.push(p.x, p.y, p.z);

      let color;
      if (sample.closest_distance_km < bubbleRadius) {
        color = new THREE.Color(0xff0033); // violation: red
      } else if (sample.closest_distance_km < bubbleRadius * 3) {
        color = new THREE.Color(0xffcc00); // caution: amber
      } else {
        color = new THREE.Color(0x00ff66); // clear: phosphor green
      }
      colors.push(color.r, color.g, color.b);
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    const material = new THREE.LineBasicMaterial({ vertexColors: true, linewidth: 2 });
    this.trajectoryLine = new THREE.Line(geometry, material);
    this.scene.add(this.trajectoryLine);
  }
}

fetch('snapshot.json')
  .then((r) => r.json())
  .then((snapshot) => {
    document.getElementById('loading').style.display = 'none';
    window.dashboard = new Dashboard(snapshot);
  })
  .catch((err) => {
    document.getElementById('loading').textContent = 'FAILED TO LOAD TELEMETRY: ' + err;
  });
