// COSMIC OBSERVER — retro-futuristic 3D dashboard
// Consumes static/snapshot.json (built by scripts/generate_dashboard_snapshot.py)
// and renders a holographic globe with satellite tracking overlays.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { feature as topoFeature } from 'topojson-client';

// Real 110m-resolution world coastlines (Natural Earth data via the
// world-atlas package), decoded client-side and drawn using this app's
// OWN lat/lon -> xyz placement function (latLonAltToVector3) — the exact
// same function used for every satellite and launch-site marker. This
// guarantees the continents can never be mismatched/mirrored relative to
// everything else in the scene, since there is no separate texture/UV
// mapping involved at all, only shared coordinate math.
const LAND_TOPOLOGY_URL = 'https://cdn.jsdelivr.net/npm/world-atlas@2/land-110m.json';

const R_EARTH_KM = 6378.137;
const GLOBE_RADIUS = 4.0; // scene units representing R_EARTH_KM

const PALETTE = {
  cyan: 0x00ffff,
  magenta: 0xff0066,
  amber: 0xffcc00,
  phosphorGreen: 0x00ff66,
  violet: 0xcc66ff,
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

// Procedural cratered-rock look for non-Earth bodies (Mercury first) —
// no photographic texture (avoids the exact UV/orientation pitfalls this
// project hit with Earth's original texture, see dashboard.js history),
// just a muted, realistic base color modulated by fractal value noise
// sampled in OBJECT space (vPosition), so the crater pattern rotates
// rigidly with the sphere rather than swimming with the camera.
const ROCKY_BODY_VERTEX_SHADER = `
varying vec3 vPosition;
varying vec3 vNormalWorld;
void main() {
  vPosition = position;
  vNormalWorld = normalize(mat3(modelMatrix) * normal);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const ROCKY_BODY_FRAGMENT_SHADER = `
uniform vec3 uBaseColor;
uniform vec3 uLightDir;
varying vec3 vPosition;
varying vec3 vNormalWorld;

float hash(vec3 p) {
  p = fract(p * vec3(443.897, 441.423, 437.195));
  p += dot(p, p.yzx + 19.19);
  return fract((p.x + p.y) * p.z);
}

float valueNoise(vec3 p) {
  vec3 i = floor(p);
  vec3 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(mix(hash(i + vec3(0.0,0.0,0.0)), hash(i + vec3(1.0,0.0,0.0)), f.x),
        mix(hash(i + vec3(0.0,1.0,0.0)), hash(i + vec3(1.0,1.0,0.0)), f.x), f.y),
    mix(mix(hash(i + vec3(0.0,0.0,1.0)), hash(i + vec3(1.0,0.0,1.0)), f.x),
        mix(hash(i + vec3(0.0,1.0,1.0)), hash(i + vec3(1.0,1.0,1.0)), f.x), f.y),
    f.z);
}

float fbm(vec3 p) {
  float v = 0.0;
  float amp = 0.5;
  for (int i = 0; i < 5; i++) {
    v += amp * valueNoise(p);
    p *= 2.02;
    amp *= 0.5;
  }
  return v;
}

void main() {
  float n = fbm(vPosition * 2.4);
  float craterMask = smoothstep(0.35, 0.68, n);
  vec3 darkColor = uBaseColor * 0.7;
  vec3 lightColor = uBaseColor * 1.35;
  vec3 albedo = mix(darkColor, lightColor, craterMask);

  // Real terminator: uLightDir is the body's own true current direction
  // toward the Sun (data.solar_system.BodyPosition.sun_direction), not
  // an arbitrary fixed vector — the lit/unlit split you see actually
  // matches where the Sun is right now, from that body.
  float lambert = max(dot(normalize(vNormalWorld), normalize(uLightDir)), 0.0);
  // Ambient floor kept fairly high (0.42) so the night side stays
  // readable in a dashboard rather than going physically-accurate pitch
  // black — a deliberate visibility choice, not a claim of literal
  // night-side brightness.
  float ambient = 0.42;
  vec3 shaded = albedo * (ambient + (1.0 - ambient) * lambert) * 1.25;

  gl_FragColor = vec4(min(shaded, vec3(1.0)), 1.0);
}
`;

// Unlit, warm, additively-glowing Sun — same rim-glow technique as
// ATMOSPHERE_*_SHADER (a real light source has no "far side" to shade).
const SUN_FRAGMENT_SHADER = `
uniform vec3 uGlowColor;
varying vec3 vNormal;
void main() {
  float rim = 1.0 - max(dot(normalize(vNormal), vec3(0.0, 0.0, 1.0)), 0.0);
  vec3 core = uGlowColor * 1.4;
  gl_FragColor = vec4(mix(core, uGlowColor, rim), 1.0);
}
`;

class Dashboard {
  constructor(snapshot) {
    this.snapshot = snapshot;
    this.frameIndex = 0;
    this.playing = false;

    this._initScene();
    this._buildEarth();
    this._initSun();
    this._initMoons();
    this._buildSatellites();
    this._buildGroundTracks();
    this._initHud();
    this._initPicking();
    this._buildLaunchSites();
    this._initMissionPlanner();
    this._initCelestialBodies();
    this._initSmallBodies();

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

    // Remembered so "return to Earth" can restore the exact original vantage.
    this.homeCamera = {
      position: this.camera.position.clone(),
      target: new THREE.Vector3(0, 0, 0),
      minDistance: this.controls.minDistance,
      maxDistance: this.controls.maxDistance,
    };

    window.addEventListener('resize', () => {
      this.camera.aspect = window.innerWidth / window.innerHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(window.innerWidth, window.innerHeight);
    });
  }

  _buildEarth() {
    // Plain dark sphere — no day/night, no texture/UV mapping at all
    // (removed per explicit request: no sun simulation, and a texture's
    // own UV convention is one more thing that can silently disagree
    // with this app's coordinate math). Continents are drawn separately
    // in _buildContinentOutlines() as real coastline geometry placed
    // with this app's own lat/lon -> xyz function — the same one used
    // for every satellite and launch-site marker — so there is no way
    // for the map to disagree with where anything else is plotted.
    const geometry = new THREE.SphereGeometry(GLOBE_RADIUS, 64, 64);
    const material = new THREE.MeshBasicMaterial({ color: 0x03060c });
    this.earth = new THREE.Mesh(geometry, material);
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

    this._buildContinentOutlines();
  }

  _buildContinentOutlines() {
    fetch(LAND_TOPOLOGY_URL)
      .then((r) => r.json())
      .then((topology) => {
        const geo = topoFeature(topology, topology.objects.land);
        const polygons = [];
        for (const f of geo.features) {
          const geomType = f.geometry.type;
          const polys = geomType === 'Polygon' ? [f.geometry.coordinates] : f.geometry.coordinates;
          for (const poly of polys) polygons.push(...poly); // each poly = array of rings; flatten to rings
        }

        const material = new THREE.LineBasicMaterial({ color: PALETTE.cyan, transparent: true, opacity: 0.85 });
        const outlineRadius = 5; // km above the surface, avoids z-fighting with the base sphere

        for (const ring of polygons) {
          // Split at antimeridian crossings so a ring spanning +/-180 deg
          // longitude draws as separate arcs instead of one spurious
          // line slicing straight across the globe's interior.
          let segment = [];
          let prevLon = null;
          const segments = [segment];

          for (const [lon, lat] of ring) {
            if (prevLon !== null && Math.abs(lon - prevLon) > 180) {
              segment = [];
              segments.push(segment);
            }
            segment.push(latLonAltToVector3(lat, lon, outlineRadius));
            prevLon = lon;
          }

          for (const seg of segments) {
            if (seg.length < 2) continue;
            const geometry = new THREE.BufferGeometry().setFromPoints(seg);
            this.scene.add(new THREE.Line(geometry, material));
          }
        }
      })
      .catch((err) => console.error('Failed to load continent outlines:', err));
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

    // Moons are individual Mesh spheres (not a batched Points cloud like
    // satellites/small bodies), so they need their own intersect pass.
    const moonMeshes = (this.moonIndex || []).map((m) => m.mesh);
    if (moonMeshes.length > 0) {
      const moonHits = this.raycaster.intersectObjects(moonMeshes);
      if (moonHits.length > 0) {
        const entry = this.moonIndex.find((m) => m.mesh === moonHits[0].object);
        if (entry) {
          this._showMoonInfo(entry);
          return;
        }
      }
    }

    const pointObjects = Object.values(this.groups).map((g) => g.points);
    if (this.issPoint) pointObjects.push(this.issPoint);
    if (this.smallBodyPoints) pointObjects.push(this.smallBodyPoints);

    const intersects = this.raycaster.intersectObjects(pointObjects);
    if (intersects.length === 0) return;

    const hit = intersects[0];
    if (this.smallBodyPoints && hit.object === this.smallBodyPoints) {
      this._showSmallBodyInfo(this.smallBodyIndex[hit.index]);
      return;
    }

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

    document.getElementById('stat-frame').textContent = `${frameIndex + 1} / ${this.snapshot.frame_times_iso.length}`;
    document.getElementById('stat-epoch').textContent = this.snapshot.frame_times_iso[frameIndex].replace('T', ' ').slice(0, 19);
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

  // -------------------------------------------------------------------
  // Celestial body travel — step 1: Mercury.
  //
  // Selecting a body fetches its REAL current Earth-relative direction
  // from JPL Horizons (data.solar_system, via /api/solar-system) and
  // flies the camera out along that true direction. The body's PHYSICAL
  // SIZE is rendered to the same real scale Earth already uses
  // (GLOBE_RADIUS units per R_EARTH_KM); the DISPLAY DISTANCE it's
  // placed at is not to that same scale (real interplanetary distances
  // would put it far outside any navigable scene) — see
  // _displayDistanceUnits(). Passing bodies/moons/comets en route are a
  // later step, not implemented yet.
  // -------------------------------------------------------------------

  _initSun() {
    // The Sun itself, placed along its real current direction from
    // Earth (same data.solar_system mechanism as any other body) — this
    // is also what supplies uLightDir for every rocky body's shader
    // (see _getOrBuildBody), so Mercury's lit side is genuinely the side
    // actually facing the Sun right now, not an arbitrary fixed angle.
    fetch('/api/solar-system/sun/position')
      .then((r) => r.json())
      .then((position) => this._buildSun(position))
      .catch((err) => console.error('Failed to load Sun position:', err));
  }

  _buildSun(position) {
    // The Sun's TRUE size (radius_km=696,000, ~109x Earth's) would dwarf
    // this entire scene at the same scale factor everything else uses —
    // its displayed radius here is a fixed, compressed dramatization
    // (like body travel distances, see _displayDistanceUnits), not a
    // to-scale rendering. Its DIRECTION from Earth is real.
    const displayRadius = 2.6;
    const displayDistance = this._displayDistanceUnits(position.distance_km) + 10;

    const direction = new THREE.Vector3(...position.direction).normalize();
    const sunPosition = direction.clone().multiplyScalar(displayDistance);

    // The Sun's real direction from Earth varies (it's genuinely
    // wherever it currently is along Earth's orbit) and could easily
    // fall outside the original fixed home camera framing — recompose
    // the home view once, on load, so Earth AND the newly-placed Sun
    // are both actually visible together, instead of leaving the Sun
    // to be found only by manually rotating.
    const framingBack = direction.clone().multiplyScalar(-14);
    const homePosition = framingBack.add(new THREE.Vector3(0, 6, 0));
    this.camera.position.copy(homePosition);
    this.camera.lookAt(0, 0, 0);
    this.controls.update();
    this.homeCamera.position.copy(homePosition);

    const geometry = new THREE.SphereGeometry(displayRadius, 32, 32);
    const material = new THREE.MeshBasicMaterial({ color: 0xfff2c2 });
    this.sunMesh = new THREE.Mesh(geometry, material);
    this.sunMesh.position.copy(sunPosition);
    this.scene.add(this.sunMesh);

    const glowGeometry = new THREE.SphereGeometry(displayRadius * 1.3, 32, 32);
    const glowMaterial = new THREE.ShaderMaterial({
      uniforms: { uGlowColor: { value: new THREE.Color(0xffcc66) } },
      vertexShader: ATMOSPHERE_VERTEX_SHADER,
      fragmentShader: SUN_FRAGMENT_SHADER,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
      transparent: true,
    });
    this.scene.add(new THREE.Mesh(glowGeometry, glowMaterial));
  }

  _initCelestialBodies() {
    this.bodies = {};      // bodyKey -> { mesh, radiusUnits, minDistance, maxDistance, cameraPosition }
    this.bodyMeta = {};    // bodyKey -> { display_name, radius_km, color_hex }

    const select = document.getElementById('body-select');
    fetch('/api/solar-system/bodies')
      .then((r) => r.json())
      .then((bodies) => {
        this.bodyMeta = bodies;
        for (const [key, meta] of Object.entries(bodies)) {
          const opt = document.createElement('option');
          opt.value = key;
          opt.textContent = meta.display_name.toUpperCase();
          select.appendChild(opt);
        }
      })
      .catch((err) => console.error('Failed to load celestial body list:', err));

    select.addEventListener('change', () => this._travelToBody(select.value));
  }

  _displayDistanceUnits(distanceKm) {
    // Compressed (not-to-scale) placement distance so real interplanetary
    // ranges stay navigable in this scene: log-scaled, clamped to a band
    // just beyond Earth's own orbit-camera range.
    const raw = 10 + 4 * Math.log10(distanceKm / 1e6);
    return THREE.MathUtils.clamp(raw, 10, 26);
  }

  _getOrBuildBody(bodyKey, meta, position) {
    if (this.bodies[bodyKey]) return this.bodies[bodyKey];

    const radiusUnits = GLOBE_RADIUS * (meta.radius_km / R_EARTH_KM);
    const displayDistance = this._displayDistanceUnits(position.distance_km);

    const direction = new THREE.Vector3(...position.direction).normalize();
    const bodyPosition = direction.clone().multiplyScalar(displayDistance);

    // Real direction from THIS body toward the Sun right now (see
    // data.solar_system.BodyPosition.sun_direction) — the shader's
    // terminator is genuinely where the Sun currently falls on it.
    const lightDir = new THREE.Vector3(...position.sun_direction).normalize();

    const geometry = new THREE.SphereGeometry(radiusUnits, 64, 64);
    const material = new THREE.ShaderMaterial({
      uniforms: {
        uBaseColor: { value: new THREE.Color(meta.color_hex) },
        uLightDir: { value: lightDir },
      },
      vertexShader: ROCKY_BODY_VERTEX_SHADER,
      fragmentShader: ROCKY_BODY_FRAGMENT_SHADER,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.copy(bodyPosition);
    this.scene.add(mesh);

    // A faint line from Earth toward the body along the real direction,
    // so the "real direction" claim reads visually, not just as a number.
    const lineGeometry = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), bodyPosition]);
    const lineMaterial = new THREE.LineBasicMaterial({ color: PALETTE.amber, transparent: true, opacity: 0.25 });
    this.scene.add(new THREE.Line(lineGeometry, lineMaterial));

    const cameraOffset = direction.clone().multiplyScalar(-radiusUnits * 3.5);
    cameraOffset.y += radiusUnits * 0.8;

    const body = {
      mesh,
      radiusUnits,
      minDistance: radiusUnits * 1.5,
      maxDistance: radiusUnits * 10,
      cameraPosition: bodyPosition.clone().add(cameraOffset),
    };
    this.bodies[bodyKey] = body;
    this._attachMoonsForParent(bodyKey, bodyPosition);
    return body;
  }

  // -------------------------------------------------------------------
  // Real moons (Mars: Phobos/Deimos; Earth: the Moon — Mercury/Venus
  // genuinely have none, so they get none). Each moon is positioned
  // relative to its OWN parent planet's real current position (see
  // data.solar_system.BodyPosition.relative_to), not to Earth.
  // -------------------------------------------------------------------

  _initMoons() {
    this.moonMeta = {};       // key -> meta
    this.moonsByParent = {};  // parentKey -> [key, ...]
    this.moonIndex = [];      // for picking: {key, meta, mesh, position}

    fetch('/api/solar-system/moons')
      .then((r) => r.json())
      .then((moons) => {
        this.moonMeta = moons;
        for (const [key, meta] of Object.entries(moons)) {
          (this.moonsByParent[meta.parent] = this.moonsByParent[meta.parent] || []).push(key);
        }
        // Earth is already in the scene at the origin from startup, so
        // its moon can be attached immediately rather than waiting for
        // a travel selection (Mars's moons attach lazily in
        // _getOrBuildBody, once Mars itself is actually built).
        this._attachMoonsForParent('earth', new THREE.Vector3(0, 0, 0));
      })
      .catch((err) => console.error('Failed to load moon list:', err));
  }

  _moonDisplayDistanceUnits(distanceKm, parentRadiusUnits) {
    // A moon's TRUE orbital radius (e.g. Phobos ~9376 km from Mars)
    // happens to already render reasonably at this scene's real-size
    // scale factor for Mars, but for tiny/close cases the log-compressed
    // formula (consistent with _displayDistanceUnits' approach) keeps
    // it comfortably clear of the parent's own surface either way.
    const raw = parentRadiusUnits + 2.5 + 3 * Math.log10(distanceKm / 1000);
    return Math.max(parentRadiusUnits + 1.2, raw);
  }

  async _attachMoonsForParent(parentKey, parentPosition) {
    const keys = this.moonsByParent && this.moonsByParent[parentKey];
    if (!keys) return;

    for (const key of keys) {
      if (this.moonIndex.some((m) => m.key === key)) continue; // already attached
      const meta = this.moonMeta[key];

      let position;
      try {
        const resp = await fetch(`/api/solar-system/moons/${key}/position`);
        if (!resp.ok) continue;
        position = await resp.json();
      } catch (err) {
        console.error(`Failed to load position for moon '${key}':`, err);
        continue;
      }

      const parentRadiusUnits = parentKey === 'earth' ? GLOBE_RADIUS : this.bodies[parentKey].radiusUnits;
      // A floor on displayed radius: Phobos/Deimos's TRUE scale (a few
      // km) would render as a sub-pixel dot at this scene's scale factor
      // — bumped up to stay visibly a sphere, same spirit as the Sun's
      // fixed compressed display radius.
      const radiusUnits = Math.max(0.08, GLOBE_RADIUS * (meta.radius_km / R_EARTH_KM));
      const displayDistance = this._moonDisplayDistanceUnits(position.distance_km, parentRadiusUnits);

      const direction = new THREE.Vector3(...position.direction).normalize();
      const meshPosition = parentPosition.clone().add(direction.multiplyScalar(displayDistance));

      const lightDir = new THREE.Vector3(...position.sun_direction).normalize();
      const geometry = new THREE.SphereGeometry(radiusUnits, 32, 32);
      const material = new THREE.ShaderMaterial({
        uniforms: {
          uBaseColor: { value: new THREE.Color(meta.color_hex) },
          uLightDir: { value: lightDir },
        },
        vertexShader: ROCKY_BODY_VERTEX_SHADER,
        fragmentShader: ROCKY_BODY_FRAGMENT_SHADER,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.copy(meshPosition);
      this.scene.add(mesh);

      this.moonIndex.push({ key, meta, mesh, position });
    }
  }

  _showMoonInfo(entry) {
    const { meta, position } = entry;
    document.getElementById('info-panel').innerHTML = `
      <table>
        <tr><td>NAME</td><td>${meta.display_name}</td></tr>
        <tr><td>ORBITS</td><td class="value">${meta.parent.toUpperCase()}</td></tr>
        <tr><td>DIST (${meta.parent.toUpperCase()})</td><td class="value">${position.distance_km.toFixed(0)} km</td></tr>
        <tr><td>RADIUS</td><td class="value">${meta.radius_km} km</td></tr>
      </table>
      <div class="dim" style="margin-top:4px;">real ephemeris (JPL Horizons), current position</div>`;
  }

  _animateCamera(toPosition, toTarget, durationMs, onComplete) {
    const fromPosition = this.camera.position.clone();
    const fromTarget = this.controls.target.clone();
    const startTime = performance.now();
    const wasAutoRotate = this.controls.autoRotate;
    this.controls.enabled = false;
    this.controls.autoRotate = false;

    const step = (now) => {
      const t = Math.min(1, (now - startTime) / durationMs);
      const eased = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; // easeInOutQuad
      this.camera.position.lerpVectors(fromPosition, toPosition, eased);
      this.controls.target.lerpVectors(fromTarget, toTarget, eased);
      this.controls.update();

      if (t < 1) {
        requestAnimationFrame(step);
      } else {
        this.controls.enabled = true;
        this.controls.autoRotate = wasAutoRotate;
        if (onComplete) onComplete();
      }
    };
    requestAnimationFrame(step);
  }

  async _travelToBody(bodyKey) {
    const statusEl = document.getElementById('celestial-status');

    if (bodyKey === 'earth') {
      statusEl.textContent = 'RETURNING TO EARTH…';
      this._animateCamera(this.homeCamera.position, this.homeCamera.target, 2200, () => {
        this.controls.minDistance = this.homeCamera.minDistance;
        this.controls.maxDistance = this.homeCamera.maxDistance;
        statusEl.textContent = 'EARTH — HOME VIEW';
      });
      return;
    }

    const meta = this.bodyMeta[bodyKey];
    if (!meta) return;
    statusEl.textContent = `CALCULATING TRAJECTORY TO ${meta.display_name.toUpperCase()}…`;

    let position;
    try {
      const resp = await fetch(`/api/solar-system/${bodyKey}/position`);
      if (!resp.ok) throw new Error(await extractErrorDetail(resp));
      position = await resp.json();
    } catch (err) {
      statusEl.textContent = `HATA: ${err.message || err}`;
      return;
    }

    const body = this._getOrBuildBody(bodyKey, meta, position);
    const distanceAu = (position.distance_km / 1.496e8).toFixed(3);
    statusEl.textContent = `${meta.display_name.toUpperCase()} — ${distanceAu} AU FROM EARTH (REAL-TIME DIRECTION)`;

    this._animateCamera(body.cameraPosition, body.mesh.position, 2800, () => {
      this.controls.minDistance = body.minDistance;
      this.controls.maxDistance = body.maxDistance;
    });
  }

  // -------------------------------------------------------------------
  // Real small bodies (asteroids/comets) — a curated set of genuinely
  // tracked real objects (see data.small_bodies), plotted at their
  // current real direction/distance from Earth (SBDB orbital elements +
  // two-body Kepler propagation server-side). Positions are current
  // real-time snapshots, not tied to the time-slider/frame animation the
  // TLE satellites use.
  // -------------------------------------------------------------------

  _initSmallBodies() {
    this.smallBodyIndex = []; // parallel array: index i -> {key, state}

    fetch('/api/small-bodies')
      .then((r) => r.json())
      .then((bodies) => Promise.all(
        Object.entries(bodies).map(([key, meta]) =>
          fetch(`/api/small-bodies/${key}/position`)
            .then((r) => (r.ok ? r.json() : null))
            .then((state) => ({ key, meta, state }))
            .catch(() => null)
        )
      ))
      .then((results) => this._buildSmallBodyMarkers(results.filter((r) => r && r.state)))
      .catch((err) => console.error('Failed to load small bodies:', err));
  }

  _buildSmallBodyMarkers(entries) {
    if (entries.length === 0) return;

    const positions = new Float32Array(entries.length * 3);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const material = new THREE.PointsMaterial({
      size: 0.28,
      map: makeCircleSprite(PALETTE.violet),
      transparent: true,
      depthWrite: false,
      sizeAttenuation: true,
    });

    entries.forEach(({ key, meta, state }, i) => {
      const direction = new THREE.Vector3(...state.direction).normalize();
      const displayDistance = this._displayDistanceUnits(state.distance_km);
      const p = direction.multiplyScalar(displayDistance);
      positions[i * 3] = p.x; positions[i * 3 + 1] = p.y; positions[i * 3 + 2] = p.z;
      this.smallBodyIndex.push({ key, meta, state });
    });
    geometry.attributes.position.needsUpdate = true;

    this.smallBodyPoints = new THREE.Points(geometry, material);
    this.scene.add(this.smallBodyPoints);
  }

  _showSmallBodyInfo(entry) {
    const { meta, state } = entry;
    const distanceAu = (state.distance_km / 1.496e8).toFixed(3);
    document.getElementById('info-panel').innerHTML = `
      <table>
        <tr><td>NAME</td><td>${meta.display_name}</td></tr>
        <tr><td>TYPE</td><td class="value">${state.orbit_class_name.toUpperCase()}</td></tr>
        <tr><td>DIST (EARTH)</td><td class="value">${distanceAu} AU</td></tr>
        <tr><td>HAZARDOUS</td><td class="value">${state.is_potentially_hazardous ? 'YES' : 'NO'}</td></tr>
      </table>
      <div class="dim" style="margin-top:4px;">real orbital elements (JPL SBDB), current position</div>`;
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
