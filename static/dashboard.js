// COSMIC OBSERVER — retro-futuristic 3D dashboard
// Consumes static/snapshot.json (built by scripts/generate_dashboard_snapshot.py)
// and renders a holographic globe with satellite tracking overlays.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
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
const MAX_BODY_RADIUS_UNITS = 7.5; // see _getOrBuildBody's Jupiter-sizing note

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

// Tactical-hologram look for non-Earth bodies (Mercury/Venus/Mars,
// moons): no photographic texture (this project deliberately never
// samples a diffuse image — see dashboard.js history around Earth's
// original texture UV problems) — instead a dark radar-globe base with
// faint object-space "topographic" scanlines plus a Fresnel rim glow in
// the interface's own cyan, so every body reads as a scanned HUD
// projection rather than a photoreal render. uLightDir (the body's real,
// per-body current direction toward the Sun — see
// data.solar_system.BodyPosition.sun_direction) is kept as a subtle
// secondary brightness cue so the underlying "real astronomy" data is
// still legible, just subordinate to the tactical aesthetic.
const ROCKY_BODY_VERTEX_SHADER = `
varying vec3 vPosition;
varying vec3 vNormalWorld;
varying vec3 vWorldPosition;
void main() {
  vPosition = position;
  vNormalWorld = normalize(mat3(modelMatrix) * normal);
  vec4 worldPos = modelMatrix * vec4(position, 1.0);
  vWorldPosition = worldPos.xyz;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const ROCKY_BODY_FRAGMENT_SHADER = `
uniform vec3 uBaseColor;
uniform vec3 uRimColor;
uniform vec3 uLightDir;
varying vec3 vPosition;
varying vec3 vNormalWorld;
varying vec3 vWorldPosition;

void main() {
  vec3 normal = normalize(vNormalWorld);
  vec3 viewDir = normalize(cameraPosition - vWorldPosition);

  // Fresnel rim: near-grazing angles (silhouette edge) glow brightest —
  // the literal "kenarlari parlasin" ask. Higher exponent keeps this a
  // thin bright edge rather than a wide halo eating the whole disk once
  // the bloom pass spreads it further.
  float fresnel = pow(1.0 - max(dot(normal, viewDir), 0.0), 3.4);

  // Topographic scanlines: horizontal contour bands in object space
  // (latitude-like), read as a radar/wireframe survey rather than a
  // literal surface photo.
  float lat = normalize(vPosition).y;
  float bands = sin(lat * 55.0) * 0.5 + 0.5;
  float scan = smoothstep(0.85, 1.0, bands);

  // Real per-body Sun direction, kept as a gentle secondary shading cue
  // (not the dominant read anymore — the tactical rim/scanline look is).
  float lambert = max(dot(normal, normalize(uLightDir)), 0.0);
  float ambient = 0.45;

  vec3 darkBase = vec3(0.015, 0.03, 0.045);
  vec3 topoColor = uBaseColor * 0.55;
  vec3 base = mix(darkBase, topoColor, scan * 0.4) * (ambient + (1.0 - ambient) * lambert);

  vec3 rimGlow = uRimColor * fresnel * 1.3;

  gl_FragColor = vec4(base + rimGlow, 1.0);
}
`;

// Sun: a pulsing procedural "energy source" core (not a flat lit ball —
// a real light source has no far side to shade) plus the same
// additive Fresnel corona shell technique as ATMOSPHERE_*_SHADER,
// tuned warm so it reads as the system's power source, not a
// photographic star.
const SUN_FRAGMENT_SHADER = `
uniform vec3 uGlowColor;
uniform float uTime;
varying vec3 vNormal;
void main() {
  float rim = 1.0 - max(dot(normalize(vNormal), vec3(0.0, 0.0, 1.0)), 0.0);
  vec3 core = uGlowColor * 1.4;
  gl_FragColor = vec4(mix(core, uGlowColor, rim), 1.0);
}
`;

const SUN_CORE_VERTEX_SHADER = `
varying vec3 vNormal;
void main() {
  vNormal = normalize(normalMatrix * normal);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const SUN_CORE_FRAGMENT_SHADER = `
uniform vec3 uColor;
uniform float uTime;
varying vec3 vNormal;
void main() {
  float pulse = 0.88 + 0.12 * sin(uTime * 1.7);
  float rim = 1.0 - max(dot(normalize(vNormal), vec3(0.0, 0.0, 1.0)), 0.0);
  vec3 core = uColor * (1.25 + rim * 0.5) * pulse;
  gl_FragColor = vec4(core, 1.0);
}
`;

// Real ring systems (Saturn's — bright, wide, iconic; Uranus's — real
// but far fainter and narrower, discovered only in 1977) — a flat
// radial band, scanned/tactical rather than photographic: concentric
// contour bands plus Saturn's genuine, famous Cassini Division gap. See
// _addPlanetRings for how uInnerRadius/uOuterRadius are set to keep
// everything positioned correctly regardless of the compressed display
// radius each planet actually renders at.
const RING_VERTEX_SHADER = `
varying float vRadius;
void main() {
  vRadius = length(position.xy);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const RING_FRAGMENT_SHADER = `
uniform vec3 uBaseColor;
uniform vec3 uRimColor;
uniform float uInnerRadius;
uniform float uOuterRadius;
uniform float uMaxAlpha;
varying float vRadius;

void main() {
  float t = clamp((vRadius - uInnerRadius) / (uOuterRadius - uInnerRadius), 0.0, 1.0);

  // A dark gap at ~2/3 of the way out — calibrated to land on Saturn's
  // real, famous Cassini Division; for other ringed planets (Uranus)
  // this reads as a generic band-gap rather than one specific named
  // real gap, since precisely placing every one of Uranus's several
  // narrow real rings is out of scope for this pass.
  float gap = smoothstep(0.60, 0.63, t) - smoothstep(0.66, 0.69, t);
  float bands = sin(t * 46.0) * 0.5 + 0.5;

  vec3 color = mix(uBaseColor * 0.55, uBaseColor * 1.15, bands);
  color = mix(color, uRimColor * 0.4, gap);

  float edgeFade = smoothstep(0.0, 0.05, t) * (1.0 - smoothstep(0.95, 1.0, t));
  float alpha = uMaxAlpha * (1.0 - gap * 0.8) * edgeFade;

  gl_FragColor = vec4(color, alpha);
}
`;

class Dashboard {
  constructor(snapshot) {
    this.snapshot = snapshot;
    this.frameIndex = 0;
    this.playing = false;

    this._initScene();
    this._initLabels();
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
    this.controls.maxDistance = 70; // wide enough to manually zoom out to any pre-built planet (see _displayDistanceUnits)
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

    // Very low-level ambient fill so no shaded body's dark side goes to
    // literal black (the "hacimlerini belli edecek hafif ortam ışığı"
    // ask) — the actual visible ambient floor lives in
    // ROCKY_BODY_FRAGMENT_SHADER's own `ambient` term (a THREE.Light
    // doesn't affect a custom ShaderMaterial on its own), this scene
    // light is here for completeness/any future standard-material mesh.
    this.scene.add(new THREE.AmbientLight(0x0a1a2a, 1.0));

    this._buildStarfield();
    this._buildReferenceGrid();
    this._initPostProcessing();

    window.addEventListener('resize', () => {
      this.camera.aspect = window.innerWidth / window.innerHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(window.innerWidth, window.innerHeight);
      this.composer.setSize(window.innerWidth, window.innerHeight);
      this.bloomPass.setSize(window.innerWidth, window.innerHeight);
    });
  }

  _initPostProcessing() {
    // A restrained "CRT glow" bloom — cyan rim-light/wireframe/HUD text
    // reads like it's glowing on an old tactical display, without
    // blowing out the whole scene. Threshold kept low so even
    // moderately bright cyan lines catch it (their fresnel rim term
    // deliberately pushes color values past 1.0 for exactly this).
    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloomPass = new UnrealBloomPass(
      new THREE.Vector2(window.innerWidth, window.innerHeight),
      0.32,  // strength
      0.28,  // radius
      0.4,   // threshold
    );
    this.composer.addPass(this.bloomPass);
    this.composer.addPass(new OutputPass());
  }

  _buildStarfield() {
    // Subtle, UI-colored star field for depth — not a literal black
    // void behind the scene. Positioned on a large shell well outside
    // any interactive content (planets/asteroids top out well under 30
    // scene units — see _displayDistanceUnits).
    const starCount = 2200;
    const positions = new Float32Array(starCount * 3);
    for (let i = 0; i < starCount; i++) {
      const r = 220 + Math.random() * 180;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = r * Math.cos(phi);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const material = new THREE.PointsMaterial({
      size: 0.9,
      map: makeCircleSprite(0x88bbdd),
      color: 0x88bbdd,
      transparent: true,
      opacity: 0.5,
      depthWrite: false,
      sizeAttenuation: true,
    });
    this.scene.add(new THREE.Points(geometry, material));
    this._buildDeepStarfield();
  }

  _buildDeepStarfield() {
    // A second, much farther and sparser star layer — extra depth/
    // parallax richness beyond the reference-grid shell (radius 55) and
    // the primary starfield (radius 220-400), without altering the
    // restrained interface palette: same cool cyan-blue family, just a
    // few individually dimmer/smaller points, plus an occasional faint
    // warm-white one for subtle variation (real starfields aren't
    // perfectly monochrome) — never bright or saturated enough to read
    // as decoration competing with the tactical HUD elements.
    const starCount = 1400;
    const positions = new Float32Array(starCount * 3);
    const colors = new Float32Array(starCount * 3);
    const coolColor = new THREE.Color(0x6fa0c9);
    const warmColor = new THREE.Color(0xd9d0b8);
    for (let i = 0; i < starCount; i++) {
      const r = 500 + Math.random() * 400;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = r * Math.cos(phi);

      const c = Math.random() < 0.12 ? warmColor : coolColor;
      colors[i * 3] = c.r; colors[i * 3 + 1] = c.g; colors[i * 3 + 2] = c.b;
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    const material = new THREE.PointsMaterial({
      size: 0.5,
      map: makeCircleSprite(0xffffff),
      vertexColors: true,
      transparent: true,
      opacity: 0.32,
      depthWrite: false,
      sizeAttenuation: true,
    });
    this.scene.add(new THREE.Points(geometry, material));
  }

  _buildReferenceGrid() {
    // A large, very faint spherical wireframe — a "radar globe" backdrop
    // that reads as scientific instrumentation rather than empty space,
    // without competing with the actual tracked objects in front of it.
    const gridGeometry = new THREE.SphereGeometry(55, 24, 16);
    const gridMaterial = new THREE.LineBasicMaterial({
      color: PALETTE.cyan, transparent: true, opacity: 0.05,
    });
    this.scene.add(new THREE.LineSegments(new THREE.WireframeGeometry(gridGeometry), gridMaterial));
  }

  _addWireframeOverlay(mesh, radiusUnits) {
    // A separate, deliberately low-poly sphere (independent of the
    // shaded mesh's higher-detail geometry, which would otherwise
    // wireframe into an unreadable dense mesh) for a clean lat/lon HUD
    // wireframe — the literal "tel kafes" ask — parented to the body so
    // it inherits its position/rotation automatically.
    const wireGeometry = new THREE.SphereGeometry(radiusUnits * 1.015, 16, 12);
    const wireMaterial = new THREE.LineBasicMaterial({
      color: PALETTE.cyan, transparent: true, opacity: 0.22,
    });
    mesh.add(new THREE.LineSegments(new THREE.WireframeGeometry(wireGeometry), wireMaterial));
  }

  _addPlanetRings(mesh, radiusUnits, opts) {
    const { innerRatio, outerRatio, tiltDeg, colorHex, maxAlpha, wireOpacity } = opts;
    const innerRadius = radiusUnits * innerRatio;
    const outerRadius = radiusUnits * outerRatio;

    const geometry = new THREE.RingGeometry(innerRadius, outerRadius, 128, 1);
    const material = new THREE.ShaderMaterial({
      uniforms: {
        uBaseColor: { value: new THREE.Color(colorHex) },
        uRimColor: { value: new THREE.Color(PALETTE.cyan) },
        uInnerRadius: { value: innerRadius },
        uOuterRadius: { value: outerRadius },
        uMaxAlpha: { value: maxAlpha },
      },
      vertexShader: RING_VERTEX_SHADER,
      fragmentShader: RING_FRAGMENT_SHADER,
      transparent: true,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const ring = new THREE.Mesh(geometry, material);

    // RingGeometry lies flat in the local XY plane by default; rotate
    // it to lie roughly in the orbital plane, then apply the planet's
    // REAL axial tilt for its recognizable ring silhouette (Saturn
    // ~26.7 deg; Uranus a dramatic ~97.77 deg — it rotates almost on
    // its side, a genuine, striking real fact, not a stylization choice).
    // This is a STYLISTIC tilt around a fixed local axis using the
    // planet's real obliquity angle, not a precise real-time render of
    // its actual pole orientation as seen from Earth right now (that
    // needs additional ephemeris this pass doesn't fetch) — an honest
    // simplification, not a claim of exact current orientation.
    ring.rotation.x = Math.PI / 2 - THREE.MathUtils.degToRad(tiltDeg);
    mesh.add(ring);

    // A faint concentric wireframe echo, consistent with every other
    // body's tactical wireframe treatment.
    const wireGeometry = new THREE.RingGeometry(innerRadius, outerRadius, 64, 6);
    const wireMaterial = new THREE.LineBasicMaterial({ color: PALETTE.cyan, transparent: true, opacity: wireOpacity });
    const wireframe = new THREE.LineSegments(new THREE.WireframeGeometry(wireGeometry), wireMaterial);
    wireframe.rotation.copy(ring.rotation);
    mesh.add(wireframe);
  }

  // -------------------------------------------------------------------
  // Body labels — a DOM overlay (not 3D sprites/CSS2DRenderer, kept
  // deliberately simple) naming every real body/moon in the scene, so a
  // planet you haven't traveled to yet — but which was built up front by
  // _initCelestialBodies and is now just a small glowing dot in the
  // background from wherever you're currently looking — is still
  // identifiable, and so the currently-focused body is unambiguous. Only
  // "real sphere" bodies get a permanent label (planets/moons/Sun/Earth);
  // the small asteroid/comet markers stay click-to-info only, so the
  // scene doesn't get cluttered with a dozen permanent tags.
  // -------------------------------------------------------------------

  _initLabels() {
    this.labelLayer = document.getElementById('body-labels');
    this.labels = []; // { key, mesh, el }
    this.focusedBodyKey = 'earth';
  }

  _addLabel(key, mesh, text, colorHex) {
    const el = document.createElement('div');
    el.className = 'body-label';
    el.textContent = text;
    el.style.color = '#' + new THREE.Color(colorHex).getHexString();
    this.labelLayer.appendChild(el);
    this.labels.push({ key, mesh, el });
  }

  _updateLabels() {
    if (!this.labels) return;
    const v = new THREE.Vector3();
    for (const label of this.labels) {
      label.mesh.getWorldPosition(v);
      v.project(this.camera);

      const behindCamera = v.z > 1;
      if (behindCamera) {
        label.el.style.display = 'none';
        continue;
      }
      label.el.style.display = 'block';
      const x = (v.x * 0.5 + 0.5) * window.innerWidth;
      const y = (-v.y * 0.5 + 0.5) * window.innerHeight;
      label.el.style.left = `${x}px`;
      label.el.style.top = `${y}px`;
      label.el.classList.toggle('focused', label.key === this.focusedBodyKey);
    }
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
    this._addWireframeOverlay(this.earth, GLOBE_RADIUS);
    this._addLabel('earth', this.earth, 'EARTH', PALETTE.cyan);

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

    if (this.sunCoreMaterial) {
      this.sunCoreMaterial.uniforms.uTime.value = performance.now() / 1000;
    }

    this._updateLabels();
    this.composer.render();
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
    // A pulsing procedural core (not a flat lit ball) reads as a
    // tactical energy source rather than a photographic star; uTime is
    // advanced each frame in _animate().
    this.sunCoreMaterial = new THREE.ShaderMaterial({
      uniforms: {
        uColor: { value: new THREE.Color(0xfff2c2) },
        uTime: { value: 0 },
      },
      vertexShader: SUN_CORE_VERTEX_SHADER,
      fragmentShader: SUN_CORE_FRAGMENT_SHADER,
    });
    this.sunMesh = new THREE.Mesh(geometry, this.sunCoreMaterial);
    this.sunMesh.position.copy(sunPosition);
    this.scene.add(this.sunMesh);
    this._addLabel('sun', this.sunMesh, 'SUN', 0xfff2c2);

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

    // A procedural, no-external-texture stand-in for a lens-flare halo
    // (this project deliberately avoids photographic/binary texture
    // assets — see ROCKY_BODY_*_SHADER's own docstring): two soft
    // additive billboards at the Sun's own position, different sizes/
    // opacities, which combined with the bloom pass reads as a warm
    // optical glint rather than a flat sprite. Not a true screen-space
    // multi-ghost flare (that needs per-frame projection math) — a
    // reasonable, honest simplification for this pass.
    const haloSprites = [
      { scale: displayRadius * 5.5, opacity: 0.22 },
      { scale: displayRadius * 9.0, opacity: 0.1 },
    ];
    this.sunHaloSprites = haloSprites.map(({ scale, opacity }) => {
      const spriteMaterial = new THREE.SpriteMaterial({
        map: makeCircleSprite(0xffe6a8),
        transparent: true,
        opacity,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      });
      const sprite = new THREE.Sprite(spriteMaterial);
      sprite.scale.set(scale, scale, 1);
      sprite.position.copy(sunPosition);
      this.scene.add(sprite);
      return sprite;
    });
  }

  _initCelestialBodies() {
    this.bodies = {};      // bodyKey -> { mesh, radiusUnits, minDistance, maxDistance, cameraPosition }
    this.bodyMeta = {};    // bodyKey -> { display_name, radius_km, color_hex }

    const select = document.getElementById('body-select');
    fetch('/api/solar-system/bodies')
      .then((r) => r.json())
      .then(async (allBodies) => {
        // The Sun is handled entirely separately (_initSun/_buildSun —
        // its own pulsing-core shader, halo sprites, and the initial
        // home-camera framing) even though data.solar_system.BODIES
        // includes it alongside the planets for API uniformity. Building
        // it again here through the generic rocky-planet path would
        // create a second, different-looking, differently-positioned
        // "Sun" object and a redundant/nonsensical dropdown entry.
        const { sun: _sun, ...bodies } = allBodies;
        this.bodyMeta = bodies;
        for (const [key, meta] of Object.entries(bodies)) {
          const opt = document.createElement('option');
          opt.value = key;
          opt.textContent = meta.display_name.toUpperCase();
          select.appendChild(opt);
        }

        // Build every real planet up front (not only the one currently
        // traveled to) — otherwise a previously-unvisited planet simply
        // doesn't exist in the scene, which is exactly what made
        // distant bodies unidentifiable/impossible to label: there was
        // nothing there to label. Positions are fetched once and cached
        // resiliently server-side either way (core.resilient_fetch), so
        // this costs a handful of parallel requests at load, not per visit.
        await Promise.all(Object.entries(bodies).map(async ([key, meta]) => {
          try {
            const resp = await fetch(`/api/solar-system/${key}/position`);
            if (!resp.ok) return;
            const position = await resp.json();
            this._getOrBuildBody(key, meta, position);
          } catch (err) {
            console.error(`Failed to pre-build body '${key}':`, err);
          }
        }));
      })
      .catch((err) => console.error('Failed to load celestial body list:', err));

    select.addEventListener('change', () => this._travelToBody(select.value));
  }

  _displayDistanceUnits(distanceKm, radiusUnits = 0) {
    // Compressed (not-to-scale) placement distance so real interplanetary
    // ranges stay navigable in this scene: log-scaled. Every real body is
    // now built up front (see _initCelestialBodies), not only the one
    // currently traveled to — so this distance has to keep an
    // unfocused body from home view looking like a small, identifiable,
    // non-cluttering background object, not just "far enough for one
    // focused body to look good." Pushed further out than the original
    // single-body-at-a-time range for exactly that reason; the actual
    // TRAVEL close-up shot is unaffected, since _getOrBuildBody's camera
    // offset is computed relative to the body's own position/radius, not
    // to this absolute distance. The radiusUnits floor is a safety net
    // so a large body (Jupiter) can never end up placed closer to the
    // origin than its own surface — see MAX_BODY_RADIUS_UNITS.
    const raw = 22 + 7 * Math.log10(distanceKm / 1e6);
    const compressed = THREE.MathUtils.clamp(raw, 22, 60);
    return Math.max(compressed, radiusUnits * 3.2);
  }

  _getOrBuildBody(bodyKey, meta, position) {
    if (this.bodies[bodyKey]) return this.bodies[bodyKey];

    // Jupiter's TRUE size (radius_km=69,911, ~11x Earth's) would put its
    // true-scale radius (~44 scene units) past this scene's own
    // interplanetary placement range (~26 units) — same "true size,
    // compressed distance" tension the Sun already has, just less
    // extreme. Capped like the Sun's own display radius rather than
    // rendered fully to scale; still clearly the largest planet (Earth
    // is ~4 units) without breaking the scene's geometry. Every other
    // current body (Mercury 1.5 - Earth 4 units) is well under this cap,
    // so this only ever actually changes anything for Jupiter.
    const trueRadiusUnits = GLOBE_RADIUS * (meta.radius_km / R_EARTH_KM);
    const radiusUnits = Math.min(trueRadiusUnits, MAX_BODY_RADIUS_UNITS);
    const displayDistance = this._displayDistanceUnits(position.distance_km, radiusUnits);

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
        uRimColor: { value: new THREE.Color(PALETTE.cyan) },
        uLightDir: { value: lightDir },
      },
      vertexShader: ROCKY_BODY_VERTEX_SHADER,
      fragmentShader: ROCKY_BODY_FRAGMENT_SHADER,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.copy(bodyPosition);
    this.scene.add(mesh);
    this._addWireframeOverlay(mesh, radiusUnits);
    if (bodyKey === 'saturn') {
      // Real proportions: Saturn's main A/B/C ring system spans roughly
      // 1.28x-2.35x its own radius (74,500 km-136,780 km, vs a 58,232 km
      // radius).
      this._addPlanetRings(mesh, radiusUnits, {
        innerRatio: 1.28, outerRatio: 2.35, tiltDeg: 26.7,
        colorHex: 0xd9c99a, maxAlpha: 0.5, wireOpacity: 0.16,
      });
    } else if (bodyKey === 'uranus') {
      // Real proportions: Uranus's main ring system spans roughly
      // 1.5x-2.05x its own radius (~37,850 km-51,150 km, vs a 25,362 km
      // radius) — real, but genuinely much narrower and fainter than
      // Saturn's (not a stylistic understatement; discovered only in
      // 1977, centuries after Saturn's were first resolved).
      this._addPlanetRings(mesh, radiusUnits, {
        innerRatio: 1.5, outerRatio: 2.05, tiltDeg: 97.77,
        colorHex: 0x9fd6d6, maxAlpha: 0.22, wireOpacity: 0.1,
      });
    }

    // A faint line from Earth toward the body along the real direction,
    // so the "real direction" claim reads visually, not just as a number.
    const lineGeometry = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), bodyPosition]);
    const lineMaterial = new THREE.LineBasicMaterial({ color: PALETTE.amber, transparent: true, opacity: 0.25 });
    this.scene.add(new THREE.Line(lineGeometry, lineMaterial));

    // A vertical offset built from a fixed world-Y vector would NOT be
    // guaranteed perpendicular to `direction` (a real astronomical
    // direction, which can point anywhere) — on days it happens to
    // point close to world-Y, adding a raw Y offset partially cancels
    // against the back-offset instead of lifting the camera, landing
    // much closer than intended (confirmed: Mars's real direction today
    // is ~97% aligned with world-Y, which shrank a "3.59x radius" framing
    // distance down to ~2.7x). Building the vertical component from an
    // actual cross product keeps the framing distance/angle consistent
    // regardless of which way the body's real direction happens to point.
    const upHint = Math.abs(direction.y) > 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);
    const sideways = new THREE.Vector3().crossVectors(direction, upHint).normalize();
    const verticalOffset = new THREE.Vector3().crossVectors(sideways, direction).normalize().multiplyScalar(radiusUnits * 0.8);
    const cameraOffset = direction.clone().multiplyScalar(-radiusUnits * 3.5).add(verticalOffset);

    const body = {
      mesh,
      radiusUnits,
      minDistance: radiusUnits * 1.5,
      maxDistance: radiusUnits * 10,
      cameraPosition: bodyPosition.clone().add(cameraOffset),
    };
    this.bodies[bodyKey] = body;
    this._addLabel(bodyKey, mesh, meta.display_name.toUpperCase(), meta.color_hex);
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
          uRimColor: { value: new THREE.Color(PALETTE.cyan) },
          uLightDir: { value: lightDir },
        },
        vertexShader: ROCKY_BODY_VERTEX_SHADER,
        fragmentShader: ROCKY_BODY_FRAGMENT_SHADER,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.copy(meshPosition);
      this.scene.add(mesh);
      this._addWireframeOverlay(mesh, radiusUnits);

      this.moonIndex.push({ key, meta, mesh, position });
      this._addLabel(key, mesh, meta.display_name.toUpperCase(), meta.color_hex);
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

    this.focusedBodyKey = bodyKey;

    if (bodyKey === 'earth') {
      statusEl.textContent = 'RETURNING TO EARTH…';
      // Widen the constraint to whichever is looser (current vs. home)
      // BEFORE the tween starts — otherwise OrbitControls.update()'s own
      // clamp inside the final animation frame (which runs before
      // onComplete) can clip the landing shot against the body we're
      // LEAVING's tighter limits, landing closer than intended.
      this.controls.minDistance = Math.min(this.controls.minDistance, this.homeCamera.minDistance);
      this.controls.maxDistance = Math.max(this.controls.maxDistance, this.homeCamera.maxDistance);
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

    // See the 'earth' branch above for why this widening has to happen
    // before the tween starts, not in onComplete.
    this.controls.minDistance = Math.min(this.controls.minDistance, body.minDistance);
    this.controls.maxDistance = Math.max(this.controls.maxDistance, body.maxDistance);
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
