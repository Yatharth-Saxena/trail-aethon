import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

const MOON_MODEL = "/moon.glb";

// Cubic smoothstep
function smoothstep(min, max, value) {
  const x = Math.max(0, Math.min(1, (value - min) / (max - min)));
  return x * x * (3 - 2 * x);
}

// Quintic smootherstep
function smootherstep(min, max, value) {
  const x = Math.max(0, Math.min(1, (value - min) / (max - min)));
  return x * x * x * (x * (x * 6 - 15) + 10);
}

// 1. High-Resolution Photorealistic Earth Surface Texture (2048x1024)
function createPhotorealisticEarthTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 2048;
  canvas.height = 1024;
  const ctx = canvas.getContext("2d");

  const oceanGrad = ctx.createLinearGradient(0, 0, 0, 1024);
  oceanGrad.addColorStop(0, "#081324");
  oceanGrad.addColorStop(0.3, "#0b1b36");
  oceanGrad.addColorStop(0.5, "#0e2347");
  oceanGrad.addColorStop(0.7, "#091730");
  oceanGrad.addColorStop(1, "#050d1a");
  ctx.fillStyle = oceanGrad;
  ctx.fillRect(0, 0, 2048, 1024);

  const coasts = [
    { x: 440, y: 360, rx: 240, ry: 200 },
    { x: 500, y: 700, rx: 120, ry: 220 },
    { x: 1050, y: 320, rx: 340, ry: 180 },
    { x: 1100, y: 560, rx: 220, ry: 250 },
    { x: 1620, y: 680, rx: 160, ry: 130 },
  ];
  ctx.fillStyle = "rgba(22, 105, 122, 0.45)";
  coasts.forEach((c) => {
    ctx.beginPath();
    ctx.ellipse(c.x, c.y, c.rx * 1.25, c.ry * 1.25, 0, 0, Math.PI * 2);
    ctx.fill();
  });

  const landmasses = [
    { x: 420, y: 320, r: 180, col: "#1f402b" },
    { x: 340, y: 260, r: 140, col: "#2a4c33" },
    { x: 500, y: 400, r: 110, col: "#4a4529" },
    { x: 520, y: 620, r: 160, col: "#153d22" },
    { x: 480, y: 780, r: 100, col: "#2a4530" },
    { x: 1020, y: 260, r: 130, col: "#2d5236" },
    { x: 1080, y: 440, r: 180, col: "#8a7342" },
    { x: 1100, y: 620, r: 160, col: "#1b4728" },
    { x: 1300, y: 280, r: 240, col: "#244d31" },
    { x: 1200, y: 420, r: 130, col: "#7c683c" },
    { x: 1450, y: 460, r: 120, col: "#1d4a2b" },
    { x: 1620, y: 680, r: 140, col: "#7a482b" },
    { x: 1024, y: 960, r: 280, col: "#dce8f2" },
  ];
  landmasses.forEach((lm) => {
    ctx.fillStyle = lm.col;
    ctx.beginPath();
    ctx.arc(lm.x, lm.y, lm.r, 0, Math.PI * 2);
    ctx.fill();
    for (let i = 0; i < 24; i++) {
      const angle = (i / 24) * Math.PI * 2;
      const subR = lm.r * (0.45 + Math.sin(i * 4.7) * 0.35);
      const subX = lm.x + Math.cos(angle) * lm.r * 0.75;
      const subY = lm.y + Math.sin(angle) * lm.r * 0.75;
      ctx.beginPath();
      ctx.arc(subX, subY, subR, 0, Math.PI * 2);
      ctx.fill();
    }
  });

  ctx.fillStyle = "rgba(230, 240, 250, 0.75)";
  [{ x: 380, y: 300, r: 40 }, { x: 460, y: 700, r: 35 }, { x: 1320, y: 360, r: 60 }, { x: 1040, y: 220, r: 30 }]
    .forEach((m) => {
      ctx.beginPath();
      ctx.arc(m.x, m.y, m.r, 0, Math.PI * 2);
      ctx.fill();
    });

  const northCap = ctx.createLinearGradient(0, 0, 0, 100);
  northCap.addColorStop(0, "rgba(240, 248, 255, 0.95)");
  northCap.addColorStop(1, "rgba(240, 248, 255, 0)");
  ctx.fillStyle = northCap;
  ctx.fillRect(0, 0, 2048, 100);

  const southCap = ctx.createLinearGradient(0, 920, 0, 1024);
  southCap.addColorStop(0, "rgba(240, 248, 255, 0)");
  southCap.addColorStop(1, "rgba(240, 248, 255, 0.98)");
  ctx.fillStyle = southCap;
  ctx.fillRect(0, 920, 2048, 104);

  return new THREE.CanvasTexture(canvas);
}

// Dynamically compute exact 3D coordinates and scale to land seamlessly into .eclipse-moon
function getHeaderMoonTarget(camera) {
  const el = document.querySelector(".eclipse-moon");
  if (!el) {
    const aspect = window.innerWidth / window.innerHeight;
    return {
      x: -aspect * 2.2,
      y: 2.1,
      z: 4.2,
      scale: 0.22,
    };
  }

  const rect = el.getBoundingClientRect();
  const screenCenterX = rect.left + rect.width / 2;
  const screenCenterY = rect.top + rect.height / 2;

  // NDC: [-1, 1]
  const ndcX = (screenCenterX / window.innerWidth) * 2 - 1;
  const ndcY = -((screenCenterY / window.innerHeight) * 2 - 1);

  // Depth plane
  const targetZ = 4.2;
  const distZ = camera.position.z - targetZ;

  // Frustum dimensions at targetZ
  const vFovRad = THREE.MathUtils.degToRad(camera.fov / 2);
  const visibleHeight = 2 * distZ * Math.tan(vFovRad);
  const visibleWidth = visibleHeight * (window.innerWidth / window.innerHeight);

  // Exact 3D world position
  const worldX = camera.position.x + ndcX * (visibleWidth / 2);
  const worldY = camera.position.y + ndcY * (visibleHeight / 2);

  // Match Moon3D inside .eclipse-moon:
  // In Moon3D: camera FOV = 25deg, dist = 5.0, model diameter = 1.82
  const moon3dFrustumHeight = 2 * 5.0 * Math.tan(THREE.MathUtils.degToRad(12.5));
  const moon3dDiameterFraction = 1.82 / moon3dFrustumHeight; // ~0.8209
  const headerMoonPixelDiameter = rect.height * moon3dDiameterFraction;

  // Target world diameter in OpeningTransition at visibleHeight
  const targetWorldDiameter = headerMoonPixelDiameter * (visibleHeight / window.innerHeight);
  const targetScale = targetWorldDiameter / 1.1;

  return {
    x: worldX,
    y: worldY,
    z: targetZ,
    scale: targetScale,
  };
}

// Timeline constants (in absolute seconds)
const TIMING = {
  ORBIT_ACCEL_END: 1.5,   // 0.0s -> 1.5s: accelerates around Earth
  SWING_FRONT_END: 2.3,   // 1.5s -> 2.3s: swings around directly in front
  HERO_HOLD_END:   2.7,   // 2.3s -> 2.7s: orbital motion stops, foreground focal point
  JUMP_START:      2.7,   // 2.7s: very fast cinematic warp jump begins
  JUMP_END:        3.1,   // 3.1s: warp jump arrives at .eclipse-moon (400ms duration)
  TOTAL_DURATION:  3.15,  // 3.15s: handoff completes, unmounts
};

export default function OpeningTransition({ onComplete }) {
  const mountRef = useRef(null);
  const phaseRef = useRef("IDLE"); // IDLE | RUNNING | COMPLETE
  const [phaseUI, setPhaseUI] = useState("IDLE");
  const [isEarthHovered, setIsEarthHovered] = useState(false);
  const [reticlePos, setReticlePos] = useState({ x: 50, y: 50, visible: false });
  const [telemetryText, setTelemetryText] = useState("ORBITAL VECTOR NOMINAL // CLICK EARTH TO INITIALIZE");
  const [warpProgress, setWarpProgress] = useState(0);
  const [warpFlash, setWarpFlash] = useState(0);
  const [bgOpacity, setBgOpacity] = useState(1);
  const [uiOpacity, setUiOpacity] = useState(1);

  // Live clock for the header
  const [openingDateTime, setOpeningDateTime] = useState(() => {
    const now = new Date();
    return {
      date: now.toLocaleDateString("en-US", { weekday: "short", day: "numeric", month: "short", year: "numeric" }),
      time: now.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" }),
    };
  });
  useEffect(() => {
    const tick = () => {
      const now = new Date();
      setOpeningDateTime({
        date: now.toLocaleDateString("en-US", { weekday: "short", day: "numeric", month: "short", year: "numeric" }),
        time: now.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" }),
      });
    };
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  // Audio: Lock chime on Earth click
  const playLockChime = () => {
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      const ctx = new AudioCtx();
      const osc1 = ctx.createOscillator();
      const gain1 = ctx.createGain();
      osc1.type = "sine";
      osc1.frequency.setValueAtTime(640, ctx.currentTime);
      osc1.frequency.exponentialRampToValueAtTime(1920, ctx.currentTime + 0.15);
      gain1.gain.setValueAtTime(0.2, ctx.currentTime);
      gain1.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.35);
      osc1.connect(gain1);
      gain1.connect(ctx.destination);
      osc1.start(); osc1.stop(ctx.currentTime + 0.35);

      const osc2 = ctx.createOscillator();
      const gain2 = ctx.createGain();
      osc2.type = "triangle";
      osc2.frequency.setValueAtTime(180, ctx.currentTime + 0.04);
      osc2.frequency.exponentialRampToValueAtTime(50, ctx.currentTime + 0.4);
      gain2.gain.setValueAtTime(0.25, ctx.currentTime + 0.04);
      gain2.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.45);
      osc2.connect(gain2);
      gain2.connect(ctx.destination);
      osc2.start(ctx.currentTime + 0.04); osc2.stop(ctx.currentTime + 0.45);
    } catch (e) { console.warn("Audio:", e); }
  };

  // Audio: Crisp spatial warp whoosh for the 400ms jump
  const playWarpSound = () => {
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      const ctx = new AudioCtx();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.setValueAtTime(340, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(1600, ctx.currentTime + 0.12);
      osc.frequency.exponentialRampToValueAtTime(480, ctx.currentTime + 0.35);
      gain.gain.setValueAtTime(0.01, ctx.currentTime);
      gain.gain.linearRampToValueAtTime(0.22, ctx.currentTime + 0.08);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.38);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(); osc.stop(ctx.currentTime + 0.4);
    } catch (e) { console.warn("Audio:", e); }
  };

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    // ─── Scene ──────────────────────────────────────────────────────────────
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x050607, 0.007);

    const camera = new THREE.PerspectiveCamera(36, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 0, 11.5);

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, powerPreference: "high-performance" });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.25;
    mount.appendChild(renderer.domElement);

    // ─── Cinema Lighting ───────────────────────────────────────────────────
    const sunLight = new THREE.DirectionalLight(0xfffaed, 5.2);
    sunLight.position.set(-9, 5, 8);
    scene.add(sunLight);
    const rimLight = new THREE.DirectionalLight(0x94a3b8, 1.2);
    rimLight.position.set(9, -4, -6);
    scene.add(rimLight);
    const fillLight = new THREE.HemisphereLight(0x7fa3c4, 0x050607, 1.2);
    scene.add(fillLight);

    // ─── Earth Group (tilted 23.4°) ─────────────────────────────────────────
    const earthGroup = new THREE.Group();
    earthGroup.position.set(0, 0, 0);
    earthGroup.rotation.z = THREE.MathUtils.degToRad(23.4);
    scene.add(earthGroup);

    const earthGeo = new THREE.SphereGeometry(2.2, 64, 64);
    const earthMat = new THREE.MeshStandardMaterial({
      map: createPhotorealisticEarthTexture(),
      roughness: 0.55,
      metalness: 0.15,
      transparent: true,
      opacity: 1.0,
    });
    const textureLoader = new THREE.TextureLoader();
    textureLoader.load("/earth.jpg", (tex) => {
      tex.colorSpace = THREE.SRGBColorSpace;
      earthMat.map = tex;
      earthMat.roughness = 0.5;
      earthMat.needsUpdate = true;
    }, undefined, (err) => console.warn("Procedural Earth fallback:", err));

    const earthMesh = new THREE.Mesh(earthGeo, earthMat);
    earthMesh.rotation.y = -1.0; // Start with Europe/Africa/Asia facing camera, not the Pacific
    earthGroup.add(earthMesh);

    // ─── Earth Atmosphere Glow ───────────────────────────────────────────────
    const atmosGeo = new THREE.SphereGeometry(2.28, 48, 48);
    const atmosMat = new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.16,
      side: THREE.BackSide,
      blending: THREE.AdditiveBlending,
    });
    const atmosMesh = new THREE.Mesh(atmosGeo, atmosMat);
    earthGroup.add(atmosMesh);

    // ─── Moon Container ─────────────────────────────────────────────────────
    const moonContainer = new THREE.Group();
    scene.add(moonContainer);

    let moonMesh = null;
    const loader = new GLTFLoader();
    let isDisposed = false;
    loader.load(MOON_MODEL, (gltf) => {
      if (isDisposed) return;
      const model = gltf.scene;
      const box = new THREE.Box3().setFromObject(model);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      const scale = (0.55 * 2) / maxDim;
      model.scale.setScalar(scale);
      model.position.sub(center.multiplyScalar(scale));
      if (moonMesh) moonContainer.remove(moonMesh);
      moonMesh = model;
      moonContainer.add(moonMesh);
    }, undefined, (err) => console.warn("Moon GLTF Load Error:", err));

    // ─── Orbit Line ─────────────────────────────────────────────────────────
    const ORBIT_RADIUS = 5.2;
    const ORBIT_INCLINE = THREE.MathUtils.degToRad(18);
    let orbitAngle = Math.PI * 0.85; // Initial moon position

    const orbitPathPoints = [];
    for (let i = 0; i <= 128; i++) {
      const theta = (i / 128) * Math.PI * 2;
      const x = Math.cos(theta) * ORBIT_RADIUS;
      const y = Math.sin(theta) * ORBIT_RADIUS * Math.sin(ORBIT_INCLINE);
      const z = Math.sin(theta) * ORBIT_RADIUS * Math.cos(ORBIT_INCLINE);
      orbitPathPoints.push(new THREE.Vector3(x, y, z));
    }
    const orbitGeo = new THREE.BufferGeometry().setFromPoints(orbitPathPoints);
    const orbitMat = new THREE.LineDashedMaterial({ color: 0x38bdf8, dashSize: 0.15, gapSize: 0.1, transparent: true, opacity: 0.35 });
    const orbitLine = new THREE.Line(orbitGeo, orbitMat);
    orbitLine.computeLineDistances();
    scene.add(orbitLine);

    // ─── Warp Motion Light Streak ──────────────────────────────────────────
    const streakGeo = new THREE.BufferGeometry();
    const streakPositions = new Float32Array(6);
    streakGeo.setAttribute("position", new THREE.BufferAttribute(streakPositions, 3));
    const streakMat = new THREE.LineBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0,
      blending: THREE.AdditiveBlending,
      linewidth: 3,
    });
    const streakLine = new THREE.Line(streakGeo, streakMat);
    scene.add(streakLine);

    // ─── Starfield ──────────────────────────────────────────────────────────
    const PARTICLE_COUNT = 1800;
    const particleGeo = new THREE.BufferGeometry();
    const posArray = new Float32Array(PARTICLE_COUNT * 3);
    const velocityArray = new Float32Array(PARTICLE_COUNT * 3);
    for (let i = 0; i < PARTICLE_COUNT * 3; i += 3) {
      posArray[i]   = (Math.random() - 0.5) * 80;
      posArray[i+1] = (Math.random() - 0.5) * 50;
      posArray[i+2] = (Math.random() - 0.5) * 60 - 10;
      velocityArray[i+2] = Math.random() * 0.06 + 0.02;
    }
    particleGeo.setAttribute("position", new THREE.BufferAttribute(posArray, 3));
    const starCanvas = document.createElement("canvas");
    starCanvas.width = 16; starCanvas.height = 16;
    const starCtx = starCanvas.getContext("2d");
    const starGrad = starCtx.createRadialGradient(8, 8, 0, 8, 8, 8);
    starGrad.addColorStop(0, "rgba(255,255,255,1)");
    starGrad.addColorStop(0.5, "rgba(180,220,255,0.7)");
    starGrad.addColorStop(1, "rgba(255,255,255,0)");
    starCtx.fillStyle = starGrad;
    starCtx.beginPath(); starCtx.arc(8, 8, 8, 0, Math.PI * 2); starCtx.fill();
    const particleMat = new THREE.PointsMaterial({
      size: 0.18,
      map: new THREE.CanvasTexture(starCanvas),
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      color: 0xf0f8ff,
    });
    const starParticles = new THREE.Points(particleGeo, particleMat);
    scene.add(starParticles);

    // ─── Raycasting for Earth Hover ─────────────────────────────────────────
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2(-999, -999);
    const handleMouseMove = (e) => {
      mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
      mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
    };
    window.addEventListener("mousemove", handleMouseMove);

    // ─── Animation Timeline State ───────────────────────────────────────────
    let animationFrameId;
    const clock = new THREE.Clock();
    let clickTime = null;
    let startOrbitAngle = 0;
    let targetFrontAngle = 0;
    let warpSoundTriggered = false;

    const foregroundCenterPos = new THREE.Vector3(0, 0, 4.2);

    const getOrbitPosition = (angle) => new THREE.Vector3(
      Math.cos(angle) * ORBIT_RADIUS,
      Math.sin(angle) * ORBIT_RADIUS * Math.sin(ORBIT_INCLINE),
      Math.sin(angle) * ORBIT_RADIUS * Math.cos(ORBIT_INCLINE)
    );

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      const delta = Math.min(0.033, clock.getDelta());

      // ── 1. IDLE STATE ─────────────────────────────────────────────────────
      if (phaseRef.current === "IDLE") {
        earthMesh.rotation.y += delta * 0.12;
        orbitAngle += delta * 0.25;

        const pos = getOrbitPosition(orbitAngle);
        moonContainer.position.copy(pos);
        moonContainer.rotation.y += delta * 0.25;
        moonContainer.scale.setScalar(0.55);

        // Hover raycast on Earth
        raycaster.setFromCamera(mouse, camera);
        const hits = raycaster.intersectObject(earthMesh, true);
        const hovered = hits.length > 0;
        setIsEarthHovered(hovered);

        if (hovered && hits[0]) {
          const worldPos = new THREE.Vector3();
          hits[0].object.getWorldPosition(worldPos);
          worldPos.project(camera);
          const screenX = ((worldPos.x + 1) * window.innerWidth) / 2;
          const screenY = ((-worldPos.y + 1) * window.innerHeight) / 2;
          setReticlePos({ x: screenX, y: screenY, visible: true });
        } else {
          setReticlePos((prev) => ({ ...prev, visible: false }));
        }

        earthGroup.scale.lerp(
          new THREE.Vector3(hovered ? 1.04 : 1.0, hovered ? 1.04 : 1.0, hovered ? 1.04 : 1.0),
          1 - Math.exp(-6 * delta)
        );
      }

      // ── 2. RUNNING STATE (PRECISE TIMELINE) ────────────────────────────────
      if (phaseRef.current === "RUNNING") {
        if (clickTime === null) {
          clickTime = clock.getElapsedTime();
          startOrbitAngle = orbitAngle;

          // Target front angle: where the orbit is directly in front of Earth (sin(theta) = 1 => theta = PI/2)
          const currentMod = ((startOrbitAngle % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2);
          let diffToFront = (Math.PI / 2) - currentMod;
          if (diffToFront <= 0) diffToFront += Math.PI * 2;
          // Add 1 full revolution (2 * PI) so the Moon visibly orbits before coming in front
          const totalSweep = diffToFront + Math.PI * 2;
          targetFrontAngle = startOrbitAngle + totalSweep;
        }

        const elapsed = clock.getElapsedTime() - clickTime;
        setWarpProgress(Math.min(1.0, elapsed / TIMING.TOTAL_DURATION));

        // ── Phase A: 0.0s – 1.5s (Moon accelerates in orbit around Earth) ───
        if (elapsed <= TIMING.ORBIT_ACCEL_END) {
          const t = smootherstep(0, TIMING.SWING_FRONT_END, elapsed);
          const currentAngle = THREE.MathUtils.lerp(startOrbitAngle, targetFrontAngle, t);
          orbitAngle = currentAngle;

          const orbitPos = getOrbitPosition(currentAngle);
          moonContainer.position.copy(orbitPos);
          moonContainer.scale.setScalar(0.55);
          moonContainer.rotation.y += delta * 1.2;

          earthMesh.rotation.y += delta * 0.28;
          setTelemetryText("ORBIT ACCELERATION // LUNAR VELOCITY ESCALATING");
        }

        // ── Phase B: 1.5s – 2.3s (Moon swoops directly into center foreground)
        else if (elapsed > TIMING.ORBIT_ACCEL_END && elapsed <= TIMING.SWING_FRONT_END) {
          const orbitT = smootherstep(0, TIMING.SWING_FRONT_END, elapsed);
          const currentAngle = THREE.MathUtils.lerp(startOrbitAngle, targetFrontAngle, orbitT);
          orbitAngle = currentAngle;

          const orbitPos = getOrbitPosition(currentAngle);
          // Pull into center foreground
          const frontFactor = smootherstep(TIMING.ORBIT_ACCEL_END, TIMING.SWING_FRONT_END, elapsed);
          const currentPos = orbitPos.clone().lerp(foregroundCenterPos, frontFactor);

          moonContainer.position.copy(currentPos);
          moonContainer.scale.setScalar(THREE.MathUtils.lerp(0.55, 1.45, frontFactor));
          moonContainer.rotation.y += delta * 0.8;

          earthMesh.rotation.y += delta * 0.15;
          setTelemetryText("ORBIT BREAKAWAY // LUNAR VECTOR CONVERGING TO FOREGROUND");
        }

        // ── Phase C: 2.3s – 2.7s (Moon STOPS orbit, foreground hero focus) ──
        else if (elapsed > TIMING.SWING_FRONT_END && elapsed <= TIMING.HERO_HOLD_END) {
          // Orbital motion has STOPPED! Moon holds hero center position
          moonContainer.position.copy(foregroundCenterPos);
          moonContainer.scale.setScalar(1.45);
          moonContainer.rotation.y += delta * 0.35; // Gentle axial rotation

          setTelemetryText("LUNAR FOREGROUND LOCK // TARGET IDENTIFIED: AETHON HEADER");
        }

        // ── Phase D: 2.7s – 3.1s (THE WARP JUMP: 400ms!) ───────────────────
        else if (elapsed > TIMING.JUMP_START && elapsed <= TIMING.JUMP_END) {
          if (!warpSoundTriggered) {
            playWarpSound();
            warpSoundTriggered = true;
          }

          const jumpP = (elapsed - TIMING.JUMP_START) / (TIMING.JUMP_END - TIMING.JUMP_START);
          // Aggressive warp ease-out: fires out instantly and snaps into position
          const warpT = 1 - Math.pow(1 - jumpP, 4);

          const target = getHeaderMoonTarget(camera);
          const targetPos = new THREE.Vector3(target.x, target.y, target.z);

          const currentJumpPos = foregroundCenterPos.clone().lerp(targetPos, warpT);
          moonContainer.position.copy(currentJumpPos);
          moonContainer.scale.setScalar(THREE.MathUtils.lerp(1.45, target.scale, warpT));
          moonContainer.rotation.y += delta * 0.45;

          // Dynamic light streak trailing the jumping Moon
          const tailPos = foregroundCenterPos.clone().lerp(currentJumpPos, Math.max(0, warpT - 0.35));
          streakLine.geometry.attributes.position.setXYZ(0, tailPos.x, tailPos.y, tailPos.z);
          streakLine.geometry.attributes.position.setXYZ(1, currentJumpPos.x, currentJumpPos.y, currentJumpPos.z);
          streakLine.geometry.attributes.position.needsUpdate = true;
          streakMat.opacity = Math.sin(jumpP * Math.PI) * 0.85;

          // Flash bloom overlay
          setWarpFlash(Math.sin(jumpP * Math.PI));
          setTelemetryText("WARP JUMP // MOON SEATED AT AETHON LOGO");
        }

        // ── Phase E: 3.1s+ (Landing & Seamless Handoff) ─────────────────────
        else if (elapsed > TIMING.JUMP_END) {
          const target = getHeaderMoonTarget(camera);
          moonContainer.position.set(target.x, target.y, target.z);
          moonContainer.scale.setScalar(target.scale);
          streakMat.opacity = 0;
          setWarpFlash(0);

          if (elapsed >= TIMING.TOTAL_DURATION) {
            phaseRef.current = "COMPLETE";
            setPhaseUI("COMPLETE");
            if (onComplete) onComplete();
          }
        }

        // ── Earth Smooth Dissolve (2.3s -> 2.85s) ───────────────────────────
        if (elapsed >= 2.3) {
          const dissolveT = smoothstep(2.3, 2.85, elapsed);
          earthMat.opacity = 1.0 - dissolveT;
          orbitLine.material.opacity = (1.0 - dissolveT) * 0.35;
          earthGroup.position.z = -dissolveT * 4.5;
          earthGroup.scale.setScalar(1.0 - dissolveT * 0.12);

          if (dissolveT >= 1.0) {
            earthGroup.visible = false;
            orbitLine.visible = false;
          }
        }

        // ── Background & UI Dissolve (Revealing AETHON dashboard underneath) ──
        if (elapsed >= 2.3) {
          const uiT = smoothstep(2.3, 2.7, elapsed);
          setUiOpacity(1.0 - uiT);

          const bgT = smoothstep(2.35, 2.9, elapsed);
          setBgOpacity(1.0 - bgT);
        }

        // Starfield Particles during flight
        const starRecession = smoothstep(1.5, 3.1, elapsed);
        const positions = starParticles.geometry.attributes.position.array;
        for (let i = 0; i < PARTICLE_COUNT * 3; i += 3) {
          positions[i + 2] += (0.15 + starRecession * 4.5) * velocityArray[i + 2];
          if (positions[i + 2] > camera.position.z + 5) {
            positions[i + 2] = camera.position.z - 50;
          }
        }
        starParticles.geometry.attributes.position.needsUpdate = true;
      }

      renderer.render(scene, camera);
    };

    animate();

    const handleResize = () => {
      if (!mountRef.current) return;
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };
    window.addEventListener("resize", handleResize);

    return () => {
      isDisposed = true;
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("resize", handleResize);
      scene.traverse((obj) => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
          if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose());
          else obj.material.dispose();
        }
      });
      renderer.dispose();
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement);
    };
  }, []);

  const handleEarthClick = () => {
    if (phaseRef.current !== "IDLE") return;
    playLockChime();
    phaseRef.current = "RUNNING";
    setPhaseUI("RUNNING");
  };

  if (phaseUI === "COMPLETE") return null;

  return (
    <div
      className={`opening-screen-root phase-${phaseUI.toLowerCase()} ${isEarthHovered ? "is-earth-hovered" : ""}`}
      onClick={handleEarthClick}
    >
      {/* Background dark backdrop that smoothly dissolves to reveal dashboard */}
      <div className="opening-backdrop-fade" style={{ opacity: bgOpacity }} />

      {/* WebGL 3D Canvas - Transparent background, Moon is 100% visible at all times */}
      <div ref={mountRef} className="opening-canvas-container" />

      {/* Atmospheric space overlay elements that dissolve with Earth */}
      <div className="opening-vignette" style={{ opacity: bgOpacity }} />
      <div className="opening-grid-overlay" style={{ opacity: bgOpacity }} />

      {/* Full Dashboard-Style Header — mirrors main app header */}
      <div className="opening-top-bar" style={{ opacity: uiOpacity }}>
        <div className="brand-cluster" style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <div className="brand-copy">
            <div className="wordmark" style={{ fontSize: "clamp(1rem,2vw,1.35rem)", fontWeight: 800, letterSpacing: "0.18em", color: "#f0f8ff" }}>AETHON</div>
            <div className="tagline" style={{ fontSize: "clamp(0.55rem,1vw,0.72rem)", letterSpacing: "0.22em", color: "#94a3b8", fontWeight: 600, textTransform: "uppercase" }}>SEE. UNDERSTAND. ASSIST.</div>
          </div>
        </div>
        <div className="opening-status-chip" style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <span style={{ display: "inline-flex", alignItems: "center", gap: "5px", color: "#94a3b8", fontSize: "0.7rem", letterSpacing: "0.12em" }}>
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#94a3b8", display: "inline-block" }} />
            OFFLINE MODE
          </span>
          <span style={{ width: 1, height: 14, background: "rgba(148,163,184,0.3)", display: "inline-block" }} />
          <span style={{ color: "#94a3b8", fontSize: "0.7rem", letterSpacing: "0.1em" }}>{openingDateTime.date}</span>
          <span style={{ width: 1, height: 14, background: "rgba(148,163,184,0.3)", display: "inline-block" }} />
          <span style={{ color: "#94a3b8", fontSize: "0.7rem", letterSpacing: "0.1em" }}>{openingDateTime.time}</span>
          <span style={{ width: 1, height: 14, background: "rgba(148,163,184,0.3)", display: "inline-block" }} />
          <img src="/isro-logo.png" alt="ISRO" style={{ height: 28, opacity: 0.85, objectFit: "contain" }} />
        </div>
      </div>

      {/* Earth Entry Callout */}
      {phaseUI === "IDLE" && (
        <div className={`opening-hover-callout ${isEarthHovered ? "active" : ""}`}>
        <div className="callout-text-box">
            <span className="callout-kicker">EARTH ENTRY POINT IDENTIFIED</span>
            <span className="callout-main">CLICK EARTH TO ENTER AETHON</span>
          </div>
        </div>
      )}


      {/* HUD Bottom Bar & Progress */}
      <div className="opening-bottom-bar" style={{ opacity: uiOpacity }}>
        <div className="telemetry-log-line">
          <span className="telemetry-prompt">&gt;</span>
          <span className="telemetry-content">{telemetryText}</span>
        </div>

        {phaseUI !== "IDLE" && (
          <div className="opening-warp-progress-bar">
            <div
              className="opening-warp-progress-fill"
              style={{ width: `${Math.round(warpProgress * 100)}%` }}
            />
          </div>
        )}

        <div className="opening-corner-telemetry">
          <span>EARTH ORBIT: NOMINAL</span>
          <span>LUNAR DIST: 384,400 KM</span>
          <span>SYS.VER: 1.0.0</span>
        </div>
      </div>

      {/* Instant Warp Flash Flare */}
      {warpFlash > 0.01 && (
        <div
          className="opening-warp-overlay"
          style={{ opacity: warpFlash * 0.45 }}
        />
      )}
    </div>
  );
}
