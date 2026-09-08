import { useEffect, useMemo, useRef, useState, useCallback, Fragment } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import {
  Bot,
  Camera,
  Check,
  ChevronDown,
  FileText,
  Maximize2,
  Menu,
  Mic,
  Pause,
  Play,
  RotateCcw,
  Send,
  Square,
  UserRound,
  Layers,
  AlertTriangle,
  RefreshCw,
  Sliders,
  Volume2,
  VolumeX,
  FlipHorizontal,
  CameraOff
} from "lucide-react";
import { AudioRecorder } from "../lib/audioRecorder.js";
import OpeningTransition from "../components/OpeningTransition";
import Skeleton3D from "../components/Skeleton3D.jsx";

const ISRO_LOGO = "/isro-logo.png";
const ASSEMBLY = "/assembly.png";
const MOON_MODEL = "/moon.glb";
const API_BASE = typeof window !== "undefined" ? window.location.origin : "http://localhost:8000";
const WS_URL = typeof window !== "undefined"
  ? `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/ws`
  : "ws://localhost:8000/ws";

// --- Toast notifications (camera actions confirm here, not in the chat) ---
const TOAST_DURATION_MS = 2800;

// --- Voice activity detection ---
// How long after the last interim speech result the mic stops being treated
// as "hearing speech". Long enough not to flicker between words.
const SPEECH_ACTIVE_DEBOUNCE_MS = 750;

// --- Wake-word state machine ---
const WAKE_STATE = {
  // Mic armed and transcribing, but nothing leaves the browser until the
  // wake word is matched locally.
  IDLE_LISTENING: "IDLE_LISTENING",
  // Wake word heard; everything said from here on is the command.
  CAPTURING_COMMAND: "CAPTURING_COMMAND"
};
// Silence after the wake word that finalises the command and sends it once.
const COMMAND_END_SILENCE_MS = 1300;
// Hard cap on a single command capture, so a stuck session always recovers.
const COMMAND_CAPTURE_TIMEOUT_MS = 12000;
// Wake acknowledgement chime.
const WAKE_CHIME_FREQ_HZ = 880;
const WAKE_CHIME_DURATION_MS = 130;
const WAKE_CHIME_GAIN = 0.07;

function getColorClass(colorName = "") {
  const c = String(colorName).toLowerCase();
  if (c.includes("red") || c.includes("crimson")) return "object-red";
  if (c.includes("cyan")) return "object-cyan";
  if (c.includes("blue")) return "object-blue";
  if (c.includes("green")) return "object-green";
  if (c.includes("yellow")) return "object-yellow";
  if (c.includes("orange")) return "object-orange";
  if (c.includes("purple")) return "object-purple";
  if (c.includes("pink")) return "object-pink";
  if (c.includes("black")) return "object-black";
  if (c.includes("silver") || c.includes("gray")) return "object-gray";
  if (c.includes("brown") || c.includes("wood")) return "object-brown";
  return "object-white";
}

function getCategoryBadgeStyle(cat = "") {
  const c = String(cat).toUpperCase();
  if (c === "TOOL") return { bg: "rgba(245, 158, 11, 0.15)", color: "#f59e0b", border: "rgba(245, 158, 11, 0.4)" };
  if (c === "SAMPLE" || c === "SPECIMEN") return { bg: "rgba(6, 182, 212, 0.15)", color: "#06b6d4", border: "rgba(6, 182, 212, 0.4)" };
  if (c === "TECH" || c === "COMM") return { bg: "rgba(59, 130, 246, 0.15)", color: "#38bdf8", border: "rgba(59, 130, 246, 0.4)" };
  if (c === "CARGO") return { bg: "rgba(16, 185, 129, 0.15)", color: "#10b981", border: "rgba(16, 185, 129, 0.4)" };
  if (c === "CONTROL") return { bg: "rgba(234, 179, 8, 0.15)", color: "#eab308", border: "rgba(234, 179, 8, 0.4)" };
  if (c === "CREW" || c === "ASTRONAUT") return { bg: "rgba(0, 230, 200, 0.15)", color: "#00e6c8", border: "rgba(0, 230, 200, 0.4)" };
  return { bg: "rgba(168, 85, 247, 0.15)", color: "#c084fc", border: "rgba(168, 85, 247, 0.4)" };
}

function mapPerceptionObject(obj) {
  const isAstro = (
    obj.category === "ASTRONAUT" ||
    obj.raw_label === "Astronaut" ||
    obj.label === "Astronaut" ||
    obj.raw_label === "Person" ||
    obj.label === "Person"
  );
  const trackId = obj.track_id || 1;
  const l = obj.label || (isAstro ? "Astronaut" : "Object");
  const colName = obj.color || (isAstro ? "EVA Spacesuit" : "Neutral");
  const disp = isAstro ? (obj.display_name || `✦ ASTRONAUT #${trackId}`) : (obj.display_name || `${colName} ${l}`);
  return {
    label: l,
    raw_label: obj.raw_label || l,
    class_label: obj.class_label || (isAstro ? "Astronaut" : l),
    category: isAstro ? "ASTRONAUT" : (obj.category || "PAYLOAD"),
    category_badge: isAstro ? "CREW" : (obj.category_badge || "ITEM"),
    display_name: disp,
    role: obj.role || (isAstro ? `Mission Operator #${trackId} / EVA Specialist` : ""),
    confidence: obj.confidence || (isAstro ? 0.98 : 0.90),
    colorName: colName,
    color_hex: isAstro ? "#00e6c8" : (obj.color_hex || "#38bdf8"),
    held: Boolean(obj.held || obj.is_held),
    heldBy: obj.held_by || obj.heldBy || "",
    moving: Boolean(obj.moving || obj.is_moving),
    velocity: obj.velocity || 0,
    track_id: trackId,
    track_status: obj.track_status || "TRACKED",
    position: obj.position || null,
    bbox: obj.bbox || [0, 0, 0, 0],
    normalized_bbox: obj.normalized_bbox || null,
    posture: obj.posture || "Seated",
    action: obj.action || "",
    activity: obj.activity || "",
    hands: obj.hands || [],
    colorClass: getColorClass(colName || l)
  };
}

const navItems = [
  { label: "Monitor", icon: Camera },
  { label: "Experiment", icon: Play },
  { label: "Logs", icon: FileText },
];

function GlassPanel({ className = "", children, ...props }) {
  return <section className={`glass-panel ${className}`} {...props}>{children}</section>;
}

function PanelTitle({ title, subtitle, action }) {
  return (
    <div className="panel-title-row">
      <div className="panel-title-copy">
        <h2>{title}</h2>
        {subtitle ? <p>{subtitle}</p> : null}
      </div>
      {action ? <div className="panel-title-action">{action}</div> : null}
    </div>
  );
}

function StatusDot({ label, tone = "white" }) {
  return (
    <span className={`status-dot status-dot-${tone}`}>
      <span className="status-dot-orb" />
      {label}
    </span>
  );
}

function Moon3D({ className = "" }) {
  const mountRef = useRef(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(25, 1, 0.1, 100);
    camera.position.set(0, 0, 5.0);

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setClearColor(0x000000, 0);
    mount.appendChild(renderer.domElement);

    const rig = new THREE.Group();
    rig.rotation.x = THREE.MathUtils.degToRad(5);
    scene.add(rig);

    const keyLight = new THREE.DirectionalLight(0xf3f7fa, 3.4);
    keyLight.position.set(-3.2, 2.2, 4.5);
    scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0x8ea9bb, 1.25);
    rimLight.position.set(3.6, -1.4, -3);
    scene.add(rimLight);

    const fillLight = new THREE.HemisphereLight(0xadbcc7, 0x050709, 1.1);
    scene.add(fillLight);

    mount.dataset.loadState = "loading";
    const loader = new GLTFLoader();
    let disposed = false;

    // Access original moon.glb directly
    loader.load(
      MOON_MODEL,
      (gltf) => {
        if (disposed) return;
        mount.dataset.loadState = "loaded";
        const model = gltf.scene;
        const bounds = new THREE.Box3().setFromObject(model);
        const center = bounds.getCenter(new THREE.Vector3());
        const size = bounds.getSize(new THREE.Vector3());
        const largestDimension = Math.max(size.x, size.y, size.z) || 1;
        model.position.sub(center);
        model.scale.setScalar(1.82 / largestDimension);
        model.rotation.x = THREE.MathUtils.degToRad(4);
        model.traverse((child) => {
          if (child instanceof THREE.Mesh) {
            child.castShadow = true;
            child.receiveShadow = true;
          }
        });
        rig.add(model);
      },
      undefined,
      (err) => {
        console.warn("Moon GLTF Load Error:", err);
        mount.dataset.loadState = "error";
      },
    );

    let isDragging = false;
    let lastX = 0;
    let targetRotation = 0;
    let rotation = 0;
    const clock = new THREE.Clock();

    const onPointerDown = (event) => {
      isDragging = true;
      lastX = event.clientX;
      mount.setPointerCapture?.(event.pointerId);
      mount.dataset.dragging = "true";
    };
    const onPointerMove = (event) => {
      if (!isDragging) return;
      const delta = event.clientX - lastX;
      lastX = event.clientX;
      targetRotation += delta * 0.018;
    };
    const stopDragging = (event) => {
      isDragging = false;
      mount.releasePointerCapture?.(event.pointerId);
      mount.dataset.dragging = "false";
    };

    mount.addEventListener("pointerdown", onPointerDown);
    mount.addEventListener("pointermove", onPointerMove);
    mount.addEventListener("pointerup", stopDragging);
    mount.addEventListener("pointercancel", stopDragging);
    mount.addEventListener("pointerleave", (event) => { if (isDragging) stopDragging(event); });

    const resize = () => {
      const width = Math.max(1, mount.clientWidth);
      const height = Math.max(1, mount.clientHeight);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(mount);
    resize();

    let frame = 0;
    const render = () => {
      const delta = Math.min(clock.getDelta(), 0.05);
      if (!isDragging) targetRotation += delta * 0.46;
      rotation += (targetRotation - rotation) * 0.12;
      rig.rotation.y = rotation;
      renderer.render(scene, camera);
      frame = requestAnimationFrame(render);
    };
    render();

    return () => {
      disposed = true;
      cancelAnimationFrame(frame);
      resizeObserver.disconnect();
      mount.removeEventListener("pointerdown", onPointerDown);
      mount.removeEventListener("pointermove", onPointerMove);
      mount.removeEventListener("pointerup", stopDragging);
      mount.removeEventListener("pointercancel", stopDragging);
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  return <div ref={mountRef} className={`moon-3d-mount ${className}`} role="img" aria-label="Interactive rotating moon model" title="Hold and drag to rotate the moon" />;
}

export default function Home() {
  // Opening transition state
  const [isIntroActive, setIsIntroActive] = useState(true);

  // Navigation & View state
  const [currentView, setCurrentView] = useState("Monitor");

  // Camera & Telemetry state
  const [fps, setFps] = useState(60);
  // Real capture -> display latency reported by the backend, not just FPS.
  const [latencyMs, setLatencyMs] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [cameras, setCameras] = useState([{ index: 0, name: "Camera 0 (USB)", active: true }]);
  const [selectedCamera, setSelectedCamera] = useState(0);
  const [showCameraMenu, setShowCameraMenu] = useState(false);
  const [streamKey, setStreamKey] = useState(Date.now());
  const [useBrowserWebcam, setUseBrowserWebcam] = useState(true);
  const [browserWebcams, setBrowserWebcams] = useState([]);
  const [selectedBrowserCamId, setSelectedBrowserCamId] = useState(() => {
    try {
      return localStorage.getItem("aethon_selected_camera_id") || null;
    } catch (e) {
      return null;
    }
  });
  const [cameraError, setCameraError] = useState(null);
  const [cameraLoading, setCameraLoading] = useState(false);

  useEffect(() => {
    const updateDevices = () => {
      if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
        navigator.mediaDevices.enumerateDevices().then((devices) => {
          const rawInputs = devices.filter((d) => d.kind === "videoinput");
          setBrowserWebcams(rawInputs);
          if (rawInputs.length > 0) {
            const savedId = localStorage.getItem("aethon_selected_camera_id");
            const hasSaved = rawInputs.some((d) => d.deviceId === savedId);
            if (hasSaved) {
              setSelectedBrowserCamId(savedId);
            } else if (!selectedBrowserCamId) {
              setSelectedBrowserCamId(rawInputs[0].deviceId);
            }
          }
        });
      }
    };
    updateDevices();
    if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
      navigator.mediaDevices.addEventListener("devicechange", updateDevices);
      return () => navigator.mediaDevices.removeEventListener("devicechange", updateDevices);
    }
  }, []);
  const [isCameraOn, setIsCameraOn] = useState(true);
  const [mirrorFeed, setMirrorFeed] = useState(true);
  const mirrorFeedRef = useRef(true);
  useEffect(() => { mirrorFeedRef.current = mirrorFeed; }, [mirrorFeed]);
  const cameraStageRef = useRef(null);
  const videoRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const latestPerceptionRef = useRef(null);
  const wsRef = useRef(null);
  const prevPoseLandmarksRef = useRef({});
  const prevHandLandmarksRef = useRef({});

  // Browser Webcam MediaDevices & AI streaming pipeline effect
  useEffect(() => {
    let localStream = null;
    let uploadInterval = null;
    let animId = null;
    let isUploading = false;
    let isDisposed = false;
    const captureCanvas = document.createElement("canvas");

    // Clear stale landmark smoothing cache when switching camera sources
    prevPoseLandmarksRef.current = {};
    prevHandLandmarksRef.current = {};
    latestPerceptionRef.current = null;

    if (useBrowserWebcam && isCameraOn) {
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        setCameraLoading(true);
        setCameraError(null);

        const videoConstraints = selectedBrowserCamId
          ? {
              deviceId: { exact: selectedBrowserCamId },
              width: { ideal: 1280 },
              height: { ideal: 720 }
            }
          : {
              width: { ideal: 1280 },
              height: { ideal: 720 }
            };

        navigator.mediaDevices
          .getUserMedia({ video: videoConstraints })
          .then((stream) => {
            if (isDisposed) {
              stream.getTracks().forEach((t) => t.stop());
              return;
            }
            localStream = stream;
            setCameraLoading(false);
            setCameraError(null);

            if (videoRef.current) {
              videoRef.current.srcObject = stream;
            }
            if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
              navigator.mediaDevices.enumerateDevices().then((devices) => {
                const rawInputs = devices.filter((d) => d.kind === "videoinput");
                setBrowserWebcams(rawInputs);
              });
            }

            // 1. Ultra low-latency frame uploader: sends browser webcam frames directly over WebSocket binary channel
            uploadInterval = setInterval(() => {
              const video = videoRef.current;
              if (!video || video.readyState < 2 || isUploading) return;
              const w = 480;
              const h = Math.round((video.videoHeight / (video.videoWidth || 1)) * w) || 270;
              if (captureCanvas.width !== w || captureCanvas.height !== h) {
                captureCanvas.width = w;
                captureCanvas.height = h;
              }
              const ctx = captureCanvas.getContext("2d");
              if (!ctx) return;
              ctx.drawImage(video, 0, 0, w, h);

              const ws = wsRef.current;
              if (ws && ws.readyState === WebSocket.OPEN) {
                // Prevent buffer bloat / network queue delay: skip if socket buffer is backed up
                if (ws.bufferedAmount > 65536) return;
                isUploading = true;
                captureCanvas.toBlob(
                  (blob) => {
                    try {
                      if (blob && ws.readyState === WebSocket.OPEN) {
                        ws.send(blob);
                      }
                    } catch (e) {
                    } finally {
                      isUploading = false;
                    }
                  },
                  "image/jpeg",
                  0.48
                );
              } else {
                captureCanvas.toBlob(
                  (blob) => {
                    if (!blob) return;
                    isUploading = true;
                    fetch(`${API_BASE}/api/camera/upload_frame`, {
                      method: "POST",
                      body: blob,
                      headers: { "Content-Type": "image/jpeg" }
                    })
                      .catch(() => {})
                      .finally(() => {
                        isUploading = false;
                      });
                  },
                  "image/jpeg",
                  0.48
                );
              }
            }, 30); // ~33 FPS continuous low-latency ingestion

            // 2. Real-time overlay canvas drawing loop
            const renderOverlays = () => {
  try {
              const canvas = overlayCanvasRef.current;
              const video = videoRef.current;
              if (canvas && video && video.readyState >= 2) {
                const cw = video.clientWidth;
                const ch = video.clientHeight;
                if (canvas.width !== cw || canvas.height !== ch) {
                  canvas.width = cw;
                  canvas.height = ch;
                }
                const ctx = canvas.getContext("2d");
                if (ctx) {
                  ctx.clearRect(0, 0, cw, ch);
                  const p = latestPerceptionRef.current;
                  if (p) {
                    const vw = video.videoWidth || (p.frame_width || 1280);
                    const vh = video.videoHeight || (p.frame_height || 720);
                    const videoAR = vw / vh;
                    const canvasAR = cw / ch;

                    let renderWidth = cw;
                    let renderHeight = ch;
                    let offsetX = 0;
                    let offsetY = 0;

                    if (videoAR > canvasAR) {
                      renderHeight = ch;
                      renderWidth = ch * videoAR;
                      offsetX = (cw - renderWidth) / 2;
                    } else {
                      renderWidth = cw;
                      renderHeight = cw / videoAR;
                      offsetY = (ch - renderHeight) / 2;
                    }

                    const mirror = mirrorFeedRef.current;

                    const mapNormalizedPoint = (nx, ny, jointKey = null, refStore = null) => {
                      const rawX = offsetX + nx * renderWidth;
                      const rawY = offsetY + ny * renderHeight;
                      let targetX = rawX;
                      let targetY = rawY;

                      if (jointKey !== null && refStore && refStore.current) {
                        const prev = refStore.current[jointKey];
                        if (prev) {
                          const dist = Math.hypot(rawX - prev.x, rawY - prev.y);
                          const alpha = Math.min(0.96, Math.max(0.58, dist / 20.0));
                          targetX = prev.x + (rawX - prev.x) * alpha;
                          targetY = prev.y + (rawY - prev.y) * alpha;
                        }
                        refStore.current[jointKey] = { x: targetX, y: targetY };
                      }

                      const finalX = mirror ? (cw - targetX) : targetX;
                      return { x: finalX, y: targetY };
                    };

                    // Draw Anatomical Body Skeleton Mesh (Multi-Pose)
                    const posesToRender = p.poses || (p.pose ? (p.pose.all_poses || [p.pose]) : []);
                    posesToRender.forEach((poseItem, poseIdx) => {
                      if (!poseItem || !poseItem.landmarks) return;
                      const lms = poseItem.landmarks;
                      const pt = (i) => {
                        const lm = lms[i];
                        if (!lm || (lm.visibility ?? 1) <= 0.20) return null;
                        const pMapped = mapNormalizedPoint(lm.x, lm.y, `pose-${poseIdx}-${i}`, prevPoseLandmarksRef);
                        return {
                          x: pMapped.x,
                          y: pMapped.y,
                          vis: lm.visibility ?? 1
                        };
                      };

                      const p11 = pt(11); // L Shoulder
                      const p12 = pt(12); // R Shoulder
                      const p23 = pt(23); // L Hip
                      const p24 = pt(24); // R Hip

                      // 1. Torso Biometric Polygonal Mesh Facets
                      if (p11 && p12 && p23 && p24) {
                        const sternumX = (p11.x + p12.x) * 0.5;
                        const sternumY = (p11.y + p12.y) * 0.5;
                        const midHipX = (p23.x + p24.x) * 0.5;
                        const midHipY = (p23.y + p24.y) * 0.5;
                        const solarPlexusX = sternumX * 0.35 + midHipX * 0.65;
                        const solarPlexusY = sternumY * 0.35 + midHipY * 0.65;

                        ctx.save();
                        // Translucent Torso Mesh Fill
                        ctx.beginPath();
                        ctx.moveTo(p11.x, p11.y);
                        ctx.lineTo(p12.x, p12.y);
                        ctx.lineTo(p24.x, p24.y);
                        ctx.lineTo(p23.x, p23.y);
                        ctx.closePath();
                        ctx.fillStyle = "rgba(0, 230, 200, 0.12)";
                        ctx.fill();

                        // Torso Wireframe Lattice Struts (Cross-Mesh Bracing)
                        ctx.strokeStyle = "rgba(0, 230, 200, 0.35)";
                        ctx.lineWidth = 1;
                        ctx.beginPath();
                        ctx.moveTo(p11.x, p11.y); ctx.lineTo(p24.x, p24.y);
                        ctx.moveTo(p12.x, p12.y); ctx.lineTo(p23.x, p23.y);
                        ctx.moveTo(sternumX, sternumY); ctx.lineTo(p23.x, p23.y);
                        ctx.moveTo(sternumX, sternumY); ctx.lineTo(p24.x, p24.y);
                        ctx.stroke();

                        // Central Vertebral Spine Column
                        ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
                        ctx.lineWidth = 2.5;
                        ctx.beginPath();
                        ctx.moveTo(sternumX, sternumY);
                        ctx.lineTo(midHipX, midHipY);
                        ctx.stroke();

                        // Solar plexus biometric core node
                        ctx.fillStyle = "#00e6c8";
                        ctx.beginPath();
                        ctx.arc(solarPlexusX, solarPlexusY, 4, 0, Math.PI * 2);
                        ctx.fill();
                        ctx.restore();
                      }

                      // Extract active body and hand gestures for direct skeletal mesh synchronization
                      const activeGesture = poseItem.primary_gesture || p.primary_gesture || (p.gestures && p.gestures.primary_gesture) || "NONE";
                      const bodyGestures = poseItem.gestures || p.body_gestures || (p.gestures && p.gestures.body_gestures) || [];
                      const hasRaisedRight = bodyGestures.some(bg => bg.gesture === "RIGHT_HAND_RAISED") || activeGesture === "RIGHT_HAND_RAISED";
                      const hasRaisedLeft = bodyGestures.some(bg => bg.gesture === "LEFT_HAND_RAISED") || activeGesture === "LEFT_HAND_RAISED";
                      const hasRaisedBoth = bodyGestures.some(bg => bg.gesture === "BOTH_HANDS_RAISED") || activeGesture === "BOTH_HANDS_RAISED";
                      const hasSaluteRight = bodyGestures.some(bg => bg.gesture === "SALUTE_RIGHT") || activeGesture === "SALUTE_RIGHT";
                      const hasSaluteLeft = bodyGestures.some(bg => bg.gesture === "SALUTE_LEFT") || activeGesture === "SALUTE_LEFT";
                      const hasArmsCrossed = bodyGestures.some(bg => bg.gesture === "ARMS_CROSSED") || activeGesture === "ARMS_CROSSED";
                      const hasTPose = bodyGestures.some(bg => bg.gesture === "T_POSE") || activeGesture === "T_POSE";

                      const handGestureMap = {};
                      if (p.hands) {
                        for (const h of p.hands) {
                          const g = h.gesture;
                          if (g && g !== "NONE" && g !== "OPEN_HAND" && g !== "REACHING") {
                            handGestureMap[h.side || "Right"] = {
                              gesture: g,
                              confidence: h.gesture_confidence || 0.92
                            };
                          }
                        }
                      }

                      // 2. Complete Human Anatomical Skeleton Connections
                      const connections = [
                        // Clavicle / Shoulders
                        [11, 12],
                        // Left Arm (upper arm, forearm)
                        [11, 13], [13, 15],
                        // Right Arm (upper arm, forearm)
                        [12, 14], [14, 16],
                        // Left Hand Anchors
                        [15, 17], [15, 19], [15, 21], [17, 19],
                        // Right Hand Anchors
                        [16, 18], [16, 20], [16, 22], [18, 20],
                        // Torso / Flanks & Pelvis
                        [11, 23], [12, 24], [23, 24],
                        // Left Leg (thigh, shin, foot)
                        [23, 25], [25, 27], [27, 29], [29, 31], [27, 31],
                        // Right Leg (thigh, shin, foot)
                        [24, 26], [26, 28], [28, 30], [30, 32], [28, 32],
                        // Head / Facial Perimeter
                        [0, 1], [1, 2], [2, 3], [3, 7],
                        [0, 4], [4, 5], [5, 6], [6, 8],
                        [9, 10]
                      ];

                      // Anatomical Cervical Spine / Neck (from sternum to nose/cranium)
                      const p0 = pt(0);
                      if (p0 && p11 && p12) {
                        const sternumX = (p11.x + p12.x) * 0.5;
                        const sternumY = (p11.y + p12.y) * 0.5;
                        ctx.strokeStyle = "rgba(0, 230, 200, 0.9)";
                        ctx.lineWidth = 3.0;
                        ctx.beginPath();
                        ctx.moveTo(sternumX, sternumY);
                        ctx.lineTo(p0.x, p0.y);
                        ctx.stroke();

                        // Cranial contour halo
                        ctx.fillStyle = "rgba(0, 230, 200, 0.15)";
                        ctx.beginPath();
                        ctx.arc(p0.x, p0.y, 16, 0, Math.PI * 2);
                        ctx.fill();
                        ctx.strokeStyle = "rgba(0, 230, 200, 0.6)";
                        ctx.lineWidth = 1.2;
                        ctx.stroke();
                      }

                      for (const [idx1, idx2] of connections) {
                        const a = pt(idx1);
                        const b = pt(idx2);
                        if (a && b) {
                          const isRightArm = (idx1 === 12 && idx2 === 14) || (idx1 === 14 && idx2 === 16);
                          const isLeftArm = (idx1 === 11 && idx2 === 13) || (idx1 === 13 && idx2 === 15);
                          const isRightActive = isRightArm && (hasRaisedRight || hasRaisedBoth || hasSaluteRight || hasArmsCrossed || hasTPose || handGestureMap["Right"]);
                          const isLeftActive = isLeftArm && (hasRaisedLeft || hasRaisedBoth || hasSaluteLeft || hasArmsCrossed || hasTPose || handGestureMap["Left"]);

                          if (isRightActive || isLeftActive) {
                            ctx.strokeStyle = "#38bdf8";
                            ctx.lineWidth = 4.0;
                          } else {
                            const isLimb = idx1 >= 13 || idx2 >= 13;
                            ctx.strokeStyle = isLimb ? "rgba(56, 189, 248, 0.85)" : "rgba(0, 230, 200, 0.85)";
                            ctx.lineWidth = isLimb ? 2.5 : 3.0;
                          }
                          ctx.beginPath();
                          ctx.moveTo(a.x, a.y);
                          ctx.lineTo(b.x, b.y);
                          ctx.stroke();
                        }
                      }

                      // 3. Biometric Articulation Nodes (Concentric Joint Rings)
                      for (let i = 0; i < lms.length; i++) {
                        const pNode = pt(i);
                        if (pNode) {
                          const isMajor = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28].includes(i);
                          const isGesturingJoint = (i === 16 && (hasRaisedRight || hasSaluteRight || handGestureMap["Right"])) ||
                                                   (i === 15 && (hasRaisedLeft || hasSaluteLeft || handGestureMap["Left"])) ||
                                                   ((i === 13 || i === 14) && (hasRaisedBoth || hasArmsCrossed || hasTPose));

                          if (isGesturingJoint) {
                            ctx.strokeStyle = "#38bdf8";
                            ctx.lineWidth = 2.5;
                            ctx.beginPath();
                            ctx.arc(pNode.x, pNode.y, 8, 0, Math.PI * 2);
                            ctx.stroke();

                            ctx.fillStyle = "#ffffff";
                            ctx.beginPath();
                            ctx.arc(pNode.x, pNode.y, 3.5, 0, Math.PI * 2);
                            ctx.fill();
                          } else if (isMajor) {
                            ctx.strokeStyle = "rgba(0, 230, 200, 0.75)";
                            ctx.lineWidth = 1.5;
                            ctx.beginPath();
                            ctx.arc(pNode.x, pNode.y, 6, 0, Math.PI * 2);
                            ctx.stroke();

                            ctx.fillStyle = "#ffffff";
                            ctx.beginPath();
                            ctx.arc(pNode.x, pNode.y, 2.5, 0, Math.PI * 2);
                            ctx.fill();
                          } else {
                            ctx.fillStyle = "rgba(255, 255, 255, 0.85)";
                            ctx.beginPath();
                            ctx.arc(pNode.x, pNode.y, 2, 0, Math.PI * 2);
                            ctx.fill();
                          }
                        }
                      }

                      // 4. On-Limb Body Gesture HUD Badges (Synchronized at Skeleton Nodes)
                      const drawLimbBadge = (ptNode, text, color = "#38bdf8") => {
                        if (!ptNode) return;
                        ctx.font = "700 10px Inter, sans-serif";
                        const tw = ctx.measureText(text).width;
                        const bx = Math.max(8, ptNode.x - tw / 2 - 6);
                        const by = Math.max(16, ptNode.y - 18);
                        ctx.fillStyle = "rgba(10, 15, 20, 0.88)";
                        ctx.fillRect(bx, by, tw + 12, 16);
                        ctx.strokeStyle = color;
                        ctx.lineWidth = 1;
                        ctx.strokeRect(bx, by, tw + 12, 16);
                        ctx.fillStyle = color;
                        ctx.fillText(text, bx + 6, by + 12);
                      };

                      if (hasRaisedBoth && pt(15) && pt(16)) {
                        drawLimbBadge({ x: (pt(15).x + pt(16).x) * 0.5, y: Math.min(pt(15).y, pt(16).y) - 10 }, "▲ BOTH ARMS RAISED", "#38bdf8");
                      } else {
                        if (hasRaisedRight && pt(16)) drawLimbBadge(pt(16), "▲ RIGHT ARM RAISED", "#38bdf8");
                        if (hasRaisedLeft && pt(15)) drawLimbBadge(pt(15), "▲ LEFT ARM RAISED", "#38bdf8");
                      }
                      if (hasSaluteRight && pt(16)) drawLimbBadge(pt(16), "★ SALUTE [R]", "#fbbf24");
                      if (hasSaluteLeft && pt(15)) drawLimbBadge(pt(15), "★ SALUTE [L]", "#fbbf24");
                      if (hasArmsCrossed && p11 && p12) {
                        drawLimbBadge({ x: (p11.x + p12.x) * 0.5, y: (p11.y + p12.y) * 0.5 + 20 }, "✦ ARMS CROSSED", "#38bdf8");
                      }
                      if (hasTPose && p11 && p12) {
                        drawLimbBadge({ x: (p11.x + p12.x) * 0.5, y: (p11.y + p12.y) * 0.5 - 15 }, "⟷ T-POSE", "#38bdf8");
                      }
                    });

                    // Draw Objects & Astronaut Bounding Boxes
                    let astronautDrawn = false;
                    if (p.objects && p.objects.length > 0) {
                      const fw = p.frame_width || vw;
                      const fh = p.frame_height || vh;
                      for (const obj of p.objects) {
                        if (!obj.bbox || obj.bbox.length < 4) continue;
                        if (obj.track_status === "CANDIDATE") continue;

                        let nbx1, nby1, nbx2, nby2;
                        if (obj.normalized_bbox && obj.normalized_bbox.length >= 4) {
                          [nbx1, nby1, nbx2, nby2] = obj.normalized_bbox;
                        } else {
                          nbx1 = obj.bbox[0] / fw;
                          nby1 = obj.bbox[1] / fh;
                          nbx2 = obj.bbox[2] / fw;
                          nby2 = obj.bbox[3] / fh;
                        }

                        const pt1 = mapNormalizedPoint(nbx1, nby1);
                        const pt2 = mapNormalizedPoint(nbx2, nby2);
                        const wBox = Math.abs(pt2.x - pt1.x);
                        const hBox = Math.abs(pt2.y - pt1.y);
                        const x = Math.min(pt1.x, pt2.x);
                        const y = Math.min(pt1.y, pt2.y);

                        const isAstronaut = (
                          obj.category === "ASTRONAUT" ||
                          obj.raw_label === "Astronaut" ||
                          obj.label === "Astronaut" ||
                          obj.raw_label === "Person" ||
                          obj.label === "Person"
                        );

                        if (isAstronaut) {
                          // Reject ceiling false-positives
                          if (nby2 < 0.35) continue;
                          // Corroborate with verified astronaut pose skeleton
                          const posesCheck = p.poses || (p.pose ? (p.pose.all_poses || [p.pose]) : []);
                          if (posesCheck.length === 0) continue;

                          astronautDrawn = true;

                          // Draw Holographic Cyan Corner Brackets for Astronaut
                          const astroColor = "#00e6c8";
                          const cornerLen = Math.min(26, Math.max(12, Math.min(wBox, hBox) * 0.22));

                          ctx.strokeStyle = astroColor;
                          ctx.lineWidth = 2.5;

                          // Top-left
                          ctx.beginPath();
                          ctx.moveTo(x, y + cornerLen);
                          ctx.lineTo(x, y);
                          ctx.lineTo(x + cornerLen, y);
                          ctx.stroke();

                          // Top-right
                          ctx.beginPath();
                          ctx.moveTo(x + wBox - cornerLen, y);
                          ctx.lineTo(x + wBox, y);
                          ctx.lineTo(x + wBox, y + cornerLen);
                          ctx.stroke();

                          // Bottom-left
                          ctx.beginPath();
                          ctx.moveTo(x, y + hBox - cornerLen);
                          ctx.lineTo(x, y + hBox);
                          ctx.lineTo(x + cornerLen, y + hBox);
                          ctx.stroke();

                          // Bottom-right
                          ctx.beginPath();
                          ctx.moveTo(x + wBox - cornerLen, y + hBox);
                          ctx.lineTo(x + wBox, y + hBox);
                          ctx.lineTo(x + wBox, y + hBox - cornerLen);
                          ctx.stroke();

                          // Subtle perimeter outline
                          ctx.strokeStyle = "rgba(0, 230, 200, 0.25)";
                          ctx.lineWidth = 1;
                          ctx.strokeRect(x, y, wBox, hBox);

                          const trackId = obj.track_id || 1;
                          const postureStr = (obj.posture || p.posture || "SEATED").toUpperCase();
                          const actionStr = (obj.action || obj.activity || (p.current_action && p.current_action.label) || "ACTIVE").toUpperCase();
                          const labelText = `✦ ASTRONAUT #${trackId} // [${actionStr}] (${postureStr})`;
                          ctx.font = "700 11px Inter, sans-serif";
                          const tw = ctx.measureText(labelText).width;
                          const badgeH = 20;
                          const badgeY = Math.max(0, y - badgeH - 2);

                          ctx.fillStyle = "rgba(8, 20, 28, 0.92)";
                          ctx.fillRect(x, badgeY, tw + 18, badgeH);
                          ctx.strokeStyle = astroColor;
                          ctx.lineWidth = 1.2;
                          ctx.strokeRect(x, badgeY, tw + 18, badgeH);

                          // Cyan pulsing reticle dot
                          ctx.fillStyle = astroColor;
                          ctx.beginPath();
                          ctx.arc(x + 9, badgeY + 10, 3.5, 0, 2 * Math.PI);
                          ctx.fill();

                          ctx.fillStyle = "#ffffff";
                          ctx.fillText(labelText, x + 18, badgeY + 14);
                          continue;
                        }

                        // Everyday & Payload Objects
                        const objColor = obj.color_hex || "#38bdf8";
                        ctx.strokeStyle = objColor;
                        ctx.lineWidth = 2;
                        ctx.strokeRect(x, y, wBox, hBox);

                        // Tactical HUD corner ticks
                        const tickLen = Math.min(10, Math.max(6, Math.min(wBox, hBox) * 0.15));
                        ctx.lineWidth = 3;
                        ctx.beginPath();
                        ctx.moveTo(x, y + tickLen); ctx.lineTo(x, y); ctx.lineTo(x + tickLen, y);
                        ctx.moveTo(x + wBox - tickLen, y); ctx.lineTo(x + wBox, y); ctx.lineTo(x + wBox, y + tickLen);
                        ctx.moveTo(x, y + hBox - tickLen); ctx.lineTo(x, y + hBox); ctx.lineTo(x + tickLen, y + hBox);
                        ctx.moveTo(x + wBox - tickLen, y + hBox); ctx.lineTo(x + wBox, y + hBox); ctx.lineTo(x + wBox, y + hBox - tickLen);
                        ctx.stroke();

                        // Badges and text
                        const catBadge = obj.category_badge || (obj.category || "ITEM");
                        const heldTag = (obj.is_held || obj.held) ? `● HELD: ${obj.held_by || obj.heldBy || "Crew"} | ` : "";
                        const moveTag = (obj.moving || obj.is_moving) && !(obj.is_held || obj.held) ? "▲ MOVING | " : "";
                        const posStr = obj.position ? ` (${obj.position.cx},${obj.position.cy})` : "";
                        const labelText = `[${catBadge}] ${heldTag}${moveTag}${obj.display_name || obj.label}${posStr}`;

                        ctx.font = "600 11px Inter, sans-serif";
                        const tw = ctx.measureText(labelText).width;
                        const badgeH = 19;
                        const badgeY = Math.max(0, y - badgeH - 2);

                        ctx.fillStyle = (obj.is_held || obj.held) ? "rgba(234, 179, 8, 0.92)" : "rgba(13, 20, 28, 0.90)";
                        ctx.fillRect(x, badgeY, tw + 12, badgeH);
                        ctx.strokeStyle = (obj.is_held || obj.held) ? "#eab308" : objColor;
                        ctx.lineWidth = 1;
                        ctx.strokeRect(x, badgeY, tw + 12, badgeH);

                        ctx.fillStyle = (obj.is_held || obj.held) ? "#0d1117" : "#f1f5f9";
                        ctx.fillText(labelText, x + 6, badgeY + 13.5);
                      }
                    }

                    // Fallback: only if astronaut was NOT already drawn from objects, but verified pose is present
                    if (!astronautDrawn && p.person_detected && p.pose && p.pose.landmarks && p.pose.landmarks.length >= 6) {
                      const fw = p.frame_width || vw;
                      const fh = p.frame_height || vh;
                      let nbx1, nby1, nbx2, nby2;
                      if (p.pose.normalized_bbox && p.pose.normalized_bbox.length >= 4) {
                        [nbx1, nby1, nbx2, nby2] = p.pose.normalized_bbox;
                      } else if (p.pose.bbox && p.pose.bbox.length >= 4) {
                        nbx1 = p.pose.bbox[0] / fw;
                        nby1 = p.pose.bbox[1] / fh;
                        nbx2 = p.pose.bbox[2] / fw;
                        nby2 = p.pose.bbox[3] / fh;
                      } else {
                        const vis = p.pose.landmarks.filter(lm => (lm.visibility ?? 1) > 0.28);
                        if (vis.length >= 6) {
                          nbx1 = Math.max(0, Math.min(...vis.map(l => l.x)) - 0.04);
                          nbx2 = Math.min(1, Math.max(...vis.map(l => l.x)) + 0.04);
                          nby1 = Math.max(0, Math.min(...vis.map(l => l.y)) - 0.04);
                          nby2 = Math.min(1, Math.max(...vis.map(l => l.y)) + 0.04);
                        }
                      }

                      if (nbx1 !== undefined && nbx2 !== undefined) {
                        const pt1 = mapNormalizedPoint(nbx1, nby1);
                        const pt2 = mapNormalizedPoint(nbx2, nby2);
                        const wBox = Math.abs(pt2.x - pt1.x);
                        const hBox = Math.abs(pt2.y - pt1.y);
                        const x = Math.min(pt1.x, pt2.x);
                        const y = Math.min(pt1.y, pt2.y);

                        const astroColor = "#00e6c8";
                        const cornerLen = Math.min(26, Math.max(12, Math.min(wBox, hBox) * 0.22));

                        ctx.strokeStyle = astroColor;
                        ctx.lineWidth = 2.5;

                        ctx.beginPath();
                        ctx.moveTo(x, y + cornerLen); ctx.lineTo(x, y); ctx.lineTo(x + cornerLen, y);
                        ctx.moveTo(x + wBox - cornerLen, y); ctx.lineTo(x + wBox, y); ctx.lineTo(x + wBox, y + cornerLen);
                        ctx.moveTo(x, y + hBox - cornerLen); ctx.lineTo(x, y + hBox); ctx.lineTo(x + cornerLen, y + hBox);
                        ctx.moveTo(x + wBox - cornerLen, y + hBox); ctx.lineTo(x + wBox, y + hBox); ctx.lineTo(x + wBox, y + hBox - cornerLen);
                        ctx.stroke();

                        ctx.strokeStyle = "rgba(0, 230, 200, 0.25)";
                        ctx.lineWidth = 1;
                        ctx.strokeRect(x, y, wBox, hBox);

                        const postureStr = (p.posture || "SEATED").toUpperCase();
                        const labelText = `✦ ASTRONAUT #1 // SKELETON ACTIVE [${postureStr}]`;
                        ctx.font = "700 11px Inter, sans-serif";
                        const tw = ctx.measureText(labelText).width;
                        const badgeH = 20;
                        const badgeY = Math.max(0, y - badgeH - 2);

                        ctx.fillStyle = "rgba(8, 20, 28, 0.92)";
                        ctx.fillRect(x, badgeY, tw + 18, badgeH);
                        ctx.strokeStyle = astroColor;
                        ctx.lineWidth = 1.2;
                        ctx.strokeRect(x, badgeY, tw + 18, badgeH);

                        ctx.fillStyle = astroColor;
                        ctx.beginPath();
                        ctx.arc(x + 9, badgeY + 10, 3.5, 0, 2 * Math.PI);
                        ctx.fill();

                        ctx.fillStyle = "#ffffff";
                        ctx.fillText(labelText, x + 18, badgeY + 14);
                      }
                    }

                    // Draw MediaPipe Hands Skeletons with Unified Body-Mesh Integration
                    if (p.hands && p.hands.length > 0) {
                      const handConnections = [
                        [0, 1], [1, 2], [2, 3], [3, 4],
                        [0, 5], [5, 6], [6, 7], [7, 8],
                        [5, 9], [9, 10], [10, 11], [11, 12],
                        [9, 13], [13, 14], [14, 15], [15, 16],
                        [13, 17], [17, 18], [18, 19], [19, 20],
                        [0, 17]
                      ];
                      for (const hand of p.hands) {
                        const hlms = hand.landmarks;
                        const side = hand.side || "Right";
                        const rawGesture = hand.gesture || "NONE";
                        const isGesturing = rawGesture !== "NONE" && rawGesture !== "OPEN_HAND" && rawGesture !== "REACHING";
                        const hColor = isGesturing ? "#38bdf8" : (side === "Right" ? "rgba(80, 220, 230, 0.85)" : "rgba(180, 230, 80, 0.85)");

                        if (hlms && hlms.length >= 21) {
                          // Connect body wrist to hand wrist (Continuous Anatomical Skeletal Bridge)
                          if (p.pose && p.pose.landmarks) {
                            const poseWristIdx = side === "Right" ? 16 : 15;
                            const plm = p.pose.landmarks[poseWristIdx];
                            if (plm && (plm.visibility ?? 1) > 0.20) {
                              const pwPt = mapNormalizedPoint(plm.x, plm.y, `pose-${poseWristIdx}`, prevPoseLandmarksRef);
                              const hwPt = mapNormalizedPoint(hlms[0].x, hlms[0].y, `hand-${side}-0`, prevHandLandmarksRef);
                              ctx.strokeStyle = isGesturing ? "rgba(56, 189, 248, 0.95)" : "rgba(0, 230, 200, 0.70)";
                              ctx.lineWidth = isGesturing ? 3.0 : 2.0;
                              ctx.beginPath();
                              ctx.moveTo(pwPt.x, pwPt.y);
                              ctx.lineTo(hwPt.x, hwPt.y);
                              ctx.stroke();
                            }
                          }

                          // Draw Hand Skeleton Bones
                          ctx.strokeStyle = hColor;
                          ctx.lineWidth = isGesturing ? 2.8 : 2.0;
                          for (const [p1, p2] of handConnections) {
                            const ptA = mapNormalizedPoint(hlms[p1].x, hlms[p1].y, `hand-${side}-${p1}`, prevHandLandmarksRef);
                            const ptB = mapNormalizedPoint(hlms[p2].x, hlms[p2].y, `hand-${side}-${p2}`, prevHandLandmarksRef);
                            ctx.beginPath();
                            ctx.moveTo(ptA.x, ptA.y);
                            ctx.lineTo(ptB.x, ptB.y);
                            ctx.stroke();
                          }

                          // Draw Joint Nodes with Gesture-Specific Fingertip Reticles
                          for (let i = 0; i < hlms.length; i++) {
                            const isTip = [4, 8, 12, 16, 20].includes(i);
                            const hPt = mapNormalizedPoint(hlms[i].x, hlms[i].y, `hand-${side}-${i}`, prevHandLandmarksRef);
                            const hx = hPt.x;
                            const hy = hPt.y;

                            if (isTip) {
                              ctx.fillStyle = isGesturing ? "#ffffff" : hColor;
                              ctx.beginPath();
                              ctx.arc(hx, hy, isGesturing ? 4.5 : 3.5, 0, Math.PI * 2);
                              ctx.fill();

                              // Gesture-specific visual markers synchronized on the fingertips
                              if (rawGesture === "POINTING" && i === 8) {
                                ctx.strokeStyle = "#38bdf8";
                                ctx.lineWidth = 1.8;
                                ctx.beginPath();
                                ctx.arc(hx, hy, 9, 0, Math.PI * 2);
                                ctx.stroke();
                              } else if (rawGesture === "VICTORY" && (i === 8 || i === 12)) {
                                ctx.strokeStyle = "#38bdf8";
                                ctx.lineWidth = 1.8;
                                ctx.beginPath();
                                ctx.arc(hx, hy, 8, 0, Math.PI * 2);
                                ctx.stroke();
                              } else if ((rawGesture === "OK_SIGN" || rawGesture === "PINCH") && (i === 4 || i === 8)) {
                                ctx.strokeStyle = "#fbbf24";
                                ctx.lineWidth = 1.5;
                                ctx.beginPath();
                                ctx.arc(hx, hy, 7, 0, Math.PI * 2);
                                ctx.stroke();
                              }
                            } else {
                              ctx.fillStyle = hColor;
                              ctx.beginPath();
                              ctx.arc(hx, hy, 2.2, 0, Math.PI * 2);
                              ctx.fill();
                            }
                          }
                        }

                        // Real-time Synchronized Wrist Gesture Badge
                        if (hand.wrist) {
                          const wPt = mapNormalizedPoint(hand.wrist.x, hand.wrist.y, `hand-${side}-wrist-badge`, prevHandLandmarksRef);
                          const wx = wPt.x;
                          const wy = wPt.y;
                          const gDisplay = isGesturing
                            ? rawGesture.replace(/_/g, " ")
                            : (hand.is_grasping ? "GRASP" : (hand.is_pinching ? "PINCH" : "OPEN"));
                          const confTag = isGesturing && hand.gesture_confidence
                            ? ` [${Math.round(hand.gesture_confidence * 100)}%]`
                            : "";
                          const hText = `[${side[0]}] ${gDisplay}${confTag}`;
                          ctx.font = "700 10px Inter, sans-serif";
                          const htw = ctx.measureText(hText).width;
                          const hx = Math.max(4, wx - htw / 2);
                          const hy = Math.max(16, wy - 12);
                          ctx.fillStyle = "rgba(10, 15, 20, 0.88)";
                          ctx.fillRect(hx - 5, hy - 12, htw + 10, 16);
                          ctx.strokeStyle = isGesturing ? "#38bdf8" : hColor;
                          ctx.lineWidth = isGesturing ? 1.5 : 1.0;
                          ctx.strokeRect(hx - 5, hy - 12, htw + 10, 16);
                          ctx.fillStyle = isGesturing ? "#38bdf8" : hColor;
                          ctx.fillText(hText, hx, hy);
                        }
                      }
                    }

                    // Top Action Banner HUD
                    if (p.current_action && p.current_action.action && p.current_action.action !== "MONITORING" && p.current_action.action !== "IDLE") {
                      const actText = `CURRENT ACTION // ${p.current_action.action}`;
                      ctx.font = "700 12px Inter, sans-serif";
                      const aw = ctx.measureText(actText).width;
                      const ax = (cw - aw) / 2;
                      ctx.fillStyle = "rgba(10, 15, 20, 0.88)";
                      ctx.fillRect(ax - 12, 10, aw + 24, 26);
                      ctx.strokeStyle = "rgba(0, 230, 200, 0.8)";
                      ctx.lineWidth = 1;
                      ctx.strokeRect(ax - 12, 10, aw + 24, 26);
                      ctx.fillStyle = "#00e6c8";
                      ctx.fillText(actText, ax, 27);
                    }

                    // Real-time Gestures HUD Indicator (Hand & Body)
                    const activeGesture = p.primary_gesture || (p.gestures && p.gestures.primary_gesture);
                    if (activeGesture && activeGesture !== "NONE" && activeGesture !== "STATIONARY") {
                      const gText = `GESTURE // ${activeGesture.replace(/_/g, " ")}`;
                      ctx.font = "700 11px Inter, sans-serif";
                      const gw = ctx.measureText(gText).width;
                      const gx = cw - gw - 28;
                      const gy = 10;
                      ctx.fillStyle = "rgba(10, 15, 20, 0.88)";
                      ctx.fillRect(gx - 10, gy, gw + 20, 26);
                      ctx.strokeStyle = "rgba(56, 189, 248, 0.85)";
                      ctx.lineWidth = 1.5;
                      ctx.strokeRect(gx - 10, gy, gw + 20, 26);
                      ctx.fillStyle = "#38bdf8";
                      ctx.fillText(gText, gx, gy + 17);
                    }
                  }
                }
              }
            } catch (e) {
              console.error('RENDER ERROR', e);
            } finally {
              animId = requestAnimationFrame(renderOverlays);
            }
            };
            animId = requestAnimationFrame(renderOverlays);
          })
          .catch((err) => {
            if (isDisposed) return;
            console.warn("[Webcam Error]", err);
            setCameraLoading(false);
            setCameraError("Camera unavailable");
            fetch(`${API_BASE}/api/camera/clear_stream`, { method: "POST" }).catch(() => {});
            if (overlayCanvasRef.current) {
              const ctx = overlayCanvasRef.current.getContext("2d");
              if (ctx) ctx.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);
            }
          });
      }
    }
    return () => {
      isDisposed = true;
      if (uploadInterval) clearInterval(uploadInterval);
      if (animId) cancelAnimationFrame(animId);
      if (localStream) {
        localStream.getTracks().forEach((t) => t.stop());
      }
      if (!isCameraOn) {
        fetch(`${API_BASE}/api/camera/clear_stream`, { method: "POST" }).catch(() => {});
      }
    };
  }, [useBrowserWebcam, isCameraOn, selectedBrowserCamId, streamKey]);

  // Experiment state
  const [experimentState, setExperimentState] = useState({
    status: "IDLE",
    running: false,
    current_step: 1,
    total_steps: 5,
    progress_percentage: 0,
    steps: [
      { number: 1, label: "Pick up Object A", meta: "Pending", complete: false, current: true },
      { number: 2, label: "Place Object A on Object B", meta: "Pending", complete: false, current: false },
      { number: 3, label: "Pick up Object A again", meta: "Pending", complete: false, current: false },
      { number: 4, label: "Return Object A to Tray", meta: "Pending", complete: false, current: false },
      { number: 5, label: "Press the Complete Button", meta: "Pending", complete: false, current: false },
    ],
    next_step_label: "Pick up Object A"
  });

  // Perception state
  const [detectedObjects, setDetectedObjects] = useState([]);

  const [currentAction, setCurrentAction] = useState({
    action: "IDLE",
    label: "Ready / Monitoring",
    movement: "Stationary",
    posture: "Seated",
    narration: "Person is seated and monitoring the payload console.",
    actor: "person",
    hand: "Right hand",
    object: "Object A",
    color: "Red",
    confidence: 0.92
  });

  // Aethon Assistant state
  const [voiceMode, setVoiceMode] = useState(true);
  const [assistantDraft, setAssistantDraft] = useState("");
  const [isListening, setIsListening] = useState(false);
  const [continuousListening, setContinuousListening] = useState(true);
  const [voiceStatus, setVoiceStatus] = useState("idle"); // "idle" | "listening" | "processing" | "wake_detected"
  const [liveTranscript, setLiveTranscript] = useState("");
  // True only while speech is actually being heard, as opposed to the mic
  // merely being armed. Drives the mic pulse animation.
  const [isSpeechActive, setIsSpeechActive] = useState(false);
  // Which half of the wake-word state machine we are in.
  const [wakeState, setWakeState] = useState(WAKE_STATE.IDLE_LISTENING);
  // Transient, non-chat confirmation for camera actions.
  const [toast, setToast] = useState(null);
  const [audioRecording, setAudioRecording] = useState(false);
  const audioRecorderRef = useRef(null);
  const recognitionRef = useRef(null);
  const currentAudioRef = useRef(null);
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: "assistant",
      speaker: "Aethon",
      time: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
      text: "AETHON online. Ready for Payload Assembly experiment. Say 'Hey AETHON' or press the mic button to interact."
    }
  ]);
  const chatBottomRef = useRef(null);

  // Logs state
  const [logs, setLogs] = useState([]);
  const [sessions, setSessions] = useState([]);
  const [selectedSessionId, setSelectedSessionId] = useState("");
  const [logFilter, setLogFilter] = useState("ALL");

  // Date and Time telemetry in Header
  const [dateTimeStr, setDateTimeStr] = useState({ date: "Mon, 12 Jan 2026", time: "10:24 AM" });

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      const options = { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' };
      setDateTimeStr({
        date: now.toLocaleDateString('en-US', options),
        time: now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
      });
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  // Mouse Depth Effect tracking
  const containerRef = useRef(null);
  const handlePointerMove = useCallback((e) => {
    if (!containerRef.current) return;
    const x = e.clientX;
    const y = e.clientY;
    containerRef.current.style.setProperty("--mouse-x", `${x}px`);
    containerRef.current.style.setProperty("--mouse-y", `${y}px`);
  }, []);

  const [isAssistantActive, setIsAssistantActive] = useState(false);
  const activeTimerRef = useRef(null);
  const lastWakeTimeRef = useRef(0);

  // Local confirmation channel for hardware/camera buttons. Deliberately
  // separate from the conversation state so clicking Snapshot or Record does
  // not push chat bubbles, scroll the chat panel, or trigger TTS.
  const toastTimerRef = useRef(null);
  const showToast = useCallback((text, tone = "info") => {
    if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
    setToast({ id: Date.now(), text, tone });
    toastTimerRef.current = setTimeout(() => setToast(null), TOAST_DURATION_MS);
  }, []);

  useEffect(() => () => {
    if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
  }, []);

  // Short chime acknowledging the wake word, so the operator knows AETHON is
  // now capturing a command without having to look at the screen.
  const audioCtxRef = useRef(null);
  const playWakeChime = useCallback(() => {
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      if (!audioCtxRef.current) audioCtxRef.current = new AudioCtx();
      const ctx = audioCtxRef.current;
      if (ctx.state === "suspended") ctx.resume().catch(() => {});

      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.value = WAKE_CHIME_FREQ_HZ;
      gain.gain.setValueAtTime(WAKE_CHIME_GAIN, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + WAKE_CHIME_DURATION_MS / 1000);
      osc.connect(gain).connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + WAKE_CHIME_DURATION_MS / 1000);
    } catch (e) {
      /* chime is non-essential */
    }
  }, []);

  const WAKE_WORD_REGEX = /^\s*(?:hey|hi|hello|ok|okay|he|a|ey|ay|suno)?\s*(?:ae?thon|ae?than|ae?thane|athan|athlon|ethan|ethane|eaton|athena|atom|aidan|eden|aeon|item|anton|titan|python|than|then|hetan|tane|thanks?|ton|tan|aton)\b[,\s.:;!?]*/i;
  const REPEATED_WAKE_REGEX = /^\s*(?:ethane|ethan|ae?thon|ae?than|athan|tane)\s+(?:than|then|ethan|ethane|ae?thon|ae?than|athan|tane|tan)\b[,\s.:;!?]*/i;

  const normalizeAethonWakeWord = useCallback((text, strip = false) => {
    if (!text) return strip ? { hadWake: false, command: "" } : text;
    let cleaned = text.trim();
    let hadWake = false;

    if (REPEATED_WAKE_REGEX.test(cleaned)) {
      hadWake = true;
      if (strip) {
        cleaned = cleaned.replace(REPEATED_WAKE_REGEX, "").replace(/^[,\s.:;!?]+/, "").trim();
      } else {
        cleaned = cleaned.replace(REPEATED_WAKE_REGEX, "Hey AETHON, ");
      }
    } else {
      const match = cleaned.match(WAKE_WORD_REGEX);
      if (match) {
        hadWake = true;
        if (strip) {
          cleaned = cleaned.slice(match[0].length).replace(/^[,\s.:;!?]+/, "").trim();
        } else {
          cleaned = cleaned.replace(/^\s*(?:hey|hi|hello|he|a|ey|ay|suno)\s+(?:ae?thon|ae?than|ae?thane|athan|athlon|ethan|ethane|eaton|athena|atom|aidan|eden|aeon|item|anton|titan|python|than|then|tane|thanks?|ton|tan|aton)\b[,\s.:;!?]*/i, "Hey AETHON, ");
          cleaned = cleaned.replace(/^\s*okay?\s+(?:ae?thon|ae?than|ae?thane|athan|athlon|ethan|ethane|eaton|athena|atom|aidan|eden|aeon|item|anton|than|then|tane|thanks?|ton|tan|aton)\b[,\s.:;!?]*/i, "OK AETHON, ");
          cleaned = cleaned.replace(/^\s*(?:hetan)\b[,\s.:;!?]*/i, "Hey AETHON, ");
          cleaned = cleaned.replace(/^\s*(?:ae?thon|ae?than|ae?thane|athan|athlon|ethan|ethane|eaton|athena|atom|aidan|eden|aeon|item|anton|than|then|tane|aton)\b[,\s.:;!?]*/i, "AETHON, ");
        }
      }
    }

    if (strip) {
      return { hadWake, command: cleaned };
    }

    cleaned = cleaned.replace(/^Hey AETHON,\s*$/i, "Hey AETHON");
    cleaned = cleaned.replace(/^OK AETHON,\s*$/i, "OK AETHON");
    cleaned = cleaned.replace(/^AETHON,\s*$/i, "AETHON");
    return cleaned.trim();
  }, []);

  const stripWakeWord = useCallback((text) => {
    return normalizeAethonWakeWord(text, true);
  }, [normalizeAethonWakeWord]);

  // --- Natural Speech Synthesis (Browser Neural Voice) ---
  const voicesRef = useRef([]);
  const voiceModeRef = useRef(voiceMode);
  useEffect(() => {
    voiceModeRef.current = voiceMode;
  }, [voiceMode]);

  useEffect(() => {
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      const loadVoices = () => {
        try {
          const vList = window.speechSynthesis.getVoices() || [];
          if (vList.length > 0) {
            voicesRef.current = vList;
          }
        } catch (e) {}
      };
      loadVoices();
      if (window.speechSynthesis.onvoiceschanged !== undefined) {
        window.speechSynthesis.onvoiceschanged = loadVoices;
      }
    }
  }, []);

  // Select normal natural Indian English voice (en-IN: Neerja, Heera, Veena, Google English (India), etc.)
  const getPreferredVoice = useCallback(() => {
    let list = (voicesRef.current && voicesRef.current.length > 0)
      ? voicesRef.current
      : (typeof window !== "undefined" && "speechSynthesis" in window ? window.speechSynthesis.getVoices() : []);
    if (!list || list.length === 0) return null;

    // 1. Natural Indian English voices (en-IN: Neerja, Heera, Veena, Google English India, etc.)
    const indianEnglishFemale = list.find(v => {
      const l = (v.lang || "").toLowerCase().replace("_", "-");
      const n = (v.name || "").toLowerCase();
      const isIndian = l === "en-in" || l.startsWith("en-in") || n.includes("india");
      const isFemale = n.includes("neerja") || n.includes("heera") || n.includes("veena") ||
                       n.includes("kavya") || n.includes("swara") ||
                       (!n.includes("male") && !n.includes("prabhat") && !n.includes("ravi"));
      return isIndian && isFemale;
    });
    if (indianEnglishFemale) return indianEnglishFemale;

    // 2. Any Indian English voice (en-IN)
    const anyIndianEnglish = list.find(v => {
      const l = (v.lang || "").toLowerCase().replace("_", "-");
      const n = (v.name || "").toLowerCase();
      return l === "en-in" || l.startsWith("en-in") || n.includes("india") || n.includes("neerja") || n.includes("heera");
    });
    if (anyIndianEnglish) return anyIndianEnglish;

    // 3. Indian Hindi voice (hi-IN: Swara, Kalpana, Google हिन्दी)
    const indianHindi = list.find(v => {
      const l = (v.lang || "").toLowerCase().replace("_", "-");
      const n = (v.name || "").toLowerCase();
      return l === "hi-in" || l.startsWith("hi-in") || n.includes("hindi") || n.includes("swara");
    });
    if (indianHindi) return indianHindi;

    // 4. Fallback: Any clear English voice
    const fallbackEnglish = list.find(v => {
      const l = (v.lang || "").toLowerCase();
      return l.startsWith("en") && !v.name.toLowerCase().includes("male");
    });
    return fallbackEnglish || list[0];
  }, []);

  const lastSpokenIdRef = useRef(1); // 1 is initial welcome message
  const spokenIdsSetRef = useRef(new Set([1]));
  const lastSpokenTextRef = useRef("");
  const lastSpokenTimeRef = useRef(0);
  const activeSpeechGenerationRef = useRef(0);
  // True strictly while AETHON's own TTS is audible. Used to mute wake-word
  // recognition so the assistant cannot trigger itself.
  const isSpeakingRef = useRef(false);
  const speakingWatchdogRef = useRef(null);
  const recognitionActiveRef = useRef(false);

  const speakAssistantResponse = useCallback((text, msgId = null) => {
    if (!text || !voiceModeRef.current) return;
    const clean = text
      .replace(/[*_#`~[\]()]/g, " ")
      .replace(/https?:\/\/\S+/g, "")
      .replace(/\s+/g, " ")
      .trim();
    if (!clean) return;

    const now = Date.now();

    // Deduplication check: Avoid re-voicing the exact same message ID
    if (msgId !== null && msgId !== undefined) {
      if (spokenIdsSetRef.current.has(msgId)) {
        return;
      }
      spokenIdsSetRef.current.add(msgId);
      lastSpokenIdRef.current = msgId;
      if (spokenIdsSetRef.current.size > 100) {
        const arr = Array.from(spokenIdsSetRef.current);
        spokenIdsSetRef.current = new Set(arr.slice(-50));
      }
    }

    // Temporal identical utterance suppression (prevents repeating within 4s)
    if (
      clean.toLowerCase() === lastSpokenTextRef.current.toLowerCase() &&
      (now - lastSpokenTimeRef.current) < 4000
    ) {
      return;
    }

    lastSpokenTextRef.current = clean;
    lastSpokenTimeRef.current = now;
    lastWakeTimeRef.current = now;

    // Advance speech generation token: completely invalidates any pending promises/handlers
    const currentGen = ++activeSpeechGenerationRef.current;

    // Stop and cleanly release any previous audio element
    if (currentAudioRef.current) {
      try {
        currentAudioRef.current.pause();
        currentAudioRef.current.currentTime = 0;
        currentAudioRef.current.onplay = null;
        currentAudioRef.current.onended = null;
        currentAudioRef.current.onerror = null;
      } catch (e) {}
      currentAudioRef.current = null;
    }
    // Hard-cancel any residual browser speech synthesis so no second voice can ever speak
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      try { window.speechSynthesis.cancel(); } catch (e) {}
    }

    if (speakingWatchdogRef.current) {
      clearTimeout(speakingWatchdogRef.current);
      speakingWatchdogRef.current = null;
    }

    // Safety watchdog: guarantees isSpeakingRef can never be stuck true if an audio event is missed
    const maxSpeechMs = Math.max(3500, Math.min(clean.length * 80 + 3500, 22000));
    speakingWatchdogRef.current = setTimeout(() => {
      if (activeSpeechGenerationRef.current !== currentGen) return;
      setIsAssistantActive(false);
      isSpeakingRef.current = false;
      setVoiceStatus(continuousListeningRef.current ? "listening" : "idle");
    }, maxSpeechMs);

    // Sole voice: High-Fidelity Indian English Neural Speech Synthesis (/api/voice/tts)
    try {
      const ttsUrl = `${API_BASE}/api/voice/tts?text=${encodeURIComponent(clean)}`;
      const audio = new Audio(ttsUrl);
      currentAudioRef.current = audio;

      audio.onplay = () => {
        if (activeSpeechGenerationRef.current !== currentGen) {
          try { audio.pause(); } catch (e) {}
          return;
        }
        setIsAssistantActive(true);
        isSpeakingRef.current = true;
        setVoiceStatus("speaking");
      };

      audio.onended = () => {
        if (activeSpeechGenerationRef.current !== currentGen) return;
        if (speakingWatchdogRef.current) {
          clearTimeout(speakingWatchdogRef.current);
          speakingWatchdogRef.current = null;
        }
        setIsAssistantActive(false);
        isSpeakingRef.current = false;
        setVoiceStatus(continuousListeningRef.current ? "listening" : "idle");
        if (currentAudioRef.current === audio) {
          currentAudioRef.current = null;
        }
      };

      const speakFallback = () => {
        if (typeof window !== "undefined" && "speechSynthesis" in window) {
          try {
            window.speechSynthesis.cancel();
            const utt = new SpeechSynthesisUtterance(clean);
            utt.rate = 1.0;
            utt.onstart = () => {
              setIsAssistantActive(true);
              isSpeakingRef.current = true;
              setVoiceStatus("speaking");
            };
            utt.onend = () => {
              setIsAssistantActive(false);
              isSpeakingRef.current = false;
              setVoiceStatus(continuousListeningRef.current ? "listening" : "idle");
            };
            utt.onerror = () => {
              setIsAssistantActive(false);
              isSpeakingRef.current = false;
              setVoiceStatus(continuousListeningRef.current ? "listening" : "idle");
            };
            window.speechSynthesis.speak(utt);
          } catch (e) {}
        }
      };

      audio.onerror = (e) => {
        if (activeSpeechGenerationRef.current !== currentGen) return;
        if (speakingWatchdogRef.current) {
          clearTimeout(speakingWatchdogRef.current);
          speakingWatchdogRef.current = null;
        }
        if (currentAudioRef.current === audio) {
          currentAudioRef.current = null;
        }
        speakFallback();
      };

      const playPromise = audio.play();
      if (playPromise !== undefined) {
        playPromise.catch((err) => {
          if (activeSpeechGenerationRef.current !== currentGen || err?.name === "AbortError") {
            return;
          }
          if (speakingWatchdogRef.current) {
            clearTimeout(speakingWatchdogRef.current);
            speakingWatchdogRef.current = null;
          }
          if (currentAudioRef.current === audio) {
            currentAudioRef.current = null;
          }
          speakFallback();
        });
      }
    } catch (e) {
      if (activeSpeechGenerationRef.current === currentGen) {
        if (speakingWatchdogRef.current) {
          clearTimeout(speakingWatchdogRef.current);
          speakingWatchdogRef.current = null;
        }
        setIsAssistantActive(false);
        isSpeakingRef.current = false;
        setVoiceStatus(continuousListeningRef.current ? "listening" : "idle");
      }
    }
  }, []);

  const speakAssistantRef = useRef(speakAssistantResponse);
  useEffect(() => {
    speakAssistantRef.current = speakAssistantResponse;
  }, [speakAssistantResponse]);

  // WebSocket connection & live telemetry
  useEffect(() => {
    let ws = null;
    let reconnectTimeout = null;

    const connect = () => {
      try {
        ws = new WebSocket(WS_URL);
        wsRef.current = ws;

        ws.onopen = () => {
          wsRef.current = ws;
          console.log("[AETHON WS] Connected to backend telemetry");
        };

        ws.onmessage = (event) => {
          try {
            const data = jsonParseSafe(event.data);
            if (!data) return;

            if (data.type === "PERCEPTION") {
              if (data.camera) {
                if (data.camera.latency_ms !== undefined) setLatencyMs(data.camera.latency_ms);
                if (data.camera.fps !== undefined) setFps(data.camera.fps || 60);
              }
              if (data.perception) {
                latestPerceptionRef.current = data.perception;
                if (data.perception.current_action) {
                  setCurrentAction(data.perception.current_action);
                }
                if (data.perception.objects !== undefined) {
                  if (data.perception.objects.length > 0) {
                    const mapped = data.perception.objects
                      .filter((obj) => obj.track_status !== "CANDIDATE")
                      .map(mapPerceptionObject);
                    setDetectedObjects(mapped);
                  } else {
                    setDetectedObjects([]);
                  }
                }
              }
            } else if (data.type === "INIT" || data.type === "TELEMETRY") {
              if (data.camera) {
                if (data.camera.fps !== undefined) setFps(data.camera.fps || 60);
                if (data.camera.latency_ms !== undefined) setLatencyMs(data.camera.latency_ms);
                if (data.camera.recording !== undefined) setIsRecording(data.camera.recording);
                if (data.camera.devices) setCameras(data.camera.devices);
                if (data.camera.device_index !== undefined) setSelectedCamera(data.camera.device_index);
                if (data.camera.mirror !== undefined) setMirrorFeed(data.camera.mirror);
              }

              if (data.experiment) {
                setExperimentState(data.experiment);
              }

              if (data.perception) {
                latestPerceptionRef.current = data.perception;
                if (data.perception.current_action) {
                  setCurrentAction(data.perception.current_action);
                }
                if (data.perception.objects !== undefined) {
                  if (data.perception.objects.length > 0) {
                    const mapped = data.perception.objects
                      .filter((obj) => obj.track_status !== "CANDIDATE")
                      .map(mapPerceptionObject);
                    setDetectedObjects(mapped);
                  } else {
                    setDetectedObjects([]);
                  }
                }
              }

              if (data.voice && data.voice.is_active) {
                setIsAssistantActive(true);
                if (activeTimerRef.current) clearTimeout(activeTimerRef.current);
                activeTimerRef.current = setTimeout(() => {
                  setIsAssistantActive(false);
                }, 2800);
              }

              if (data.conversation && data.conversation.length > 0) {
                setMessages((prev) => {
                  if (prev.length === data.conversation.length) {
                    const prevLast = prev[prev.length - 1];
                    const newLast = data.conversation[data.conversation.length - 1];
                    if (prevLast?.id === newLast?.id && prevLast?.text === newLast?.text) {
                      return prev;
                    }
                  }
                  return data.conversation;
                });

              }

              if (data.logs) {
                setLogs(data.logs);
              }
            } else if (data.type === "COMMAND_RESPONSE") {
              if (data.data && data.data.conversation) {
                setMessages(data.data.conversation);
              }
            }
          } catch (err) {
            console.error("[WS Parse Error]", err);
          }
        };

        ws.onclose = () => {
          if (wsRef.current === ws) wsRef.current = null;
          reconnectTimeout = setTimeout(connect, 2000);
        };

        ws.onerror = () => {
          ws.close();
        };
      } catch (e) {
        reconnectTimeout = setTimeout(connect, 2500);
      }
    };

    connect();

    // Initial fetch fallback
    fetch(`${API_BASE}/api/camera/devices`)
      .then((r) => r.json())
      .then((d) => {
        if (d.devices) setCameras(d.devices);
        if (d.current !== undefined) setSelectedCamera(d.current);
      })
      .catch(() => {});

    fetch(`${API_BASE}/api/sessions`)
      .then((r) => r.json())
      .then((d) => {
        if (d.sessions) setSessions(d.sessions);
      })
      .catch(() => {});

    return () => {
      if (wsRef.current === ws) wsRef.current = null;
      if (ws) ws.close();
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
    };
  }, []);

  // Auto scroll chat to bottom only when new messages are actually added
  const prevMsgCountRef = useRef(messages.length);
  useEffect(() => {
    if (messages.length > prevMsgCountRef.current) {
      prevMsgCountRef.current = messages.length;
      if (chatBottomRef.current) {
        chatBottomRef.current.scrollIntoView({ behavior: "smooth" });
      }
    }
  }, [messages.length]);

  // Command sender
  const sendCommand = useCallback(async (text, customDisplay = null) => {
    if (!text || !text.trim()) return;
    lastWakeTimeRef.current = Date.now();
    const cleanText = text.trim();
    const displayText = customDisplay || normalizeAethonWakeWord(cleanText);
    setAssistantDraft("");
    setLiveTranscript("");

    // Optimistic user bubble with clean AETHON spelling
    const now = new Date();
    const timeStr = now.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
    setMessages((prev) => [
      ...prev,
      { id: prev.length + 1, role: "user", speaker: "You", time: timeStr, text: displayText }
    ]);

    try {
      const res = await fetch(`${API_BASE}/api/voice/command`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: cleanText })
      });
      const data = await res.json();
      if (data.conversation) {
        setMessages(data.conversation);
      }
      if (data.experiment_state) {
        setExperimentState(data.experiment_state);
      }
      if (data.action) {
        if (data.action === "SHOW_LOGS") setCurrentView("Logs");
        else if (data.action === "SHOW_MONITOR") setCurrentView("Monitor");
        else if (data.action === "SHOW_EXPERIMENT") setCurrentView("Experiment");
        else if (data.action === "TAKE_SNAPSHOT") handleSnapshot();
        else if (data.action === "RECORD_START") handleToggleRecording(true);
        else if (data.action === "RECORD_STOP") handleToggleRecording(false);
      }
      if (data.response) {
        const lastMsg = data.conversation && data.conversation[data.conversation.length - 1];
        const msgId = lastMsg?.role === "assistant" ? lastMsg.id : null;
        speakAssistantResponse(data.response, msgId);
      }
    } catch (e) {
      console.error("[Command Send Error]", e);
    }
  }, [speakAssistantResponse]);

  // Fallback WAV audio uploader
  const sendAudioBlob = useCallback(async (blob) => {
    try {
      setVoiceStatus("processing");
      const formData = new FormData();
      formData.append("audio", blob, "voice_input.wav");
      const res = await fetch(`${API_BASE}/api/voice/transcribe`, {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      if (data.status === "success" && data.assistant_response) {
        if (data.assistant_response.conversation) {
          setMessages(data.assistant_response.conversation);
        }
        if (data.assistant_response.response) {
          const lastMsg = data.assistant_response.conversation && data.assistant_response.conversation[data.assistant_response.conversation.length - 1];
          const msgId = lastMsg?.role === "assistant" ? lastMsg.id : null;
          speakAssistantResponse(data.assistant_response.response, msgId);
        }
      } else {
        const errMsg = "Audio received, but voice could not be transcribed clearly. Please speak closer to your microphone.";
        setMessages((prev) => [
          ...prev,
          {
            id: prev.length + 1,
            role: "assistant",
            speaker: "Aethon",
            time: new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" }),
            text: errMsg
          }
        ]);
        speakAssistantResponse(errMsg);
      }
    } catch (err) {
      console.error("[Audio Upload Error]", err);
    } finally {
      setVoiceStatus(continuousListening ? "listening" : "idle");
      setIsListening(continuousListening);
    }
  }, [continuousListening, speakAssistantResponse]);



  const continuousListeningRef = useRef(continuousListening);
  useEffect(() => {
    continuousListeningRef.current = continuousListening;
  }, [continuousListening]);

  const sendCommandRef = useRef(sendCommand);
  useEffect(() => {
    sendCommandRef.current = sendCommand;
  }, [sendCommand]);

  const stripWakeWordRef = useRef(stripWakeWord);
  useEffect(() => {
    stripWakeWordRef.current = stripWakeWord;
  }, [stripWakeWord]);

  const normalizeAethonWakeWordRef = useRef(normalizeAethonWakeWord);
  useEffect(() => {
    normalizeAethonWakeWordRef.current = normalizeAethonWakeWord;
  }, [normalizeAethonWakeWord]);

  const restartTimerRef = useRef(null);

  // --- Wake-word session state ---
  const wakeStateRef = useRef(WAKE_STATE.IDLE_LISTENING);
  const commandBufferRef = useRef("");
  const commandSilenceTimerRef = useRef(null);
  const commandTimeoutRef = useRef(null);
  const speechActiveTimerRef = useRef(null);
  const clearCommandTimers = useCallback(() => {
    if (commandSilenceTimerRef.current) {
      clearTimeout(commandSilenceTimerRef.current);
      commandSilenceTimerRef.current = null;
    }
    if (commandTimeoutRef.current) {
      clearTimeout(commandTimeoutRef.current);
      commandTimeoutRef.current = null;
    }
  }, []);

  const enterIdleListening = useCallback(() => {
    clearCommandTimers();
    commandBufferRef.current = "";
    wakeStateRef.current = WAKE_STATE.IDLE_LISTENING;
    setWakeState(WAKE_STATE.IDLE_LISTENING);
    setLiveTranscript("");
  }, [clearCommandTimers]);

  // Marks speech as active and schedules it to lapse after a short silence.
  const markSpeechActive = useCallback(() => {
    setIsSpeechActive(true);
    if (speechActiveTimerRef.current) clearTimeout(speechActiveTimerRef.current);
    speechActiveTimerRef.current = setTimeout(() => {
      setIsSpeechActive(false);
    }, SPEECH_ACTIVE_DEBOUNCE_MS);
  }, []);

  const markSpeechInactive = useCallback(() => {
    if (speechActiveTimerRef.current) clearTimeout(speechActiveTimerRef.current);
    speechActiveTimerRef.current = setTimeout(() => {
      setIsSpeechActive(false);
    }, SPEECH_ACTIVE_DEBOUNCE_MS);
  }, []);

  // Finalise a wake-word session: send the captured command exactly once,
  // suspend transcription, and return to waiting for the next "Hey AETHON".
  const finalizeCommand = useCallback(() => {
    clearCommandTimers();
    const captured = commandBufferRef.current.trim();
    commandBufferRef.current = "";

    // If only the wake word was spoken with no command text:
    const toSend = captured || "AETHON standing by";
    const display = normalizeAethonWakeWordRef.current
      ? normalizeAethonWakeWordRef.current(`aethon ${captured}`.trim())
      : `AETHON, ${captured}`;

    setVoiceStatus("processing");
    if (sendCommandRef.current) {
      sendCommandRef.current(toSend, display);
    }

    wakeStateRef.current = WAKE_STATE.IDLE_LISTENING;
    setWakeState(WAKE_STATE.IDLE_LISTENING);
    setLiveTranscript("");
    setIsSpeechActive(false);
  }, [clearCommandTimers]);

  const finalizeCommandRef = useRef(finalizeCommand);
  useEffect(() => {
    finalizeCommandRef.current = finalizeCommand;
  }, [finalizeCommand]);

  const scheduleCommandFinalize = useCallback(() => {
    if (commandSilenceTimerRef.current) clearTimeout(commandSilenceTimerRef.current);
    commandSilenceTimerRef.current = setTimeout(() => {
      if (wakeStateRef.current === WAKE_STATE.CAPTURING_COMMAND) {
        finalizeCommandRef.current();
      }
    }, COMMAND_END_SILENCE_MS);
  }, []);

  const beginCommandCapture = useCallback((initialText = "") => {
    wakeStateRef.current = WAKE_STATE.CAPTURING_COMMAND;
    setWakeState(WAKE_STATE.CAPTURING_COMMAND);
    commandBufferRef.current = initialText.trim();
    lastWakeTimeRef.current = Date.now();
    setVoiceStatus("wake_detected");
    playWakeChime();
    markSpeechActive();

    if (commandTimeoutRef.current) clearTimeout(commandTimeoutRef.current);
    commandTimeoutRef.current = setTimeout(() => {
      if (wakeStateRef.current === WAKE_STATE.CAPTURING_COMMAND) {
        finalizeCommandRef.current();
      }
    }, COMMAND_CAPTURE_TIMEOUT_MS);

    scheduleCommandFinalize();
  }, [playWakeChime, markSpeechActive, scheduleCommandFinalize]);

  const beginCommandCaptureRef = useRef(beginCommandCapture);
  useEffect(() => {
    beginCommandCaptureRef.current = beginCommandCapture;
  }, [beginCommandCapture]);

  const scheduleCommandFinalizeRef = useRef(scheduleCommandFinalize);
  useEffect(() => {
    scheduleCommandFinalizeRef.current = scheduleCommandFinalize;
  }, [scheduleCommandFinalize]);

  const markSpeechActiveRef = useRef(markSpeechActive);
  useEffect(() => {
    markSpeechActiveRef.current = markSpeechActive;
  }, [markSpeechActive]);

  const markSpeechInactiveRef = useRef(markSpeechInactive);
  useEffect(() => {
    markSpeechInactiveRef.current = markSpeechInactive;
  }, [markSpeechInactive]);

  const enterIdleListeningRef = useRef(enterIdleListening);
  useEffect(() => {
    enterIdleListeningRef.current = enterIdleListening;
  }, [enterIdleListening]);

  const startContinuousListeningRef = useRef(null);

  // Continuous Listening – always-on voice recognition
  const startContinuousListening = useCallback(() => {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRec) {
      console.warn("[Voice] Web SpeechRecognition unavailable in this browser; Push-to-Talk or hardware mic can be used.");
      return;
    }

    // Stop any existing instance and mark it superseded so its onend doesn't trigger a duplicate restart
    if (recognitionRef.current) {
      try {
        recognitionRef.current._superseded = true;
        recognitionRef.current.abort();
      } catch (e) {}
    }

    const recognition = new SpeechRec();
    recognition.lang = "en-US";
    recognition.interimResults = true;
    recognition.continuous = true;
    recognition.maxAlternatives = 1;
    recognition._superseded = false;
    recognitionRef.current = recognition;

    recognition.onstart = () => {
      recognitionActiveRef.current = true;
      setIsListening(true);
      setVoiceStatus(wakeStateRef.current === WAKE_STATE.CAPTURING_COMMAND ? "wake_detected" : "listening");
    };

    // Speech boundary events drive the "actually hearing you" indicator.
    recognition.onspeechstart = () => {
      if (isSpeakingRef.current) return;
      if (markSpeechActiveRef.current) markSpeechActiveRef.current();
    };
    recognition.onspeechend = () => {
      if (markSpeechInactiveRef.current) markSpeechInactiveRef.current();
      if (wakeStateRef.current === WAKE_STATE.CAPTURING_COMMAND && scheduleCommandFinalizeRef.current) {
        scheduleCommandFinalizeRef.current();
      }
    };

    // Wake-word state machine.
    //
    // IDLE_LISTENING     transcripts are matched against the wake word
    //                    locally; nothing is sent, no bubble, no TTS.
    // CAPTURING_COMMAND  everything after the wake word accumulates until a
    //                    short silence finalises and sends it exactly once.
    recognition.onresult = (event) => {
      // Ignore the assistant's own speech echoing back through the mic.
      // Uses isSpeakingRef (guarded by speakingWatchdogRef) instead of buggy window.speechSynthesis.speaking
      if (isSpeakingRef.current) {
        setLiveTranscript("");
        return;
      }

      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;

        if (event.results[i].isFinal) {
          if (markSpeechActiveRef.current) markSpeechActiveRef.current();

          if (wakeStateRef.current === WAKE_STATE.IDLE_LISTENING) {
            // Gate: only a local wake-word match may start a session.
            const { hadWake, command } = stripWakeWordRef.current
              ? stripWakeWordRef.current(transcript)
              : { hadWake: false, command: "" };
            if (!hadWake) {
              setLiveTranscript("");
              continue;
            }
            // Anything already said after the wake word seeds the command.
            if (beginCommandCaptureRef.current) beginCommandCaptureRef.current(command);
          } else {
            // Mid-session: strip a repeated wake word, keep the rest.
            const { command } = stripWakeWordRef.current
              ? stripWakeWordRef.current(transcript)
              : { command: transcript };
            const addition = (command || transcript).trim();
            if (addition) {
              commandBufferRef.current = `${commandBufferRef.current} ${addition}`.trim();
            }
            if (scheduleCommandFinalizeRef.current) scheduleCommandFinalizeRef.current();
          }
        } else {
          interim += transcript;
        }
      }

      if (!interim) return;
      if (markSpeechActiveRef.current) markSpeechActiveRef.current();

      if (wakeStateRef.current === WAKE_STATE.CAPTURING_COMMAND) {
        // Show what is being captured, and keep the silence timer alive.
        const preview = `${commandBufferRef.current} ${interim}`.trim();
        setLiveTranscript(preview);
        if (scheduleCommandFinalizeRef.current) scheduleCommandFinalizeRef.current();
        return;
      }

      // Idle: as soon as the wake word is heard in the interim stream,
      // enter CAPTURING_COMMAND (chime + pulse) without sending anything.
      // The command text itself is only committed on a final result so the
      // same utterance is not written into the buffer twice.
      const { hadWake, command } = stripWakeWordRef.current
        ? stripWakeWordRef.current(interim)
        : { hadWake: false, command: "" };
      if (hadWake) {
        if (beginCommandCaptureRef.current) beginCommandCaptureRef.current("");
        setLiveTranscript(
          normalizeAethonWakeWordRef.current
            ? normalizeAethonWakeWordRef.current(interim)
            : (command || interim)
        );
      } else {
        setLiveTranscript("");
      }
    };

    recognition.onerror = (e) => {
      recognitionActiveRef.current = false;
      if (e.error === "aborted" || e.error === "no-speech") return;
      console.warn("[Continuous Voice Error]", e.error);

      // Auto re-arm on transient errors rather than permanently killing the mic
      if (continuousListeningRef.current) {
        if (restartTimerRef.current) clearTimeout(restartTimerRef.current);
        const retryDelay = e.error === "service-not-allowed" ? 1200 : 500;
        restartTimerRef.current = setTimeout(() => {
          if (continuousListeningRef.current && startContinuousListeningRef.current) {
            startContinuousListeningRef.current();
          }
        }, retryDelay);
      }
    };

    recognition.onend = () => {
      recognitionActiveRef.current = false;
      // If superseded by a newer recognition instance, don't trigger a duplicate restart
      if (recognition._superseded) return;
      setIsSpeechActive(false);
      if (continuousListeningRef.current) {
        // Re-arm for the next "Hey AETHON". Create a fresh instance each time to avoid InvalidStateError.
        if (restartTimerRef.current) clearTimeout(restartTimerRef.current);
        restartTimerRef.current = setTimeout(() => {
          try {
            if (continuousListeningRef.current && startContinuousListeningRef.current) {
              startContinuousListeningRef.current();
            }
          } catch (e) {}
        }, 350);
      } else {
        setIsListening(false);
        setVoiceStatus("idle");
        if (enterIdleListeningRef.current) enterIdleListeningRef.current();
      }
    };

    try {
      recognition.start();
      recognitionActiveRef.current = true;
    } catch (e) {
      console.warn("[Continuous Voice] recognition.start() error, retrying:", e);
      recognitionActiveRef.current = false;
      if (continuousListeningRef.current) {
        if (restartTimerRef.current) clearTimeout(restartTimerRef.current);
        restartTimerRef.current = setTimeout(() => {
          if (continuousListeningRef.current && startContinuousListeningRef.current) {
            startContinuousListeningRef.current();
          }
        }, 500);
      }
    }
  }, []);

  useEffect(() => {
    startContinuousListeningRef.current = startContinuousListening;
  }, [startContinuousListening]);

  const stopContinuousListening = useCallback(() => {
    if (restartTimerRef.current) {
      clearTimeout(restartTimerRef.current);
      restartTimerRef.current = null;
    }
    if (speechActiveTimerRef.current) {
      clearTimeout(speechActiveTimerRef.current);
      speechActiveTimerRef.current = null;
    }
    if (recognitionRef.current) {
      try {
        recognitionRef.current._superseded = true;
        recognitionRef.current.abort();
      } catch (e) {}
      recognitionRef.current = null;
    }
    setIsListening(false);
    setVoiceStatus("idle");
    setLiveTranscript("");
    setIsSpeechActive(false);
    enterIdleListening();
  }, [enterIdleListening]);

  const stopContinuousListeningRef = useRef(stopContinuousListening);
  useEffect(() => {
    stopContinuousListeningRef.current = stopContinuousListening;
  }, [stopContinuousListening]);

  // Effect: start/stop continuous listening when toggle changes
  useEffect(() => {
    if (continuousListening) {
      startContinuousListening();
    } else {
      stopContinuousListening();
    }
    return () => {
      stopContinuousListening();
    };
  }, [continuousListening, startContinuousListening, stopContinuousListening]);

  // Liveness watchdog supervisor: guarantees continuous listening stays alive even if Chrome drops the session
  useEffect(() => {
    if (!continuousListening) return;
    const supervisor = setInterval(() => {
      if (
        continuousListeningRef.current &&
        !recognitionActiveRef.current &&
        !isSpeakingRef.current
      ) {
        console.log("[Continuous Voice Watchdog] Recognition dropped; reviving session...");
        if (startContinuousListeningRef.current) {
          startContinuousListeningRef.current();
        }
      }
    }, 1800);
    return () => clearInterval(supervisor);
  }, [continuousListening]);

  // Push-to-Talk Speech Recognition (with WAV recording fallback)
  const handleMicClick = async () => {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;

    // If currently recording via fallback AudioRecorder, stop and upload
    if (audioRecording && audioRecorderRef.current) {
      try {
        const blob = audioRecorderRef.current.stop();
        setAudioRecording(false);
        setIsListening(false);
        setVoiceStatus("processing");
        await sendAudioBlob(blob);
      } catch (e) {
        setAudioRecording(false);
        setIsListening(false);
        setVoiceStatus("idle");
      }
      return;
    }

    if (SpeechRec) {
      // Toggle the armed state. Arming only starts IDLE_LISTENING — nothing
      // is sent anywhere until the wake word is matched locally.
      const newState = !continuousListening;
      setContinuousListening(newState);
      enterIdleListening();

      if (newState) {
        setVoiceStatus("listening");
        showToast("Voice armed — say \"Hey AETHON\"", "info");
      } else {
        setIsListening(false);
        setVoiceStatus("idle");
        setIsSpeechActive(false);
      }
    } else {
      // Fallback manual Push-To-Talk
      try {
        const rec = new AudioRecorder();
        await rec.start();
        audioRecorderRef.current = rec;
        setAudioRecording(true);
        setVoiceStatus("listening");
      } catch (recErr) {
        console.error("[Fallback Mic Error]", recErr);
        alert("Microphone access could not be initialized. Please check browser permissions.");
        setIsListening(false);
      }
    }
  };

  // Camera action handlers.
  //
  // These confirm via the local toast, not sendCommand(): a camera button is
  // a hardware action, not a voice/chat interaction, so it must not push chat
  // bubbles, scroll the conversation panel, or speak through TTS.
  const handleToggleCamera = useCallback(() => {
    setIsCameraOn((prev) => {
      const next = !prev;
      if (next) {
        setStreamKey(Date.now());
        showToast("Camera feed enabled", "info");
      } else {
        showToast("Camera feed turned off", "warn");
      }
      return next;
    });
  }, [showToast]);

  const handleSnapshot = async () => {
    if (!isCameraOn) {
      showToast("Camera feed is turned off", "warn");
      return;
    }
    try {
      const res = await fetch(`${API_BASE}/api/camera/snapshot`, { method: "POST" });
      const data = await res.json();
      showToast(data.filename ? `Snapshot saved · ${data.filename}` : "Snapshot saved", "success");
    } catch (e) {
      showToast("Snapshot failed — check the camera feed", "error");
    }
  };

  const handleRecordToggle = async () => {
    if (!isCameraOn && !isRecording) {
      showToast("Cannot record while camera feed is turned off", "warn");
      return;
    }
    const wasRecording = isRecording;
    try {
      const endpoint = wasRecording ? "/api/camera/record/stop" : "/api/camera/record/start";
      await fetch(`${API_BASE}${endpoint}`, { method: "POST" });
      setIsRecording(!wasRecording);
      showToast(wasRecording ? "Recording stopped and saved" : "Recording started", "success");
    } catch (e) {
      setIsRecording(!wasRecording);
      showToast(wasRecording ? "Failed to stop recording" : "Failed to start recording", "error");
    }
  };

  const handleToggleMirror = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/camera/mirror/toggle`, { method: "POST" });
      const data = await res.json();
      if (data.mirror !== undefined) {
        setMirrorFeed(data.mirror);
      } else {
        setMirrorFeed((prev) => !prev);
      }
      setStreamKey(Date.now());
    } catch (e) {
      setMirrorFeed((prev) => !prev);
    }
  };

  const handleSelectBrowserCamera = (deviceId) => {
    try {
      localStorage.setItem("aethon_selected_camera_id", deviceId);
    } catch (e) {}
    setSelectedBrowserCamId(deviceId);
    setShowCameraMenu(false);
    setUseBrowserWebcam(true);
    setIsCameraOn(true);
    setCameraError(null);
    setCameraLoading(true);
    setStreamKey(Date.now());
  };

  const handleSelectCamera = async (index) => {
    setSelectedCamera(index);
    setShowCameraMenu(false);
    setUseBrowserWebcam(false);
    setIsCameraOn(true);
    try {
      const res = await fetch(`${API_BASE}/api/camera/select`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ index })
      });
      const data = await res.json();
      if (data && data.cameras) {
        setCameras(data.cameras);
      }
      if (data && data.camera_index !== undefined) {
        setSelectedCamera(data.camera_index);
      }
    } catch (e) {
      console.error("[Camera Switch Error]", e);
    } finally {
      setStreamKey(Date.now());
    }
  };

  const handleFullscreen = () => {
    if (!cameraStageRef.current) return;
    if (!document.fullscreenElement) {
      cameraStageRef.current.requestFullscreen().catch(() => {});
    } else {
      document.exitFullscreen().catch(() => {});
    }
  };

  // Experiment control buttons
  const handleStart = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/experiment/start`, { method: "POST" });
      const data = await res.json();
      if (data.state) setExperimentState(data.state);
      if (data.message) speakAssistantResponse(data.message);
    } catch (e) {}
  };

  const handlePause = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/experiment/pause`, { method: "POST" });
      const data = await res.json();
      if (data.state) setExperimentState(data.state);
      if (data.message) speakAssistantResponse(data.message);
    } catch (e) {}
  };

  const handleResume = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/experiment/resume`, { method: "POST" });
      const data = await res.json();
      if (data.state) setExperimentState(data.state);
      if (data.message) speakAssistantResponse(data.message);
    } catch (e) {}
  };

  const handleReset = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/experiment/reset`, { method: "POST" });
      const data = await res.json();
      if (data.state) setExperimentState(data.state);
      if (data.message) speakAssistantResponse(data.message);
    } catch (e) {}
  };

  const handleStop = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/experiment/stop`, { method: "POST" });
      const data = await res.json();
      if (data.state) setExperimentState(data.state);
      if (data.message) speakAssistantResponse(data.message);
    } catch (e) {}
  };

  // Demo / Test Mode simulated action injector
  const handleSimulateAction = async (action, object, target = null) => {
    try {
      const res = await fetch(`${API_BASE}/api/experiment/simulate-action`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ event: action, object, target, confidence: 0.95 })
      });
      const data = await res.json();
      if (data.state) setExperimentState(data.state);
    } catch (e) {}
  };

  const running = experimentState.running;
  const activeStep = experimentState.current_step;
  const stepProgress = `${experimentState.progress_percentage || 0}%`;

  const cameraDisplayName = useMemo(() => {
    if (useBrowserWebcam) {
      if (selectedBrowserCamId) {
        const found = browserWebcams.find((b) => b.deviceId === selectedBrowserCamId);
        if (found && found.label) return found.label;
      }
      if (browserWebcams.length > 0) {
        const chosen = browserWebcams[0];
        if (chosen && chosen.label) return chosen.label;
      }
      return "Webcam Stream";
    }
    const found = cameras.find((c) => c.index === selectedCamera);
    return found ? found.name : `Camera ${selectedCamera}`;
  }, [cameras, selectedCamera, useBrowserWebcam, selectedBrowserCamId, browserWebcams]);

  return (
    <>
      {isIntroActive && (
        <OpeningTransition onComplete={() => setIsIntroActive(false)} />
      )}
      <main
        className="aethon-app"
        ref={containerRef}
        onPointerMove={handlePointerMove}
      >
        <img className="space-backdrop" src="/space-bg.png" alt="" aria-hidden="true" />
        <div className="space-vignette" aria-hidden="true" />
        <div className="shooting-stars" aria-hidden="true">
          {Array.from({ length: 10 }, (_, index) => (
            <span key={index} className={`shooting-star star-${index + 1}`} />
          ))}
        </div>
        <div className="noise-layer" aria-hidden="true" />
        <div className="mouse-depth-field" aria-hidden="true" />

        <div className="console-shell">
          <header className="top-header">
            <div className="brand-cluster">
              <div className="brand-mark-wrap">
                <div
                  className="eclipse-moon"
                  aria-label="Rotating moon"
                >
                  <Moon3D className="moon-3d-small" />
                </div>
                <div className="brand-copy">
                  <div className="wordmark">AETHON</div>
                  <div className="tagline">SEE. UNDERSTAND. ASSIST.</div>
                </div>
              </div>
            </div>

            <div className="header-telemetry">
              <StatusDot label="OFFLINE MODE" />
              <span className="telemetry-separator" />
              <span>{dateTimeStr.date}</span>
              <span className="telemetry-separator" />
              <span>{dateTimeStr.time}</span>
            </div>

          <div className="institution-badge header-institution" aria-label="Indian Space Research Organisation">
            <img src={ISRO_LOGO} alt="Indian Space Research Organisation" className="isro-logo" />
          </div>
        </header>

        <GlassPanel className="mission-bar" aria-label="Current experiment">
          <div className="mission-label">
            <span className="mission-kicker">EXPERIMENT:</span>
            <strong>Payload Assembly (Demo)</strong>
          </div>
        </GlassPanel>

        <div className="dashboard-layout">
          <aside className="nav-rail">
            <div className="nav-rail-content">
              <button className="nav-rail-trigger" type="button" aria-label="Open dashboard navigation">
                <Menu size={22} strokeWidth={1.6} />
                <span>MENU</span>
              </button>
              <nav className="nav-stack" aria-label="Primary navigation">
                {navItems.map(({ label, icon: Icon }) => {
                  const isActive = currentView === label;
                  return (
                    <button
                      className={`nav-item ${isActive ? "active" : ""}`}
                      key={label}
                      type="button"
                      onClick={() => setCurrentView(label)}
                    >
                      <Icon size={20} strokeWidth={1.65} />
                      <span>{label}</span>
                    </button>
                  );
                })}
              </nav>
              <div className="rail-footer">
                <span className="rail-rule" />
                <p>BUILT FOR<br />A SAFER HUMAN<br />SPACEFLIGHT.</p>
                <div className="rail-orbit"><span /></div>
              </div>
            </div>
          </aside>

          {/* CENTRAL WORKSPACE */}
          <section className="workspace">
            {currentView === "Monitor" && (
              <>
                <div className="workspace-grid">
                  <GlassPanel className="camera-panel">
                    <PanelTitle
                      title="Live Camera Feed"
                      subtitle={isCameraOn ? "Real-time AI perception" : "Feed Standby / Offline"}
                      action={
                        <div className="camera-status" style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "nowrap" }}>
                          <span
                            style={{
                              background: "rgba(34, 211, 238, 0.15)",
                              border: "1px solid rgba(34, 211, 238, 0.4)",
                              borderRadius: 6,
                              padding: "3px 8px",
                              display: "inline-flex",
                              alignItems: "center",
                              gap: 5,
                              color: "#67e8f9",
                              fontSize: 11,
                              fontWeight: 700,
                              letterSpacing: "0.05em",
                              whiteSpace: "nowrap"
                            }}
                          >
                            <span
                              style={{
                                width: 6,
                                height: 6,
                                borderRadius: "50%",
                                background: "#22d3ee",
                                boxShadow: "0 0 6px #22d3ee"
                              }}
                            />
                            CAM ONLINE
                          </span>
                          <StatusDot label="REC" tone={isRecording ? "white" : "white"} />
                          <span style={{ fontSize: 11, fontFamily: "monospace", color: "#94a3b8", whiteSpace: "nowrap" }}>{`${fps} FPS`}</span>
                          {latencyMs !== null && (
                            <span style={{ fontSize: 11, fontFamily: "monospace", color: "#94a3b8", whiteSpace: "nowrap" }} title="Camera capture to telemetry latency">
                              {Math.round(latencyMs)} ms
                            </span>
                          )}
                          <button
                            type="button"
                            onClick={handleFullscreen}
                            aria-label="Toggle Fullscreen"
                            style={{ background: "transparent", border: 0, padding: "0 2px", cursor: "pointer", display: "inline-flex", alignItems: "center", color: "#94a3b8" }}
                          >
                            <Maximize2 size={16} />
                          </button>
                        </div>
                      }
                    />
                    <div
                      className="camera-stage"
                      ref={cameraStageRef}
                      aria-label="Live camera feed"
                      data-feed-status={isCameraOn ? "live" : "disabled"}
                    >
                      {!isCameraOn ? (
                        <div
                          style={{
                            width: "100%",
                            height: "100%",
                            display: "flex",
                            flexDirection: "column",
                            alignItems: "center",
                            justifyContent: "center",
                            background: "radial-gradient(circle at 50% 50%, rgba(18, 24, 30, 0.98), rgba(7, 9, 11, 0.99))",
                            position: "relative",
                            userSelect: "none",
                            padding: 24
                          }}
                        >
                          <div
                            style={{
                              width: 60,
                              height: 60,
                              borderRadius: "50%",
                              background: "rgba(239, 68, 68, 0.12)",
                              border: "1px solid rgba(239, 68, 68, 0.3)",
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "center",
                              marginBottom: 12,
                              boxShadow: "0 0 24px rgba(239, 68, 68, 0.12)"
                            }}
                          >
                            <CameraOff size={26} style={{ color: "#f87171" }} />
                          </div>
                          <div style={{ fontSize: 13, fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase", color: "#f1f5f7" }}>
                            Camera Feed Standby
                          </div>
                          <div style={{ fontSize: 12, color: "rgba(241, 245, 246, 0.45)", marginTop: 4, textAlign: "center" }}>
                            Camera feed is offline. Click below to turn camera on.
                          </div>
                          <button
                            type="button"
                            className="select-button"
                            onClick={handleToggleCamera}
                            style={{
                              marginTop: 16,
                              background: "rgba(34, 211, 238, 0.18)",
                              borderColor: "rgba(34, 211, 238, 0.5)",
                              color: "#67e8f9",
                              fontWeight: 600,
                              gap: 8,
                              padding: "0 16px",
                              minHeight: 34,
                              justifyContent: "center"
                            }}
                          >
                            <Camera size={14} />
                            <span>Turn Camera On</span>
                          </button>
                        </div>
                      ) : cameraError ? (
                        <div
                          style={{
                            display: "flex",
                            flexDirection: "column",
                            alignItems: "center",
                            justifyContent: "center",
                            height: "100%",
                            width: "100%",
                            padding: "24px 20px",
                            textAlign: "center",
                            background: "radial-gradient(ellipse at center, rgba(35, 15, 20, 0.95) 0%, rgba(12, 14, 18, 0.98) 100%)",
                            zIndex: 10
                          }}
                        >
                          <CameraOff size={36} color="#f87171" style={{ marginBottom: 12, filter: "drop-shadow(0 0 12px rgba(248, 113, 113, 0.45))" }} />
                          <div style={{ color: "#f87171", fontSize: 16, fontWeight: 700, letterSpacing: "0.04em", marginBottom: 6 }}>
                            Camera Unavailable
                          </div>
                          <div style={{ color: "rgba(220, 230, 240, 0.7)", fontSize: 12.5, maxWidth: 340, lineHeight: 1.45, marginBottom: 18 }}>
                            {cameraDisplayName ? `Unable to access "${cameraDisplayName}". Please verify the source is active or select another camera.` : "The selected video device is currently unavailable."}
                          </div>
                          <button
                            type="button"
                            className="select-button"
                            onClick={() => {
                              setCameraError(null);
                              setCameraLoading(true);
                              setStreamKey(Date.now());
                            }}
                            style={{
                              display: "inline-flex",
                              alignItems: "center",
                              gap: 7,
                              padding: "7px 16px",
                              fontSize: 12,
                              fontWeight: 600,
                              background: "rgba(248, 113, 113, 0.16)",
                              border: "1px solid rgba(248, 113, 113, 0.45)",
                              color: "#fca5a5",
                              borderRadius: 6,
                              cursor: "pointer"
                            }}
                          >
                            <RefreshCw size={13} />
                            <span>Retry Connection</span>
                          </button>
                        </div>
                      ) : useBrowserWebcam ? (
                        <>
                          <video
                            ref={videoRef}
                            autoPlay
                            playsInline
                            muted
                            onPause={(e) => {
                              try { e.target.play(); } catch (err) {}
                            }}
                            style={{
                              width: "100%",
                              height: "100%",
                              objectFit: "cover",
                              transform: mirrorFeed ? "scaleX(-1)" : "none",
                              pointerEvents: "none"
                            }}
                          />
                          <canvas
                            ref={overlayCanvasRef}
                            className="camera-overlay-canvas" style={{ transform: "none" }}
                          />
                        </>
                      ) : (
                        <img
                          key={streamKey}
                          src={`${API_BASE}/api/camera/stream?k=${streamKey}`}
                          alt="Live camera feed"
                          onError={() => {
                            setTimeout(() => setStreamKey(Date.now()), 1500);
                          }}
                        />
                      )}
                      <div className="camera-shade" />

                      <span className="camera-crosshair top-left" />
                      <span className="camera-crosshair bottom-right" />
                    </div>

                    <div className="camera-footer" style={{ position: "relative" }}>
                      <div style={{ position: "relative", display: "flex", gap: 6, alignItems: "center" }}>
                        <button
                          className="select-button"
                          type="button"
                          onClick={() => {
                            if (!showCameraMenu) {
                              if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
                                navigator.mediaDevices.enumerateDevices().then((devices) => {
                                  const videoInputs = devices.filter((d) => d.kind === "videoinput");
                                  setBrowserWebcams(videoInputs);
                                });
                              }
                              fetch(`${API_BASE}/api/camera/devices`)
                                .then((res) => res.json())
                                .then((data) => {
                                  if (data && data.devices) setCameras(data.devices);
                                })
                                .catch(() => {});
                            }
                            setShowCameraMenu(!showCameraMenu);
                          }}
                        >
                          <span>{cameraDisplayName}</span>
                          <ChevronDown size={14} />
                        </button>
                        <button
                          type="button"
                          className="select-button"
                          style={{ minWidth: 36, width: 36, padding: 0, justifyContent: "center" }}
                          onClick={() => {
                            setIsCameraOn(true);
                            setStreamKey(Date.now());
                          }}
                          title="Reconnect stream"
                        >
                          <RefreshCw size={13} />
                        </button>
                        {showCameraMenu && (
                          <div
                            style={{
                              position: "absolute",
                              bottom: "100%",
                              left: 0,
                              marginBottom: 6,
                              background: "rgba(18, 22, 26, 0.96)",
                              border: "1px solid rgba(255, 255, 255, 0.2)",
                              borderRadius: 8,
                              padding: 4,
                              zIndex: 50,
                              minWidth: 240,
                              maxHeight: 260,
                              overflowY: "auto",
                              backdropFilter: "blur(12px)"
                            }}
                          >
                            {browserWebcams.map((bcam, i) => {
                              const isSelected = useBrowserWebcam && (selectedBrowserCamId === bcam.deviceId || (!selectedBrowserCamId && i === 0));
                              const displayName = bcam.label || `Camera ${i} (USB)`;
                              return (
                                <button
                                  key={bcam.deviceId || `bcam-${i}`}
                                  type="button"
                                  style={{
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "space-between",
                                    width: "100%",
                                    textAlign: "left",
                                    padding: "8px 10px",
                                    fontSize: 12.5,
                                    background: isSelected ? "rgba(34, 211, 238, 0.18)" : "transparent",
                                    border: `1px solid ${isSelected ? "rgba(34, 211, 238, 0.4)" : "transparent"}`,
                                    borderRadius: 6,
                                    color: isSelected ? "#67e8f9" : "#f1f5f7",
                                    cursor: "pointer",
                                    fontWeight: isSelected ? 600 : 400,
                                    marginBottom: 2
                                  }}
                                  onClick={() => handleSelectBrowserCamera(bcam.deviceId)}
                                >
                                  <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 190 }}>
                                    {displayName}
                                  </span>
                                  {isSelected && (
                                    <span style={{ color: "#22d3ee", fontSize: 10, fontWeight: 700, marginLeft: 8 }}>● ACTIVE</span>
                                  )}
                                </button>
                              );
                            })}

                            {cameras.map((cam) => {
                              const isSelected = !useBrowserWebcam && selectedCamera === cam.index;
                              return (
                                <button
                                  key={`backend-cam-${cam.index}`}
                                  type="button"
                                  style={{
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "space-between",
                                    width: "100%",
                                    textAlign: "left",
                                    padding: "8px 10px",
                                    fontSize: 12.5,
                                    background: isSelected ? "rgba(34, 211, 238, 0.18)" : "transparent",
                                    border: `1px solid ${isSelected ? "rgba(34, 211, 238, 0.4)" : "transparent"}`,
                                    borderRadius: 6,
                                    color: isSelected ? "#67e8f9" : "#f1f5f7",
                                    cursor: "pointer",
                                    fontWeight: isSelected ? 600 : 400,
                                    marginBottom: 2
                                  }}
                                  onClick={() => handleSelectCamera(cam.index)}
                                >
                                  <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 190 }}>
                                    {cam.name || `Camera ${cam.index}`}
                                  </span>
                                  {isSelected && (
                                    <span style={{ color: "#22d3ee", fontSize: 10, fontWeight: 700, marginLeft: 8 }}>● ACTIVE</span>
                                  )}
                                </button>
                              );
                            })}
                          </div>
                        )}
                      </div>

                      <div className="camera-actions" style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "nowrap" }}>
                        <button
                          type="button"
                          className={mirrorFeed ? "active-action" : ""}
                          onClick={handleToggleMirror}
                          title="Toggle mirror view"
                          style={{
                            display: "inline-flex",
                            alignItems: "center",
                            gap: 5,
                            padding: "5px 10px",
                            borderRadius: 7,
                            fontSize: 12,
                            fontWeight: 600,
                            background: mirrorFeed ? "rgba(34, 197, 94, 0.18)" : "rgba(255, 255, 255, 0.06)",
                            border: `1px solid ${mirrorFeed ? "rgba(34, 197, 94, 0.45)" : "rgba(255, 255, 255, 0.15)"}`,
                            color: mirrorFeed ? "#86efac" : "#cbd5e1",
                            cursor: "pointer",
                            whiteSpace: "nowrap"
                          }}
                        >
                          <FlipHorizontal size={14} /> {mirrorFeed ? "Mirrored" : "Mirror"}
                        </button>
                        <button
                          type="button"
                          onClick={handleSnapshot}
                          title="Capture frame snapshot"
                          style={{
                            display: "inline-flex",
                            alignItems: "center",
                            gap: 5,
                            padding: "5px 10px",
                            borderRadius: 7,
                            fontSize: 12,
                            fontWeight: 600,
                            background: "rgba(255, 255, 255, 0.06)",
                            border: "1px solid rgba(255, 255, 255, 0.15)",
                            color: "#cbd5e1",
                            cursor: "pointer",
                            whiteSpace: "nowrap"
                          }}
                        >
                          <Camera size={14} /> Snapshot
                        </button>
                        <button
                          type="button"
                          className={isRecording ? "active-action" : ""}
                          onClick={handleRecordToggle}
                          title="Toggle video recording"
                          style={{
                            display: "inline-flex",
                            alignItems: "center",
                            gap: 5,
                            padding: "5px 10px",
                            borderRadius: 7,
                            fontSize: 12,
                            fontWeight: 600,
                            background: isRecording ? "rgba(239, 68, 68, 0.2)" : "rgba(255, 255, 255, 0.06)",
                            border: `1px solid ${isRecording ? "rgba(239, 68, 68, 0.5)" : "rgba(255, 255, 255, 0.15)"}`,
                            color: isRecording ? "#fca5a5" : "#cbd5e1",
                            cursor: "pointer",
                            whiteSpace: "nowrap"
                          }}
                        >
                          <span className="record-ring" style={{ width: 8, height: 8, borderRadius: "50%", background: isRecording ? "#ef4444" : "#94a3b8", display: "inline-block" }} />
                          {isRecording ? "Recording" : "Record"}
                        </button>
                      </div>

                      {/* Local confirmation for camera actions — never routed
                          through the assistant conversation. */}
                      {toast && (
                        <div
                          key={toast.id}
                          className={`camera-toast camera-toast-${toast.tone}`}
                          role="status"
                          aria-live="polite"
                        >
                          {toast.tone === "error" ? <AlertTriangle size={13} /> : <Check size={13} />}
                          <span>{toast.text}</span>
                        </div>
                      )}
                    </div>
                  </GlassPanel>

                  <GlassPanel className="progress-panel">
                    <PanelTitle title="Experiment Progress" />
                    <div className="progress-summary">
                      <strong>Step {activeStep} of {experimentState.total_steps || 5}</strong>
                      <span>{stepProgress}</span>
                    </div>
                    <div className="progress-track">
                      <span style={{ width: stepProgress }} />
                    </div>
                    <div className="step-list">
                      {(experimentState.steps || []).map((step) => (
                        <div
                          key={step.number}
                          className={`step-row ${step.current ? "current" : ""} ${step.complete ? "complete" : ""}`}
                        >
                          <span className="step-node">{step.complete ? <Check size={15} /> : step.number}</span>
                          <span className="step-copy">
                            <strong>{step.label}</strong>
                            <small>{step.meta || (step.complete ? "Completed" : step.current ? "In Progress..." : "Pending")}</small>
                          </span>
                        </div>
                      ))}
                    </div>
                  </GlassPanel>
                </div>

                <div className="context-grid">
                  <GlassPanel className="detected-panel">
                    <PanelTitle title="Detected Objects" subtitle="Astronaut & payload taxonomy" />
                    <div className="object-list" style={{ maxHeight: 185, overflowY: "auto" }}>
                      {(() => {
                        const astronautCrew = detectedObjects.filter(
                          (it) => it.category === "ASTRONAUT" || it.raw_label === "Astronaut" || it.label === "Astronaut" || it.raw_label === "Person" || it.label === "Person"
                        );
                        const payloadObjects = detectedObjects.filter(
                          (it) => it.category !== "ASTRONAUT" && it.raw_label !== "Astronaut" && it.label !== "Astronaut" && it.raw_label !== "Person" && it.label !== "Person"
                        );

                        if (detectedObjects.length === 0) {
                          return (
                            <div style={{ padding: "24px 12px", textAlign: "center", color: "rgba(255,255,255,0.40)", fontSize: 12, lineHeight: 1.5 }}>
                              No crew or experiment items in field of view<br />
                              <span style={{ fontSize: 10.5, color: "rgba(255,255,255,0.26)" }}>Position astronaut & apparatus on workspace surface</span>
                            </div>
                          );
                        }

                        return (
                          <>
                            {astronautCrew.length > 0 && (
                              <div style={{ marginBottom: payloadObjects.length > 0 ? 8 : 0 }}>
                                <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4, padding: "0 2px" }}>
                                  <span style={{ fontSize: 10, fontWeight: 700, color: "#00e6c8", letterSpacing: "0.08em", textTransform: "uppercase" }}>
                                    ✦ Crew / Astronaut ({astronautCrew.length})
                                  </span>
                                  <span style={{ height: 1, flex: 1, background: "rgba(0, 230, 200, 0.2)" }} />
                                </div>
                                {astronautCrew.map((item, idx) => (
                                  <Fragment key={`crew-${idx}`}>
                                    <div className="object-row" style={{ background: "rgba(0, 230, 200, 0.06)", border: "1px solid rgba(0, 230, 200, 0.25)", borderRadius: 6, padding: "6px 10px" }}>
                                      <span style={{ display: "flex", alignItems: "center", justifyContent: "center", width: 16, height: 16, borderRadius: "50%", background: "rgba(0, 230, 200, 0.2)", color: "#00e6c8", fontSize: 10 }}>
                                        ✦
                                      </span>
                                      <div style={{ display: "flex", flexDirection: "column", flex: 1, minWidth: 0 }}>
                                        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                                          <span style={{ fontWeight: 600, fontSize: 13, color: "#f0fdf4" }}>
                                            {item.display_name || "✦ ASTRONAUT #1"}
                                          </span>
                                          <span style={{ fontSize: 9, padding: "1px 5px", borderRadius: 3, background: "rgba(0, 230, 200, 0.15)", color: "#00e6c8", border: "1px solid rgba(0, 230, 200, 0.4)", textTransform: "uppercase", fontWeight: 600 }}>
                                            SKELETON ACTIVE
                                          </span>
                                        </div>
                                        <span style={{ fontSize: 10, color: "rgba(255,255,255,0.6)", letterSpacing: "0.02em" }}>
                                          {item.role || "Operator"} • Posture: {item.posture || currentAction.posture || "Seated"}{item.action ? ` • [${item.action}]` : (item.activity ? ` • [${item.activity}]` : "")}
                                        </span>
                                      </div>
                                      <strong style={{ color: "#00e6c8" }}>{item.confidence !== undefined ? item.confidence.toFixed(2) : "0.98"}</strong>
                                    </div>
                                    {item.hands && item.hands.length > 0 && item.hands.map((h, hIdx) => (
                                      <div className="object-row" key={`subhand-${hIdx}`} style={{ paddingLeft: 22, background: "rgba(56, 189, 248, 0.04)", borderLeft: "2px solid rgba(56, 189, 248, 0.4)", margin: "2px 0 2px 8px", borderRadius: "0 4px 4px 0" }}>
                                        <span style={{ fontSize: 11, color: "#38bdf8", fontWeight: 600 }}>
                                          └─ {h.side} Hand
                                        </span>
                                        <span style={{ fontSize: 9.5, padding: "1px 4px", borderRadius: 3, background: "rgba(56, 189, 248, 0.12)", color: "#7dd3fc", marginLeft: "auto", textTransform: "uppercase", fontWeight: 500 }}>
                                          {h.gesture && h.gesture !== "NONE" ? h.gesture : (h.is_grasping ? "GRASP" : (h.is_pinching ? "PINCH" : "TRACKED"))}
                                        </span>
                                      </div>
                                    ))}
                                  </Fragment>
                                ))}
                              </div>
                            )}

                            <div>
                              <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4, marginTop: astronautCrew.length > 0 ? 6 : 0, padding: "0 2px" }}>
                                <span style={{ fontSize: 10, fontWeight: 700, color: "rgba(255, 255, 255, 0.6)", letterSpacing: "0.08em", textTransform: "uppercase" }}>
                                  ✦ Experiment Payload & Tools ({payloadObjects.length})
                                </span>
                                <span style={{ height: 1, flex: 1, background: "rgba(255, 255, 255, 0.12)" }} />
                              </div>
                              {payloadObjects.length === 0 ? (
                                <div style={{ padding: "8px 4px", color: "rgba(255,255,255,0.36)", fontSize: 11.5 }}>
                                  No apparatus or tools in field of view
                                </div>
                              ) : (
                                payloadObjects.map((item, idx) => {
                                  const badgeStyle = getCategoryBadgeStyle(item.category_badge || item.category);
                                  return (
                                    <Fragment key={`${item.label}-${idx}`}>
                                      <div className="object-row" style={{ padding: "4px 8px", borderRadius: 4, background: item.held ? "rgba(234, 179, 8, 0.08)" : "rgba(255, 255, 255, 0.02)", border: item.held ? "1px solid rgba(234, 179, 8, 0.3)" : "1px solid transparent" }}>
                                        <span className={`object-swatch ${item.colorClass || item.color || "object-white"}`} style={{ flexShrink: 0 }} />
                                        <div style={{ display: "flex", flexDirection: "column", flex: 1, minWidth: 0 }}>
                                          <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "nowrap" }}>
                                            <span style={{ fontSize: 9, padding: "1px 4px", borderRadius: 3, background: badgeStyle.bg, color: badgeStyle.color, border: `1px solid ${badgeStyle.border}`, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.03em", flexShrink: 0 }}>
                                              {item.category_badge || "ITEM"}
                                            </span>
                                            <span style={{ fontWeight: 500, fontSize: 12.5, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                                              {item.display_name || item.label}
                                            </span>
                                          </div>
                                          <span style={{ fontSize: 10, color: "rgba(255,255,255,0.5)", textTransform: "uppercase", letterSpacing: "0.04em", marginTop: 1 }}>
                                            {item.position ? `Pos: (${item.position.cx}, ${item.position.cy})` : `Color: ${item.colorName || "Default"}`}
                                            {item.held ? ` • Held by ${item.heldBy || "Crew"}` : ""}
                                            {item.moving && !item.held ? " • Moving" : ""}
                                          </span>
                                        </div>
                                        <strong>{item.confidence !== undefined ? item.confidence.toFixed(2) : "0.90"}</strong>
                                      </div>
                                    </Fragment>
                                  );
                                })
                              )}
                            </div>
                          </>
                        );
                      })()}
                    </div>
                  </GlassPanel>

                  <GlassPanel className="action-panel">
                    <PanelTitle title="Current Action" subtitle="Perception & movement state" />
                    <div className="current-action-content">
                      <div className="action-thumbnail">
                        <span className="fake-hand" />
                        <span className="fake-block" />
                      </div>
                      <div className="action-copy">
                        <strong style={{ fontSize: 14 }}>{currentAction.label || "Ready / Monitoring"}</strong>
                        {currentAction.narration && currentAction.narration !== "Monitoring" && (
                          <span style={{ color: "#93c5fd", fontSize: 12, fontWeight: 500, lineHeight: 1.3 }}>
                            {currentAction.narration}
                          </span>
                        )}
                        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", margin: "2px 0" }}>
                          <span style={{ fontSize: 10, padding: "2px 6px", borderRadius: 4, background: "rgba(255,255,255,0.08)", border: "1px solid rgba(255,255,255,0.15)", fontFamily: "var(--mono)", color: "#e2e8f0" }}>
                            MOVE: {currentAction.movement || "Stationary"}
                          </span>
                          <span style={{ fontSize: 10, padding: "2px 6px", borderRadius: 4, background: "rgba(255,255,255,0.08)", border: "1px solid rgba(255,255,255,0.15)", fontFamily: "var(--mono)", color: "#e2e8f0" }}>
                            POSTURE: {currentAction.posture || "Seated"}
                          </span>
                        </div>
                        <span style={{ fontSize: 11.5, color: "rgba(215, 225, 230, 0.65)" }}>
                          {currentAction.hand || "Right hand"} → {currentAction.color ? `${currentAction.color} ` : ""}{currentAction.object || "Object A"} ({currentAction.confidence ? currentAction.confidence.toFixed(2) : "0.92"})
                        </span>
                        <StatusDot
                          label={running ? "In Progress" : experimentState.status === "COMPLETED" ? "Completed" : "Active Perception"}
                          tone={running ? "white" : "amber"}
                        />
                      </div>
                    </div>
                  </GlassPanel>

                  <GlassPanel className="next-panel">
                    <PanelTitle title="Next Step" />
                    <div className="next-content">
                      <div className="assembly-visual">
                        <img src={ASSEMBLY} alt="Red block positioned above a wooden block" />
                      </div>
                      <div className="next-copy">
                        <strong>{experimentState.next_step_label || "Place Object A on Object B."}</strong>
                        <span>Follow the sequence validator instructions closely.</span>
                      </div>
                    </div>
                  </GlassPanel>
                </div>

                <GlassPanel className="experiment-controls">
                  <div className="controls-lead">
                    <button
                      className="big-play-button"
                      type="button"
                      onClick={running ? handlePause : handleStart}
                      aria-label={running ? "Pause experiment" : "Start experiment"}
                    >
                      {running ? <Pause size={24} fill="currentColor" /> : <Play size={24} fill="currentColor" />}
                    </button>
                    <div className="controls-copy">
                      <strong>Experiment Controls</strong>
                    </div>
                  </div>

                  <div className="control-buttons">
                    <button className="control-button primary" type="button" onClick={handleStart}>
                      <Play size={14} fill="currentColor" /> Start
                    </button>
                    <button className="control-button" type="button" onClick={handlePause}>
                      <Pause size={14} /> Pause
                    </button>
                    <button className="control-button" type="button" onClick={handleReset}>
                      <RotateCcw size={14} /> Reset
                    </button>
                    <button className="control-button" type="button" onClick={handleStop}>
                      <Square size={13} fill="currentColor" /> End
                    </button>
                  </div>
                </GlassPanel>
              </>
            )}

            {/* EXPERIMENT VIEW */}
            {currentView === "Experiment" && (
              <GlassPanel className="workspace-alt-view">
                <div className="alt-view-header">
                  <div>
                    <h3>Experiment Definition: Payload Assembly (Demo)</h3>
                    <p style={{ margin: "4px 0 0", fontSize: 12, color: "rgba(235, 242, 245, 0.6)", fontFamily: "var(--mono)" }}>
                      SIH26174 / SIH2026174 — 5-Step Procedural Experiment
                    </p>
                  </div>
                  <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                    <span className="demo-mode-badge">DEMO / TEST SIMULATOR</span>
                    <button className="control-button primary" style={{ height: 34, padding: "0 14px" }} type="button" onClick={handleStart}>
                      <Play size={13} fill="currentColor" /> Start
                    </button>
                  </div>
                </div>

                <div className="alt-view-body">
                  <div>
                    <h4 style={{ margin: "0 0 8px", fontSize: 13, textTransform: "uppercase", letterSpacing: "0.08em", color: "rgba(240, 246, 248, 0.8)" }}>
                      Physical Objects in Experiment
                    </h4>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
                      {[
                        ["Astronaut", "Mission operator / EVA specialist", "object-cyan"],
                        ["Object A", "Red block (Pick & place item)", "object-red"],
                        ["Object B", "Wooden / Blue block (Base target)", "object-blue"],
                        ["Tray", "Payload holding tray", "object-purple"],
                        ["Complete Button", "Mission complete trigger", "object-yellow"],
                        ["Payload Tools", "Shears, scalpel, sample probes", "object-orange"],
                        ["Fluid Flasks", "Specimen containers & beakers", "object-cyan"],
                        ["Flight Tech", "Terminals, comms, avionics", "object-blue"],
                      ].map(([name, desc, clr]) => (
                        <div key={name} style={{ display: "flex", alignItems: "center", gap: 8, padding: "6px 12px", background: "rgba(255,255,255,0.04)", borderRadius: 6, border: "1px solid rgba(255,255,255,0.1)" }}>
                          <span className={`object-swatch ${clr}`} />
                          <span style={{ fontSize: 13, fontWeight: 500 }}>{name}</span>
                          <span style={{ fontSize: 11, color: "rgba(255,255,255,0.5)", fontFamily: "var(--mono)" }}>({desc})</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div>
                    <h4 style={{ margin: "0 0 8px", fontSize: 13, textTransform: "uppercase", letterSpacing: "0.08em", color: "rgba(240, 246, 248, 0.8)" }}>
                      5-Step Procedure Sequence
                    </h4>
                    <div className="step-card-grid">
                      {[
                        { num: 1, label: "Pick up Object A", act: "PICK_UP(Object A)", desc: "Lift the red block from resting position." },
                        { num: 2, label: "Place Object A on Object B", act: "PLACE(Object A, Object B)", desc: "Position the red block on top of the wooden block." },
                        { num: 3, label: "Pick up Object A again", act: "PICK_UP(Object A)", desc: "Lift the red block from Object B." },
                        { num: 4, label: "Return Object A to Tray", act: "PLACE(Object A, Tray)", desc: "Return the red block into the assembly tray." },
                        { num: 5, label: "Press the Complete Button", act: "PRESS(Complete Button)", desc: "Push the complete button to conclude." },
                      ].map((s) => (
                        <div key={s.num} className={`step-card ${experimentState.current_step === s.num ? "active-step" : ""}`}>
                          <div className="step-card-top">
                            <span style={{ fontSize: 12, fontFamily: "var(--mono)", color: "rgba(255,255,255,0.6)" }}>Step {s.num}</span>
                            {experimentState.current_step > s.num ? (
                              <span style={{ color: "#86efac", fontSize: 12, display: "flex", alignItems: "center", gap: 4 }}><Check size={13} /> Done</span>
                            ) : experimentState.current_step === s.num ? (
                              <span style={{ color: "#fde047", fontSize: 12 }}>Current</span>
                            ) : (
                              <span style={{ color: "rgba(255,255,255,0.4)", fontSize: 12 }}>Pending</span>
                            )}
                          </div>
                          <strong>{s.label}</strong>
                          <p style={{ color: "#93c5fd" }}>Expected: {s.act}</p>
                          <p>{s.desc}</p>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* SIMULATION CONTROLS FOR DEMO / TEST MODE */}
                  <div style={{ marginTop: 8, padding: 14, background: "rgba(234, 179, 8, 0.04)", border: "1px solid rgba(234, 179, 8, 0.25)", borderRadius: 10 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                      <span className="demo-mode-badge">DEMO / TEST SIMULATOR</span>
                      <span style={{ fontSize: 12, color: "rgba(255,255,255,0.7)" }}>Inject actions without physical blocks:</span>
                    </div>
                    <div className="sim-buttons-row">
                      <button className="sim-btn" type="button" onClick={() => handleSimulateAction("PICK_UP", "Object A")}>
                        Simulate Step 1: PICK_UP(Object A)
                      </button>
                      <button className="sim-btn" type="button" onClick={() => handleSimulateAction("PLACE", "Object A", "Object B")}>
                        Simulate Step 2: PLACE(Object A, Object B)
                      </button>
                      <button className="sim-btn" type="button" onClick={() => handleSimulateAction("PICK_UP", "Object A")}>
                        Simulate Step 3: PICK_UP(Object A)
                      </button>
                      <button className="sim-btn" type="button" onClick={() => handleSimulateAction("PLACE", "Object A", "Tray")}>
                        Simulate Step 4: PLACE(Object A, Tray)
                      </button>
                      <button className="sim-btn" type="button" onClick={() => handleSimulateAction("PRESS", "Complete Button")}>
                        Simulate Step 5: PRESS(Complete Button)
                      </button>
                      <button className="sim-btn" style={{ borderColor: "#fca5a5", color: "#fca5a5" }} type="button" onClick={() => handleSimulateAction("PICK_UP", "Object B")}>
                        Trigger Wrong Object (Object B)
                      </button>
                      <button className="sim-btn" style={{ borderColor: "#fde047", color: "#fde047" }} type="button" onClick={() => handleSimulateAction("PLACE", "Object A", "Tray")}>
                        Trigger Out-of-Order Action
                      </button>
                    </div>
                  </div>
                </div>
              </GlassPanel>
            )}

            {/* LOGS VIEW */}
            {currentView === "Logs" && (
              <GlassPanel className="workspace-alt-view">
                <div className="alt-view-header">
                  <div>
                    <h3>Structured Experiment Event Logs</h3>
                    <p style={{ margin: "4px 0 0", fontSize: 12, color: "rgba(235, 242, 245, 0.6)", fontFamily: "var(--mono)" }}>
                      Live JSONL local telemetry and audit trail
                    </p>
                  </div>
                  <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                    <select
                      value={logFilter}
                      onChange={(e) => setLogFilter(e.target.value)}
                      style={{
                        background: "rgba(14, 18, 22, 0.8)",
                        border: "1px solid rgba(255, 255, 255, 0.2)",
                        borderRadius: 6,
                        color: "#f1f5f7",
                        padding: "6px 10px",
                        fontSize: 12,
                        fontFamily: "var(--mono)"
                      }}
                    >
                      <option value="ALL">All Events</option>
                      <option value="STEP_COMPLETED">Step Completed</option>
                      <option value="VIOLATION">Violations</option>
                      <option value="ACTION_DETECTED">Action Detected</option>
                      <option value="VOICE_COMMAND">Voice Command</option>
                      <option value="SNAPSHOT_CAPTURED">Snapshots</option>
                    </select>

                    <button
                      className="control-button"
                      style={{ height: 32, padding: "0 10px" }}
                      type="button"
                      onClick={() => {
                        fetch(`${API_BASE}/api/logs?limit=80`)
                          .then((r) => r.json())
                          .then((d) => { if (d.logs) setLogs(d.logs); });
                      }}
                    >
                      <RefreshCw size={13} /> Refresh
                    </button>
                  </div>
                </div>

                <div className="alt-view-body" style={{ padding: 14 }}>
                  <div className="logs-table-wrap">
                    <table className="logs-table">
                      <thead>
                        <tr>
                          <th style={{ width: 90 }}>Time</th>
                          <th style={{ width: 150 }}>Event Type</th>
                          <th>Message</th>
                        </tr>
                      </thead>
                      <tbody>
                        {logs
                          .filter((l) => logFilter === "ALL" || l.event_type === logFilter)
                          .map((log, idx) => {
                            let tagClass = "log-tag-exp";
                            if (log.event_type === "STEP_COMPLETED") tagClass = "log-tag-step";
                            else if (log.event_type === "VIOLATION") tagClass = "log-tag-violation";
                            else if (log.event_type === "VOICE_COMMAND" || log.event_type === "AETHON_RESPONSE") tagClass = "log-tag-voice";
                            else if (log.event_type === "ACTION_DETECTED") tagClass = "log-tag-action";

                            return (
                              <tr key={idx}>
                                <td style={{ whiteSpace: "nowrap" }}>{log.time_str || log.timestamp?.slice(11, 19)}</td>
                                <td><span className={`log-tag ${tagClass}`}>{log.event_type}</span></td>
                                <td>{log.message}</td>
                              </tr>
                            );
                          })}
                        {logs.length === 0 && (
                          <tr>
                            <td colSpan={3} style={{ textAlign: "center", padding: 24, color: "rgba(255,255,255,0.4)" }}>
                              No event logs recorded yet. Start experiment to begin logging.
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              </GlassPanel>
            )}
          </section>

          {/* RIGHT ASSISTANT PANEL */}
          <aside className="assistant-panel">
            <GlassPanel className="assistant-shell">
              <div className="assistant-heading">
                <div className="assistant-orb" aria-label="Rotating moon">
                  <Moon3D className="moon-3d-large" />
                </div>
                <div>
                  <div className="assistant-name">AETHON</div>
                  <p>Spaceflight Mission Assistant</p>
                </div>
              </div>

              <div className="assistant-mode-row">
                <div
                  className={`waveform ${isAssistantActive || voiceStatus === "wake_detected" || voiceStatus === "processing" ? "animating" : "idle"}`}
                  aria-label="Aethon voice waveform"
                  title={isAssistantActive || voiceStatus === "wake_detected" || voiceStatus === "processing" ? "AETHON active" : "AETHON standby – Say 'Hey AETHON'"}
                >
                  <i /><i /><i /><i /><i /><i /><i /><i />
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  {audioRecording ? (
                    <span style={{
                      fontFamily: "var(--display)",
                      fontSize: "12px",
                      color: "#ef4444",
                      fontWeight: 600,
                      letterSpacing: "0.04em",
                      animation: "pulse 1s ease-in-out infinite"
                    }}>
                      ● RECORDING
                    </span>
                  ) : (
                    <span className="say-aethon-hint">
                      Say &quot;Hey AETHON&quot;
                    </span>
                  )}
                </div>
              </div>

              <div className="conversation">
                {messages.map((m) => (
                  <div
                    key={m.id}
                    className={`conversation-message ${m.role === "user" ? "user-message" : "assistant-message"}`}
                  >
                    <span className="message-avatar">
                      {m.role === "user" ? <UserRound size={17} /> : <Bot size={17} />}
                    </span>
                    <div className="message-bubble">
                      <small>{m.speaker} <time>{m.time}</time></small>
                      <p style={{ whiteSpace: "pre-line" }}>{m.text}</p>
                    </div>
                  </div>
                ))}
                <div ref={chatBottomRef} />
              </div>

              <div className="assistant-suggestions">
                {[
                  "What is the mission?",
                  "Show all steps",
                  "What's the next step?",
                  "What is my progress?",
                  "What am I doing?",
                  "What is this object?",
                  "Safety check",
                  "Which hand am I using?",
                  "Check my speed",
                  "What color is this?",
                  "Am I doing it right?",
                  "Space fact"
                ].map((prompt) => (
                  <button
                    type="button"
                    key={prompt}
                    onClick={() => sendCommand(prompt)}
                  >
                    {prompt}
                  </button>
                ))}
              </div>

              {liveTranscript && (
                <div style={{
                  padding: "6px 10px",
                  margin: "0 0 6px",
                  borderRadius: "6px",
                  background: "rgba(34, 197, 94, 0.15)",
                  border: "1px solid rgba(34, 197, 94, 0.35)",
                  color: "#86efac",
                  fontSize: "0.78rem",
                  fontFamily: "var(--mono)",
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                  animation: "pulse 2s infinite"
                }}>
                  <span style={{ color: "#22c55e", fontWeight: "bold" }}>
                    {wakeState === WAKE_STATE.CAPTURING_COMMAND ? "🎙️ Command:" : "🎙️ Hearing:"}
                  </span>
                  <span>"{liveTranscript}"</span>
                </div>
              )}

              <div className="assistant-input-wrap">
                <button
                  type="button"
                  /* `active` = mic is armed/highlighted. The pulse animation
                     is bound separately to actual detected speech, so "armed,
                     waiting" and "hearing you now" look different.
                     `capturing` = wake word matched, actively capturing command. */
                  className={`input-mic ${isListening || continuousListening || audioRecording ? "active" : ""} ${wakeState === WAKE_STATE.CAPTURING_COMMAND ? "capturing" : ""}`}
                  style={{
                    color: audioRecording
                      ? "#ef4444"
                      : wakeState === WAKE_STATE.CAPTURING_COMMAND
                      ? "#38bdf8"
                      : continuousListening
                      ? "#22c55e"
                      : isListening
                      ? "#3b82f6"
                      : "inherit",
                    animation: audioRecording
                      ? "pulse 1s ease-in-out infinite"
                      : wakeState === WAKE_STATE.CAPTURING_COMMAND
                      ? "glow-cyan 1.2s ease-in-out infinite"
                      : isSpeechActive
                      ? "pulse 1.5s ease-in-out infinite"
                      : "none"
                  }}
                  onClick={handleMicClick}
                  title={
                    audioRecording
                      ? "Click to stop recording and send"
                      : wakeState === WAKE_STATE.CAPTURING_COMMAND
                      ? "Listening for your command..."
                      : continuousListening
                      ? "Armed — say 'Hey AETHON' (click to disable)"
                      : "Push to talk"
                  }
                >
                  <Mic size={18} />
                </button>
                <input
                  aria-label="Ask Aethon"
                  value={assistantDraft}
                  onChange={(e) => setAssistantDraft(e.target.value)}
                  placeholder={
                    liveTranscript
                      ? `Hearing: "${liveTranscript}"`
                      : "Say 'Hey AETHON' or type a command..."
                  }
                  onKeyDown={(e) => {
                    if (e.key === "Enter") sendCommand(assistantDraft);
                  }}
                />
                <button
                  type="button"
                  className="send-button"
                  onClick={() => sendCommand(assistantDraft)}
                >
                  <Send size={18} fill="currentColor" />
                </button>
              </div>
            </GlassPanel>
          </aside>
        </div>
      </div>
    </main>
    </>
  );
}

function jsonParseSafe(str) {
  try {
    return JSON.parse(str);
  } catch (e) {
    return null;
  }
}
