import { useEffect, useMemo, useRef, useState, useCallback } from "react";
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
      () => {
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
  const [useBrowserWebcam, setUseBrowserWebcam] = useState(false);
  const [isCameraOn, setIsCameraOn] = useState(true);
  const [mirrorFeed, setMirrorFeed] = useState(true);
  const cameraStageRef = useRef(null);
  const videoRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const latestPerceptionRef = useRef(null);

  // Browser Webcam MediaDevices & AI streaming pipeline effect
  useEffect(() => {
    let localStream = null;
    let uploadInterval = null;
    let animId = null;
    let isUploading = false;
    const captureCanvas = document.createElement("canvas");

    if (useBrowserWebcam && isCameraOn) {
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices
          .getUserMedia({
            video: {
              width: { ideal: 1280 },
              height: { ideal: 720 },
              frameRate: { ideal: 60, min: 30 }
            }
          })
          .then((stream) => {
            localStream = stream;
            if (videoRef.current) {
              videoRef.current.srcObject = stream;
            }

            // 1. Frame uploader: sends browser webcam frames to backend AI perception pipeline
            uploadInterval = setInterval(() => {
              const video = videoRef.current;
              if (!video || video.readyState < 2 || isUploading) return;
              const w = 640;
              const h = Math.round((video.videoHeight / (video.videoWidth || 1)) * w) || 360;
              captureCanvas.width = w;
              captureCanvas.height = h;
              const ctx = captureCanvas.getContext("2d");
              if (!ctx) return;
              ctx.drawImage(video, 0, 0, w, h);
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
                0.65
              );
            }, 50); // ~20 FPS frame ingestion into backend

            // 2. Real-time overlay canvas drawing loop
            const renderOverlays = () => {
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
                    const vw = video.videoWidth || 1280;
                    const vh = video.videoHeight || 720;
                    const scaleX = cw / vw;
                    const scaleY = ch / vh;

                    // Draw MediaPipe Pose Skeleton
                    if (p.pose && p.pose.landmarks) {
                      const lms = p.pose.landmarks;
                      const connections = [
                        [11, 12], [11, 23], [12, 24], [23, 24],
                        [11, 13], [13, 15], [12, 14], [14, 16],
                        [23, 25], [25, 27], [27, 29], [29, 31],
                        [24, 26], [26, 28], [28, 30], [30, 32],
                        [11, 0], [12, 0]
                      ];
                      ctx.strokeStyle = "rgba(0, 230, 200, 0.75)";
                      ctx.lineWidth = 2.5;
                      for (const [p1, p2] of connections) {
                        if (lms[p1] && lms[p2] && (lms[p1].visibility ?? 1) > 0.25 && (lms[p2].visibility ?? 1) > 0.25) {
                          ctx.beginPath();
                          ctx.moveTo(lms[p1].x * cw, lms[p1].y * ch);
                          ctx.lineTo(lms[p2].x * cw, lms[p2].y * ch);
                          ctx.stroke();
                        }
                      }
                      // Joints
                      for (const lm of lms) {
                        if ((lm.visibility ?? 1) > 0.25) {
                          ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
                          ctx.beginPath();
                          ctx.arc(lm.x * cw, lm.y * ch, 3, 0, Math.PI * 2);
                          ctx.fill();
                        }
                      }
                    }

                    // Draw Objects & Person Bounding Boxes
                    if (p.objects && p.objects.length > 0) {
                      for (const obj of p.objects) {
                        if (!obj.bbox || obj.bbox.length < 4) continue;
                        const [bx1, by1, bx2, by2] = obj.bbox;
                        const x = bx1 * scaleX;
                        const y = by1 * scaleY;
                        const wBox = (bx2 - bx1) * scaleX;
                        const hBox = (by2 - by1) * scaleY;

                        const isPerson = (obj.raw_label === "Person" || obj.label === "Person");
                        const color = isPerson ? "#ffffff" : (obj.color_hex || "#00e6c8");

                        ctx.strokeStyle = isPerson ? "rgba(240, 240, 240, 0.9)" : color;
                        ctx.lineWidth = isPerson ? 1.5 : 2;
                        ctx.strokeRect(x, y, wBox, hBox);

                        const labelText = isPerson
                          ? `ASTRONAUT ${Math.round((obj.confidence || 0.95) * 100)}%`
                          : `${obj.is_held ? "● HELD: " : ""}${obj.moving || obj.is_moving ? "▲ MOVE: " : ""}${obj.display_name || obj.label}`;
                        ctx.font = "600 11px Inter, sans-serif";
                        const tw = ctx.measureText(labelText).width;
                        const badgeH = 18;
                        const badgeY = Math.max(0, y - badgeH);

                        ctx.fillStyle = isPerson ? "rgba(20, 25, 30, 0.88)" : color;
                        ctx.fillRect(x, badgeY, tw + 10, badgeH);

                        ctx.fillStyle = isPerson ? "#ffffff" : "#0d1117";
                        ctx.fillText(labelText, x + 5, badgeY + 13);
                      }
                    } else if (p.person_detected && p.pose && p.pose.bbox) {
                      // Draw person box from pose bbox if not in objects
                      const [bx1, by1, bx2, by2] = p.pose.bbox;
                      const x = bx1 * scaleX;
                      const y = by1 * scaleY;
                      const wBox = (bx2 - bx1) * scaleX;
                      const hBox = (by2 - by1) * scaleY;

                      ctx.strokeStyle = "rgba(240, 240, 240, 0.9)";
                      ctx.lineWidth = 1.5;
                      ctx.strokeRect(x, y, wBox, hBox);

                      const labelText = "ASTRONAUT (Active)";
                      ctx.font = "600 11px Inter, sans-serif";
                      const tw = ctx.measureText(labelText).width;
                      const badgeH = 18;
                      const badgeY = Math.max(0, y - badgeH);

                      ctx.fillStyle = "rgba(20, 25, 30, 0.88)";
                      ctx.fillRect(x, badgeY, tw + 10, badgeH);
                      ctx.fillStyle = "#ffffff";
                      ctx.fillText(labelText, x + 5, badgeY + 13);
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
                  }
                }
              }
              animId = requestAnimationFrame(renderOverlays);
            };
            animId = requestAnimationFrame(renderOverlays);
          })
          .catch((err) => {
            console.error("[Webcam Error]", err);
            alert("Could not open browser webcam: " + err.message + ". Switching back to AI sensor stream.");
            setUseBrowserWebcam(false);
          });
      }
    }
    return () => {
      if (uploadInterval) clearInterval(uploadInterval);
      if (animId) cancelAnimationFrame(animId);
      if (localStream) {
        localStream.getTracks().forEach((t) => t.stop());
      }
    };
  }, [useBrowserWebcam, isCameraOn]);

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
    narration: "Astronaut is seated and monitoring the payload console.",
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
  const [continuousListening, setContinuousListening] = useState(false);
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

      audio.onerror = (e) => {
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
          setIsAssistantActive(false);
          isSpeakingRef.current = false;
          setVoiceStatus(continuousListeningRef.current ? "listening" : "idle");
          if (currentAudioRef.current === audio) {
            currentAudioRef.current = null;
          }
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

        ws.onopen = () => {
          console.log("[AETHON WS] Connected to backend telemetry");
        };

        ws.onmessage = (event) => {
          try {
            const data = jsonParseSafe(event.data);
            if (!data) return;

            if (data.type === "INIT" || data.type === "TELEMETRY") {
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
                    const mapped = data.perception.objects.map((obj) => {
                      const l = obj.label || "Object";
                      const colName = obj.color || "Neutral";
                      const disp = obj.display_name || `${colName} ${l}`;
                      return {
                        label: l,
                        display_name: disp,
                        confidence: obj.confidence || 0.9,
                        colorName: colName,
                        held: Boolean(obj.held || obj.is_held),
                        heldBy: obj.held_by || "",
                        moving: Boolean(obj.moving || obj.is_moving),
                        velocity: Number(obj.velocity || 0),
                        colorClass: getColorClass(colName || l)
                      };
                    });
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

    // A bare wake word with no command reads as a greeting, matching the
    // backend parser's behaviour.
    const toSend = captured || "hello";
    const display = normalizeAethonWakeWordRef.current
      ? normalizeAethonWakeWordRef.current(`hey aethon ${captured}`.trim())
      : `Hey AETHON, ${captured}`;

    setVoiceStatus("processing");
    if (sendCommandRef.current) {
      sendCommandRef.current(toSend, display);
    }

    wakeStateRef.current = WAKE_STATE.IDLE_LISTENING;
    setWakeState(WAKE_STATE.IDLE_LISTENING);
    setLiveTranscript("");
    setIsSpeechActive(false);

    // Keep continuous recognition running seamlessly rather than tearing it down and
    // re-requesting the audio stream every command, which crashes Chrome's speech engine after 3-4 cycles.
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
    if (useBrowserWebcam) return "Browser Webcam (Live AI Perception)";
    const found = cameras.find((c) => c.index === selectedCamera);
    return found ? found.name : `Camera ${selectedCamera} (AI Stream)`;
  }, [cameras, selectedCamera, useBrowserWebcam]);

  return (
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
              <div className="eclipse-moon" aria-label="Rotating moon">
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
                        <div className="camera-status">
                          <button
                            type="button"
                            onClick={handleToggleCamera}
                            aria-label={isCameraOn ? "Turn camera off" : "Turn camera on"}
                            title={isCameraOn ? "Click to turn camera feed OFF" : "Click to turn camera feed ON"}
                            style={{
                              background: isCameraOn ? "rgba(34, 211, 238, 0.15)" : "rgba(239, 68, 68, 0.18)",
                              border: `1px solid ${isCameraOn ? "rgba(34, 211, 238, 0.4)" : "rgba(239, 68, 68, 0.45)"}`,
                              borderRadius: 6,
                              padding: "2px 8px",
                              display: "inline-flex",
                              alignItems: "center",
                              gap: 5,
                              color: isCameraOn ? "#67e8f9" : "#fca5a5",
                              cursor: "pointer",
                              fontSize: 11,
                              fontWeight: 700,
                              letterSpacing: "0.04em",
                              transition: "all 0.18s ease"
                            }}
                          >
                            <span
                              style={{
                                width: 7,
                                height: 7,
                                borderRadius: "50%",
                                background: isCameraOn ? "#22d3ee" : "#ef4444",
                                boxShadow: isCameraOn ? "0 0 6px #22d3ee" : "0 0 6px #ef4444"
                              }}
                            />
                            {isCameraOn ? "CAM ON" : "CAM OFF"}
                          </button>
                          <StatusDot label="REC" tone={isRecording ? "white" : "white"} />
                          <span>{isCameraOn ? `${fps} FPS` : "0 FPS"}</span>
                          {isCameraOn && latencyMs !== null && (
                            <span title="Camera capture to telemetry latency">
                              {Math.round(latencyMs)} ms
                            </span>
                          )}
                          <button
                            type="button"
                            onClick={handleFullscreen}
                            aria-label="Toggle Fullscreen"
                            style={{ background: "transparent", border: 0, padding: 0, display: "inline-flex", alignItems: "center" }}
                          >
                            <Maximize2 size={17} />
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
                            Video stream paused. Click below or in the toolbar to turn camera on.
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
                      ) : useBrowserWebcam ? (
                        <>
                          <video
                            ref={videoRef}
                            autoPlay
                            playsInline
                            muted
                            style={{
                              width: "100%",
                              height: "100%",
                              objectFit: "cover",
                              transform: mirrorFeed ? "scaleX(-1)" : "none"
                            }}
                          />
                          <canvas
                            ref={overlayCanvasRef}
                            className="camera-overlay-canvas"
                            style={{
                              transform: mirrorFeed ? "scaleX(-1)" : "none"
                            }}
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
                          onClick={() => setShowCameraMenu(!showCameraMenu)}
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
                              minWidth: 260,
                              backdropFilter: "blur(12px)"
                            }}
                          >
                            {cameras.map((cam) => (
                              <button
                                key={cam.index}
                                type="button"
                                style={{
                                  display: "flex",
                                  alignItems: "center",
                                  justifyContent: "space-between",
                                  width: "100%",
                                  textAlign: "left",
                                  padding: "8px 10px",
                                  fontSize: 13,
                                  background: !useBrowserWebcam && selectedCamera === cam.index ? "rgba(34, 211, 238, 0.15)" : "transparent",
                                  border: 0,
                                  borderRadius: 4,
                                  color: !useBrowserWebcam && selectedCamera === cam.index ? "#67e8f9" : "#f1f5f7",
                                  cursor: "pointer",
                                  fontWeight: !useBrowserWebcam && selectedCamera === cam.index ? 600 : 400
                                }}
                                onClick={() => handleSelectCamera(cam.index)}
                              >
                                <span>{cam.name || `Camera ${cam.index}`}</span>
                                {!useBrowserWebcam && selectedCamera === cam.index && (
                                  <span style={{ color: "#22d3ee", fontSize: 11, fontWeight: 700 }}>● ACTIVE</span>
                                )}
                              </button>
                            ))}
                            <button
                              type="button"
                              style={{
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "space-between",
                                width: "100%",
                                textAlign: "left",
                                padding: "8px 10px",
                                fontSize: 13,
                                background: useBrowserWebcam ? "rgba(34, 211, 238, 0.15)" : "transparent",
                                border: 0,
                                borderRadius: 4,
                                color: useBrowserWebcam ? "#67e8f9" : "#f1f5f7",
                                cursor: "pointer",
                                fontWeight: useBrowserWebcam ? 600 : 400,
                                borderTop: "1px solid rgba(255, 255, 255, 0.1)"
                              }}
                              onClick={() => {
                                setUseBrowserWebcam(true);
                                setIsCameraOn(true);
                                setShowCameraMenu(false);
                              }}
                            >
                              <span>Browser Native Webcam</span>
                              {useBrowserWebcam && (
                                <span style={{ color: "#22d3ee", fontSize: 11, fontWeight: 700 }}>● ACTIVE</span>
                              )}
                            </button>
                          </div>
                        )}
                      </div>

                      <div className="camera-actions">
                        <button
                          type="button"
                          className={isCameraOn ? "active-action" : ""}
                          onClick={handleToggleCamera}
                          title={isCameraOn ? "Click to turn camera feed OFF" : "Click to turn camera feed ON"}
                          style={{
                            display: "inline-flex",
                            alignItems: "center",
                            gap: 6,
                            padding: "4px 10px",
                            borderRadius: 7,
                            border: `1px solid ${isCameraOn ? "rgba(34, 211, 238, 0.45)" : "rgba(239, 68, 68, 0.5)"}`,
                            background: isCameraOn ? "rgba(34, 211, 238, 0.14)" : "rgba(239, 68, 68, 0.18)",
                            color: isCameraOn ? "#67e8f9" : "#fca5a5",
                            fontWeight: 600,
                            cursor: "pointer",
                            transition: "all 0.18s ease"
                          }}
                        >
                          {isCameraOn ? <Camera size={15} /> : <CameraOff size={15} />}
                          <span>{isCameraOn ? "Camera On" : "Camera Off"}</span>
                        </button>
                        <button
                          type="button"
                          className={mirrorFeed ? "active-action" : ""}
                          onClick={handleToggleMirror}
                          disabled={!isCameraOn}
                          title={mirrorFeed ? "Camera feed is mirrored / inverted (Click to toggle)" : "Click to mirror / invert camera feed"}
                          style={{
                            background: mirrorFeed ? "rgba(34, 197, 94, 0.2)" : undefined,
                            borderColor: mirrorFeed ? "rgba(34, 197, 94, 0.5)" : undefined,
                            color: mirrorFeed ? "#86efac" : undefined,
                            opacity: isCameraOn ? 1 : 0.4,
                            cursor: isCameraOn ? "pointer" : "not-allowed"
                          }}
                        >
                          <FlipHorizontal size={15} /> {mirrorFeed ? "Mirrored" : "Mirror"}
                        </button>
                        <button
                          type="button"
                          onClick={handleSnapshot}
                          disabled={!isCameraOn}
                          style={{
                            opacity: isCameraOn ? 1 : 0.4,
                            cursor: isCameraOn ? "pointer" : "not-allowed"
                          }}
                        >
                          <Camera size={16} /> Snapshot
                        </button>
                        <button
                          type="button"
                          className={isRecording ? "active-action" : ""}
                          onClick={handleRecordToggle}
                          disabled={!isCameraOn && !isRecording}
                          style={{
                            opacity: isCameraOn || isRecording ? 1 : 0.4,
                            cursor: isCameraOn || isRecording ? "pointer" : "not-allowed"
                          }}
                        >
                          <span className="record-ring" /> Record
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
                    <PanelTitle title="Detected Objects" subtitle="Everyday & experiment items" />
                    <div className="object-list" style={{ maxHeight: 155, overflowY: "auto" }}>
                      {detectedObjects.length === 0 ? (
                        <div style={{ padding: "24px 12px", textAlign: "center", color: "rgba(255,255,255,0.40)", fontSize: 12, lineHeight: 1.5 }}>
                          No experiment objects in field of view<br />
                          <span style={{ fontSize: 10.5, color: "rgba(255,255,255,0.26)" }}>Position apparatus on workspace surface</span>
                        </div>
                      ) : (
                        detectedObjects.map((item, idx) => (
                          <div className="object-row" key={`${item.label}-${idx}`}>
                            <span className={`object-swatch ${item.colorClass || item.color || "object-white"}`} />
                            <div style={{ display: "flex", flexDirection: "column", flex: 1, minWidth: 0 }}>
                              <span style={{ fontWeight: 500, fontSize: 13, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                                {item.display_name || item.label}
                              </span>
                              {item.colorName && (
                                <span style={{ fontSize: 10, color: "rgba(255,255,255,0.5)", textTransform: "uppercase", letterSpacing: "0.04em" }}>
                                  Color: {item.colorName}
                                  {item.held ? ` • Held by ${item.heldBy || "hand"}` : ""}
                                  {item.moving && !item.held ? " • Moving" : ""}
                                </span>
                              )}
                            </div>
                            <strong>{item.confidence !== undefined ? item.confidence.toFixed(2) : "0.90"}</strong>
                          </div>
                        ))
                      )}
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
                        ["Person", "Human operator performing task", "object-white"],
                        ["Object A", "Red block (Pick & place item)", "object-red"],
                        ["Object B", "Wooden / Cardboard block (Base target)", "object-blue"],
                        ["Tray", "Payload holding tray", "object-purple"],
                        ["Complete Button", "Final experiment complete trigger", "object-yellow"],
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
  );
}

function jsonParseSafe(str) {
  try {
    return JSON.parse(str);
  } catch (e) {
    return null;
  }
}
