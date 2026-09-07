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
  Sliders
} from "lucide-react";

const ISRO_LOGO = "/isro-logo.png";
const ASSEMBLY = "/assembly.png";
const MOON_MODEL = "/moon.glb";
const API_BASE = typeof window !== "undefined" ? window.location.origin : "http://localhost:8000";
const WS_URL = typeof window !== "undefined"
  ? `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/ws`
  : "ws://localhost:8000/ws";

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
  const [isRecording, setIsRecording] = useState(false);
  const [cameras, setCameras] = useState([{ index: 0, name: "Camera 0 (USB)", active: true }]);
  const [selectedCamera, setSelectedCamera] = useState(0);
  const [showCameraMenu, setShowCameraMenu] = useState(false);
  const [streamKey, setStreamKey] = useState(Date.now());
  const [useBrowserWebcam, setUseBrowserWebcam] = useState(false);
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

    if (useBrowserWebcam) {
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
                          ctx.moveTo(lms[p1].x * scaleX, lms[p1].y * scaleY);
                          ctx.lineTo(lms[p2].x * scaleX, lms[p2].y * scaleY);
                          ctx.stroke();
                        }
                      }
                      // Joints
                      for (const lm of lms) {
                        if ((lm.visibility ?? 1) > 0.25) {
                          ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
                          ctx.beginPath();
                          ctx.arc(lm.x * scaleX, lm.y * scaleY, 3, 0, Math.PI * 2);
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
                          : `${obj.is_held ? "● HELD: " : ""}${obj.display_name || obj.label}`;
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
  }, [useBrowserWebcam]);

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
  const recognitionRef = useRef(null);
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
                if (data.camera.recording !== undefined) setIsRecording(data.camera.recording);
                if (data.camera.devices) setCameras(data.camera.devices);
                if (data.camera.device_index !== undefined) setSelectedCamera(data.camera.device_index);
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
                        colorClass: getColorClass(colName || l)
                      };
                    });
                    setDetectedObjects(mapped);
                  } else {
                    setDetectedObjects([]);
                  }
                }
              }

              if (data.conversation && data.conversation.length > 0) {
                setMessages(data.conversation);
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

  // Auto scroll chat to bottom
  useEffect(() => {
    if (chatBottomRef.current) {
      chatBottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  // Command sender
  const sendCommand = async (text) => {
    if (!text || !text.trim()) return;
    const cleanText = text.trim();
    setAssistantDraft("");

    // Optimistic user bubble
    const now = new Date();
    const timeStr = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    setMessages((prev) => [
      ...prev,
      { id: prev.length + 1, role: "user", speaker: "You", time: timeStr, text: cleanText }
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
    } catch (e) {
      console.error("[Command Send Error]", e);
    }
  };

  // Wake-word patterns
  const WAKE_WORDS = ["hey aethon", "hey ethan", "hey eaton", "aethon", "hey athena", "ok aethon", "okay aethon"];

  const stripWakeWord = (text) => {
    const lower = text.toLowerCase().trim();
    for (const ww of WAKE_WORDS) {
      if (lower.startsWith(ww)) {
        const rest = text.slice(ww.length).replace(/^[,\s.]+/, "").trim();
        return { hadWake: true, command: rest };
      }
    }
    return { hadWake: false, command: text.trim() };
  };

  // Continuous Listening – always-on voice recognition with wake-word
  const startContinuousListening = useCallback(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return;

    // Stop any existing instance
    if (recognitionRef.current) {
      try { recognitionRef.current.abort(); } catch (e) {}
    }

    const recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.continuous = true;
    recognition.maxAlternatives = 1;
    recognitionRef.current = recognition;

    recognition.onstart = () => {
      setIsListening(true);
      setVoiceStatus("listening");
    };

    recognition.onresult = (event) => {
      for (let i = event.resultIndex; i < event.results.length; i++) {
        if (event.results[i].isFinal) {
          const transcript = event.results[i][0].transcript;
          const { hadWake, command } = stripWakeWord(transcript);

          if (hadWake && command) {
            setVoiceStatus("processing");
            sendCommand(command);
            setTimeout(() => setVoiceStatus("listening"), 1500);
          } else if (hadWake && !command) {
            // Just the wake word → acknowledge
            setVoiceStatus("wake_detected");
            sendCommand("hello");
            setTimeout(() => setVoiceStatus("listening"), 1500);
          }
          // If no wake word in continuous mode, ignore (ambient noise)
        }
      }
    };

    recognition.onerror = (e) => {
      if (e.error === "not-allowed") {
        setContinuousListening(false);
        setIsListening(false);
        setVoiceStatus("idle");
        return;
      }
      // Auto-restart on transient errors
      setVoiceStatus("listening");
    };

    recognition.onend = () => {
      // Auto-restart if continuous mode is still enabled
      if (continuousListening) {
        setTimeout(() => {
          try {
            recognition.start();
          } catch (e) {
            setVoiceStatus("idle");
          }
        }, 300);
      } else {
        setIsListening(false);
        setVoiceStatus("idle");
      }
    };

    try {
      recognition.start();
    } catch (e) {
      setVoiceStatus("idle");
    }
  }, [continuousListening, sendCommand]);

  const stopContinuousListening = useCallback(() => {
    if (recognitionRef.current) {
      try { recognitionRef.current.abort(); } catch (e) {}
      recognitionRef.current = null;
    }
    setIsListening(false);
    setVoiceStatus("idle");
  }, []);

  // Effect: start/stop continuous listening when toggle changes
  useEffect(() => {
    if (continuousListening) {
      startContinuousListening();
    } else {
      stopContinuousListening();
    }
    return () => {
      if (recognitionRef.current) {
        try { recognitionRef.current.abort(); } catch (e) {}
      }
    };
  }, [continuousListening]);

  // Push-to-Talk Speech Recognition (single-shot, for when continuous mode is OFF)
  const handleMicClick = async () => {
    if (continuousListening) {
      // In continuous mode, mic button toggles it off
      setContinuousListening(false);
      return;
    }

    if (isListening) {
      if (recognitionRef.current) {
        try { recognitionRef.current.abort(); } catch (e) {}
      }
      setIsListening(false);
      setVoiceStatus("idle");
      return;
    }

    setIsListening(true);
    setVoiceStatus("listening");

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.lang = "en-US";
      recognition.interimResults = false;
      recognition.maxAlternatives = 1;
      recognitionRef.current = recognition;

      recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setIsListening(false);
        setVoiceStatus("processing");
        // In push-to-talk, send directly (no wake word needed)
        const { command } = stripWakeWord(transcript);
        sendCommand(command || transcript);
        setTimeout(() => setVoiceStatus("idle"), 1500);
      };

      recognition.onerror = () => {
        setIsListening(false);
        setVoiceStatus("idle");
      };

      recognition.onend = () => {
        setIsListening(false);
        setVoiceStatus("idle");
      };

      try {
        recognition.start();
      } catch (err) {
        setIsListening(false);
        setVoiceStatus("idle");
      }
    } else {
      const promptCmd = prompt("Enter voice command:", "What's the next step?");
      setIsListening(false);
      setVoiceStatus("idle");
      if (promptCmd) {
        sendCommand(promptCmd);
      }
    }
  };

  // Camera action handlers
  const handleSnapshot = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/camera/snapshot`, { method: "POST" });
      const data = await res.json();
      sendCommand(`Snapshot captured: ${data.filename || "saved"}`);
    } catch (e) {
      sendCommand("Snapshot captured");
    }
  };

  const handleRecordToggle = async () => {
    try {
      const endpoint = isRecording ? "/api/camera/record/stop" : "/api/camera/record/start";
      const res = await fetch(`${API_BASE}${endpoint}`, { method: "POST" });
      const data = await res.json();
      setIsRecording(!isRecording);
      sendCommand(isRecording ? "Recording stopped and saved locally." : "Recording started.");
    } catch (e) {
      setIsRecording(!isRecording);
    }
  };

  const handleSelectCamera = async (index) => {
    setSelectedCamera(index);
    setShowCameraMenu(false);
    try {
      await fetch(`${API_BASE}/api/camera/select`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ index })
      });
    } catch (e) {}
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
    } catch (e) {}
  };

  const handlePause = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/experiment/pause`, { method: "POST" });
      const data = await res.json();
      if (data.state) setExperimentState(data.state);
    } catch (e) {}
  };

  const handleResume = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/experiment/resume`, { method: "POST" });
      const data = await res.json();
      if (data.state) setExperimentState(data.state);
    } catch (e) {}
  };

  const handleReset = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/experiment/reset`, { method: "POST" });
      const data = await res.json();
      if (data.state) setExperimentState(data.state);
    } catch (e) {}
  };

  const handleStop = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/experiment/stop`, { method: "POST" });
      const data = await res.json();
      if (data.state) setExperimentState(data.state);
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
                      subtitle="Real-time AI perception"
                      action={
                        <div className="camera-status">
                          <StatusDot label="REC" tone={isRecording ? "white" : "white"} />
                          <span>{fps} FPS</span>
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
                      data-feed-status="live"
                    >
                      {useBrowserWebcam ? (
                        <>
                          <video
                            ref={videoRef}
                            autoPlay
                            playsInline
                            muted
                            style={{ width: "100%", height: "100%", objectFit: "cover" }}
                          />
                          <canvas
                            ref={overlayCanvasRef}
                            className="camera-overlay-canvas"
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
                          onClick={() => setStreamKey(Date.now())}
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
                              minWidth: 230,
                              backdropFilter: "blur(12px)"
                            }}
                          >
                            <button
                              type="button"
                              style={{
                                display: "block",
                                width: "100%",
                                textAlign: "left",
                                padding: "7px 10px",
                                fontSize: 13,
                                background: !useBrowserWebcam ? "rgba(255, 255, 255, 0.15)" : "transparent",
                                border: 0,
                                borderRadius: 4,
                                color: "#f1f5f7",
                                cursor: "pointer"
                              }}
                              onClick={() => {
                                setUseBrowserWebcam(false);
                                handleSelectCamera(0);
                                setStreamKey(Date.now());
                                setShowCameraMenu(false);
                              }}
                            >
                              Camera 0: Integrated/USB (AI Pipeline)
                            </button>
                            <button
                              type="button"
                              style={{
                                display: "block",
                                width: "100%",
                                textAlign: "left",
                                padding: "7px 10px",
                                fontSize: 13,
                                background: useBrowserWebcam ? "rgba(255, 255, 255, 0.15)" : "transparent",
                                border: 0,
                                borderRadius: 4,
                                color: "#f1f5f7",
                                cursor: "pointer"
                              }}
                              onClick={() => {
                                setUseBrowserWebcam(true);
                                setShowCameraMenu(false);
                              }}
                            >
                              Browser Native Webcam (Live AI Perception)
                            </button>
                          </div>
                        )}
                      </div>

                      <div className="camera-actions">
                        <button type="button" onClick={handleSnapshot}>
                          <Camera size={16} /> Snapshot
                        </button>
                        <button
                          type="button"
                          className={isRecording ? "active-action" : ""}
                          onClick={handleRecordToggle}
                        >
                          <span className="record-ring" /> Record
                        </button>
                      </div>
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
                                  Color: {item.colorName} {item.held ? `• Held by ${item.heldBy || "hand"}` : ""}
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
                  <p>Your experiment assistant</p>
                </div>
              </div>

              <div className="assistant-mode-row">
                <div className={`waveform ${isListening ? "listening" : ""}`} aria-label="Aethon audio activity">
                  <i /><i /><i /><i /><i /><i /><i /><i />
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  <button
                    type="button"
                    className={`mode-select ${continuousListening ? "active" : ""}`}
                    onClick={() => setContinuousListening(!continuousListening)}
                    title={continuousListening ? "Stop always listening" : "Enable always-on voice (say 'Hey AETHON')"}
                    style={{
                      background: continuousListening ? "rgba(34, 197, 94, 0.25)" : undefined,
                      border: continuousListening ? "1px solid rgba(34, 197, 94, 0.5)" : undefined,
                    }}
                  >
                    <Mic size={14} style={{ color: continuousListening ? "#22c55e" : undefined }} />
                    {continuousListening ? "Always On" : "Voice Off"}
                  </button>
                  {voiceStatus !== "idle" && (
                    <span style={{
                      fontSize: "0.68rem",
                      color: voiceStatus === "listening" ? "#22c55e" : voiceStatus === "wake_detected" ? "#eab308" : "#3b82f6",
                      textTransform: "uppercase",
                      letterSpacing: "0.08em",
                      animation: "pulse 1.5s ease-in-out infinite"
                    }}>
                      {voiceStatus === "listening" ? "● Listening..." : voiceStatus === "wake_detected" ? "★ Wake Detected" : "⟳ Processing"}
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
                  "What am I doing?",
                  "What is this object?",
                  "What color is this?",
                  "What are my movements?",
                  "What's the next step?",
                  "Am I doing it right?"
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

              <div className="assistant-input-wrap">
                <button
                  type="button"
                  className={`input-mic ${isListening ? "active" : ""}`}
                  style={{
                    color: isListening
                      ? continuousListening ? "#22c55e" : "#ef4444"
                      : "inherit",
                    animation: isListening ? "pulse 1.5s ease-in-out infinite" : "none"
                  }}
                  onClick={handleMicClick}
                  title={continuousListening ? "Stop always listening" : "Push to talk"}
                >
                  <Mic size={18} />
                </button>
                <input
                  aria-label="Ask Aethon"
                  value={assistantDraft}
                  onChange={(e) => setAssistantDraft(e.target.value)}
                  placeholder={continuousListening ? "Say 'Hey AETHON' or type..." : "Type or speak to Aethon..."}
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
