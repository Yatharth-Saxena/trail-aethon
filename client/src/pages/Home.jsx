import { useEffect, useMemo, useRef, useState } from "react";
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
} from "lucide-react";

const CAMERA_FEED = "/camera-feed.jpg";
const ISRO_LOGO = "/isro-logo.png";
const ASSEMBLY = "/assembly.png";
const MOON_MODEL = "/moon.glb";

const navItems = [
  { label: "Monitor", icon: Camera, active: true },
  { label: "Experiment", icon: Play, active: false },
  { label: "Logs", icon: FileText, active: false },
];

const steps = [
  { number: 1, label: "Pick up Object A", meta: "Completed  ·  10:23:12", complete: true },
  { number: 2, label: "Place Object A on Object B", meta: "In Progress...", current: true },
  { number: 3, label: "Pick up Object A again", meta: "Pending" },
  { number: 4, label: "Return Object A to Tray", meta: "Pending" },
  { number: 5, label: "Press the Complete Button", meta: "Pending" },
];

function GlassPanel({ className = "", children }) {
  return <section className={`glass-panel ${className}`}>{children}</section>;
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
  const [running, setRunning] = useState(true);
  const [activeStep, setActiveStep] = useState(2);
  const [voiceMode, setVoiceMode] = useState(true);
  const [assistantDraft, setAssistantDraft] = useState("");
  const [assistantPrompt, setAssistantPrompt] = useState("What's the next step?");

  const experimentState = running ? "In Progress" : "Paused";
  const stepProgress = useMemo(() => `${activeStep * 20}%`, [activeStep]);

  const choosePrompt = (prompt) => {
    setAssistantPrompt(prompt);
    setAssistantDraft("");
  };

  return (
    <main className="aethon-app">
      <img className="space-backdrop" src="/space-bg.png" alt="" aria-hidden="true" />
      <div className="space-vignette" aria-hidden="true" />
      <div className="shooting-stars" aria-hidden="true">
        {Array.from({ length: 10 }, (_, index) => <span key={index} className={`shooting-star star-${index + 1}`} />)}
      </div>
      <div className="noise-layer" aria-hidden="true" />

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
            <span>Mon, 12 Jan 2025</span>
            <span className="telemetry-separator" />
            <span>10:24 AM</span>
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
              <button className="nav-rail-trigger" type="button" aria-label="Open dashboard navigation"><Menu size={22} strokeWidth={1.6} /><span>MENU</span></button>
              <nav className="nav-stack" aria-label="Primary navigation">
                {navItems.map(({ label, icon: Icon, active }) => (
                  <button className={`nav-item ${active ? "active" : ""}`} key={label} type="button" onClick={() => choosePrompt(`${label} view selected`)}>
                    <Icon size={20} strokeWidth={1.65} />
                    <span>{label}</span>
                  </button>
                ))}
              </nav>
              <div className="rail-footer">
                <span className="rail-rule" />
                <p>BUILT FOR<br />A SAFER HUMAN<br />SPACEFLIGHT.</p>
                <div className="rail-orbit"><span /></div>
              </div>
            </div>
          </aside>

          <section className="workspace">
            <div className="workspace-grid">
              <GlassPanel className="camera-panel">
                <PanelTitle
                  title="Live Camera Feed"
                  subtitle="Real-time AI perception"
                  action={
                    <div className="camera-status">
                      <StatusDot label="REC" tone="white" />
                      <span>28 FPS</span>
                      <Maximize2 size={17} />
                    </div>
                  }
                />
                <div className="camera-stage" aria-label="Live camera feed" data-feed-status="live">
                  <img src={CAMERA_FEED} alt="Live camera feed" />
                  <div className="camera-shade" />
                  <div className="detect-box person-box"><span>Person 0.96</span></div>
                  <div className="detect-box hand-box"><span>Hand (R) 0.94</span></div>
                  <div className="detect-box object-a-box"><span>Object A 0.92</span></div>
                  <div className="detect-box object-b-box"><span>Object B 0.91</span></div>
                  <div className="detect-box tray-box"><span>Tray 0.87</span></div>
                  <div className="detect-box button-box"><span>Button 0.88</span></div>
                  <span className="camera-crosshair top-left" />
                  <span className="camera-crosshair bottom-right" />
                </div>
                <div className="camera-footer">
                  <button className="select-button" type="button"><span>Camera 1 (USB)</span><ChevronDown size={14} /></button>
                  <div className="camera-actions">
                    <button type="button" onClick={() => choosePrompt("Snapshot captured")}><Camera size={16} /> Snapshot</button>
                    <button type="button" className={running ? "active-action" : ""} onClick={() => setRunning(!running)}><span className="record-ring" /> Record</button>
                  </div>
                </div>
              </GlassPanel>

              <GlassPanel className="progress-panel">
                <PanelTitle title="Experiment Progress" />
                <div className="progress-summary"><strong>Step {activeStep} of 5</strong><span>{stepProgress}</span></div>
                <div className="progress-track"><span style={{ width: stepProgress }} /></div>
                <div className="step-list">
                  {steps.map((step) => (
                    <button key={step.number} type="button" className={`step-row ${step.current ? "current" : ""} ${step.complete ? "complete" : ""}`} onClick={() => setActiveStep(step.number)}>
                      <span className="step-node">{step.complete ? <Check size={15} /> : step.number}</span>
                      <span className="step-copy"><strong>{step.label}</strong><small>{step.meta}</small></span>
                    </button>
                  ))}
                </div>
              </GlassPanel>
            </div>

            <div className="context-grid">
              <GlassPanel className="detected-panel">
                <PanelTitle title="Detected Objects" />
                <div className="object-list">
                  {[
                    ["Person", "0.96", "object-white"],
                    ["Object A (Red Block)", "0.92", "object-red"],
                    ["Object B (Wooden Block)", "0.91", "object-blue"],
                    ["Tray", "0.87", "object-purple"],
                    ["Button", "0.88", "object-yellow"],
                  ].map(([name, confidence, color]) => (
                    <div className="object-row" key={name}><span className={`object-swatch ${color}`} /><span>{name}</span><strong>{confidence}</strong></div>
                  ))}
                </div>
              </GlassPanel>

              <GlassPanel className="action-panel">
                <PanelTitle title="Current Action" />
                <div className="current-action-content">
                  <div className="action-thumbnail"><span className="fake-hand" /><span className="fake-block" /></div>
                  <div className="action-copy">
                    <strong>Placing Object A</strong>
                    <span>Right hand → Object A</span>
                    <span>Action: MOVE</span>
                    <span>Confidence: 0.92</span>
                    <StatusDot label={experimentState} tone={running ? "white" : "amber"} />
                  </div>
                </div>
              </GlassPanel>

              <GlassPanel className="next-panel">
                <PanelTitle title="Next Step" />
                <div className="next-content">
                  <div className="assembly-visual"><img src={ASSEMBLY} alt="Red block positioned above a wooden block" /></div>
                  <div className="next-copy">
                    <strong>Place Object A on Object B.</strong>
                    <span>Move the red block and place it on the wooden block.</span>
                  </div>
                </div>
              </GlassPanel>
            </div>

            <GlassPanel className="experiment-controls">
              <div className="controls-lead">
                <button className="big-play-button" type="button" onClick={() => setRunning(!running)} aria-label={running ? "Pause experiment" : "Start experiment"}>
                  {running ? <Pause size={24} fill="currentColor" /> : <Play size={24} fill="currentColor" />}
                </button>
                <div className="controls-copy">
                  <strong>Experiment Controls</strong>
                </div>
              </div>

              <div className="control-buttons">
                <button className="control-button primary" type="button" onClick={() => setRunning(true)}><Play size={14} fill="currentColor" /> Start</button>
                <button className="control-button" type="button" onClick={() => setRunning(false)}><Pause size={14} /> Pause</button>
                <button className="control-button" type="button" onClick={() => { setRunning(false); setActiveStep(1); }}><RotateCcw size={14} /> Reset</button>
                <button className="control-button" type="button" onClick={() => { setRunning(false); choosePrompt("Experiment ended"); }}><Square size={13} fill="currentColor" /> End</button>
              </div>
            </GlassPanel>
          </section>

          <aside className="assistant-panel">
            <GlassPanel className="assistant-shell">
              <div className="assistant-heading">
                <div className="assistant-orb" aria-label="Rotating moon"><Moon3D className="moon-3d-large" /></div>
                <div><div className="assistant-name">AETHON</div><p>Your experiment assistant</p></div>
              </div>
              <div className="assistant-mode-row"><div className="waveform" aria-label="Aethon audio activity"><i /><i /><i /><i /><i /><i /><i /><i /></div><button type="button" className="mode-select" onClick={() => setVoiceMode(!voiceMode)}><Mic size={14} /> {voiceMode ? "Voice" : "Text"}<ChevronDown size={14} /></button></div>

              <div className="conversation">
                <div className="conversation-message user-message"><span className="message-avatar"><UserRound size={17} /></span><div className="message-bubble"><small>You <time>10:23 AM</time></small><p>Hey Aethon, start the experiment.</p></div></div>
                <div className="conversation-message assistant-message"><span className="message-avatar"><Bot size={17} /></span><div className="message-bubble"><small>Aethon <time>10:23 AM</time></small><p>Experiment started.<br />Step 1 completed.<br />You can now place Object A<br />on Object B.</p></div></div>
                <div className="conversation-message user-message"><span className="message-avatar"><UserRound size={17} /></span><div className="message-bubble"><small>You <time>10:24 AM</time></small><p>{assistantPrompt}</p></div></div>
                <div className="conversation-message assistant-message"><span className="message-avatar"><Bot size={17} /></span><div className="message-bubble"><small>Aethon <time>10:24 AM</time></small><p>The next step is:<br /><strong>Place Object A on Object B.</strong><br />Let me know if you need any help.</p></div></div>
              </div>

              <div className="assistant-suggestions">
                {["What's the next step?", "Am I doing it right?", "Repeat the procedure", "Stop the experiment"].map((prompt) => <button type="button" key={prompt} onClick={() => choosePrompt(prompt)}>{prompt}</button>)}
              </div>
              <div className="assistant-input-wrap"><button type="button" className="input-mic" onClick={() => setVoiceMode(!voiceMode)}><Mic size={18} /></button><input aria-label="Ask Aethon" value={assistantDraft} onChange={(event) => setAssistantDraft(event.target.value)} placeholder="Type or speak to Aethon..." onKeyDown={(event) => { if (event.key === "Enter") choosePrompt(assistantDraft || "Message sent"); }} /><button type="button" className="send-button" onClick={() => choosePrompt(assistantDraft || "Message sent")}><Send size={18} fill="currentColor" /></button></div>
            </GlassPanel>
          </aside>
        </div>
      </div>
    </main>
  );
}
