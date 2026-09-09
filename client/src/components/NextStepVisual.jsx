import React, { useMemo } from "react";

/**
 * NextStepVisual
 * 
 * Cinematic, aerospace-grade vector animation demonstrating the required next step:
 * - PICK_UP: Lifting Object A (red block) upward with guide chevrons, gripper brackets, and ground pulse.
 * - PLACE: Aligning and placing Object A onto Object B (wooden block) with descent guide and touchdown shockwave.
 * - TRAY: Stowing Object A into the apparatus tray along a curved parabolic path.
 * - PRESS: Pressing the industrial mission complete button with probe depression and radiating sonar ripples.
 * - COMPLETED: Sequence verified seal with glowing hex bezel and success checkmark.
 */

export function resolveStepType(stepNumber, label = "", stepData = null, status = "") {
  if (status === "COMPLETED") return "COMPLETED";
  const text = (label || stepData?.label || "").toLowerCase();
  const action = (stepData?.expected_action || "").toUpperCase();
  const target = (stepData?.expected_target || "").toLowerCase();

  if (action === "PRESS" || text.includes("press") || text.includes("button")) {
    return "PRESS";
  }
  if (action === "PLACE" && (target.includes("tray") || text.includes("tray"))) {
    return "TRAY";
  }
  if (action === "PLACE" || text.includes("place") || text.includes("stack") || text.includes("object b")) {
    return "PLACE";
  }
  if (action === "PICK_UP" || text.includes("pick") || text.includes("lift")) {
    return "PICK_UP";
  }

  // Fallback by step number if within standard 1..5 sequence
  if (stepNumber === 1 || stepNumber === 3) return "PICK_UP";
  if (stepNumber === 2) return "PLACE";
  if (stepNumber === 4) return "TRAY";
  if (stepNumber === 5) return "PRESS";
  return "PICK_UP";
}

export default function NextStepVisual({
  stepNumber = 1,
  totalSteps = 5,
  nextStepLabel = "",
  stepData = null,
  status = "IDLE",
}) {
  const stepType = useMemo(() => {
    return resolveStepType(stepNumber, nextStepLabel, stepData, status);
  }, [stepNumber, nextStepLabel, stepData, status]);

  const stepIndexFormatted = useMemo(() => {
    const num = String(stepNumber || 1).padStart(2, "0");
    const total = String(totalSteps || 5).padStart(2, "0");
    return `${num}/${total}`;
  }, [stepNumber, totalSteps]);

  const badgeMeta = useMemo(() => {
    switch (stepType) {
      case "PLACE":
        return { text: "STACK", color: "#38bdf8", icon: "▼" };
      case "TRAY":
        return { text: "TRAY", color: "#c084fc", icon: "➜" };
      case "PRESS":
        return { text: "PRESS", color: "#f59e0b", icon: "●" };
      case "COMPLETED":
        return { text: "DONE", color: "#10b981", icon: "✔" };
      case "PICK_UP":
      default:
        return { text: "LIFT", color: "#00e6c8", icon: "▲" };
    }
  }, [stepType]);

  return (
    <div className="next-step-visual-root" role="img" aria-label={`Next Step Guidance: ${nextStepLabel || badgeMeta.text}`}>
      <svg
        viewBox="0 0 100 100"
        className="next-step-svg"
        xmlns="http://www.w3.org/2000/svg"
      >
        <defs>
          {/* Subtle grid pattern for aerospace HUD */}
          <pattern id="nsvGrid" width="10" height="10" patternUnits="userSpaceOnUse">
            <path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(0, 230, 200, 0.05)" strokeWidth="0.5" />
          </pattern>

          {/* Gradients for Object A (Red Cube) */}
          <linearGradient id="nsvRedTop" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#f87171" />
            <stop offset="100%" stopColor="#ef4444" />
          </linearGradient>
          <linearGradient id="nsvRedLeft" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#ef4444" />
            <stop offset="100%" stopColor="#dc2626" />
          </linearGradient>
          <linearGradient id="nsvRedRight" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#b91c1c" />
            <stop offset="100%" stopColor="#991b1b" />
          </linearGradient>

          {/* Gradients for Object B (Wooden/Gold Cube) */}
          <linearGradient id="nsvWoodTop" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#fcd34d" />
            <stop offset="100%" stopColor="#fbbf24" />
          </linearGradient>
          <linearGradient id="nsvWoodLeft" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#f59e0b" />
            <stop offset="100%" stopColor="#d97706" />
          </linearGradient>
          <linearGradient id="nsvWoodRight" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#b45309" />
            <stop offset="100%" stopColor="#92400e" />
          </linearGradient>

          {/* Tray gradients */}
          <linearGradient id="nsvTrayRim" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#c084fc" />
            <stop offset="100%" stopColor="#a855f7" />
          </linearGradient>
          <linearGradient id="nsvTrayBed" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="rgba(168, 85, 247, 0.28)" />
            <stop offset="100%" stopColor="rgba(88, 28, 135, 0.5)" />
          </linearGradient>

          {/* Button gradients */}
          <linearGradient id="nsvBtnBezel" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#64748b" />
            <stop offset="100%" stopColor="#1e293b" />
          </linearGradient>
          <radialGradient id="nsvBtnPlunger" cx="45%" cy="35%" r="65%">
            <stop offset="0%" stopColor="#fef08a" />
            <stop offset="65%" stopColor="#f59e0b" />
            <stop offset="100%" stopColor="#b45309" />
          </radialGradient>

          {/* Glow filter */}
          <filter id="nsvGlow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="1.2" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Ambient Grid Background */}
        <rect width="100" height="100" fill="url(#nsvGrid)" />

        {/* Outer HUD Corner Ticks */}
        <path d="M 4 10 L 4 4 L 10 4" fill="none" stroke="rgba(0, 230, 200, 0.35)" strokeWidth="1" />
        <path d="M 96 10 L 96 4 L 90 4" fill="none" stroke="rgba(0, 230, 200, 0.35)" strokeWidth="1" />
        <path d="M 4 90 L 4 96 L 10 96" fill="none" stroke="rgba(0, 230, 200, 0.35)" strokeWidth="1" />
        <path d="M 96 90 L 96 96 L 90 96" fill="none" stroke="rgba(0, 230, 200, 0.35)" strokeWidth="1" />

        {/* Step Index HUD at top-left */}
        <text
          x="8"
          y="13"
          fill="rgba(215, 225, 230, 0.65)"
          fontSize="7.5"
          fontFamily="var(--mono)"
          fontWeight="600"
          letterSpacing="0.08em"
        >
          {status === "COMPLETED" ? "COMPLETE" : `STEP ${stepIndexFormatted}`}
        </text>

        {/* Step status radar beacon dot at top-right */}
        <circle cx="91" cy="9" r="2" fill={badgeMeta.color} filter="url(#nsvGlow)" className="nsv-beacon-dot" />

        {/* ============================================================= */}
        {/* STEP ANIMATION CONTENT                                         */}
        {/* ============================================================= */}

        {stepType === "PICK_UP" && (
          <g className="nsv-pickup-group">
            {/* Ground surface plane with cross grid */}
            <ellipse cx="50" cy="78" rx="22" ry="9" fill="rgba(0, 230, 200, 0.04)" stroke="rgba(0, 230, 200, 0.2)" strokeWidth="0.8" />
            <ellipse cx="50" cy="78" rx="14" ry="6" fill="none" stroke="rgba(0, 230, 200, 0.45)" strokeDasharray="2 2" className="nsv-ground-ring" />

            {/* Vertical trajectory dashed guide line */}
            <line x1="50" y1="74" x2="50" y2="28" stroke="#00e6c8" strokeWidth="1" strokeDasharray="3 3" className="nsv-vert-flow" />

            {/* Upward chevron motion arrows */}
            <g className="nsv-chevron-up-1">
              <path d="M 46 62 L 50 58 L 54 62" fill="none" stroke="#00e6c8" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </g>
            <g className="nsv-chevron-up-2">
              <path d="M 46 52 L 50 48 L 54 52" fill="none" stroke="#00e6c8" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </g>

            {/* Lifting Object A (Red Cube) */}
            <g className="nsv-cube-lift">
              {/* Object A Cube facets */}
              <polygon points="0,-7 12,0 0,7 -12,0" fill="url(#nsvRedTop)" stroke="rgba(255,255,255,0.4)" strokeWidth="0.6" />
              <polygon points="-12,0 0,7 0,19 -12,12" fill="url(#nsvRedLeft)" stroke="rgba(255,255,255,0.2)" strokeWidth="0.6" />
              <polygon points="0,7 12,0 12,12 0,19" fill="url(#nsvRedRight)" stroke="rgba(255,255,255,0.2)" strokeWidth="0.6" />
              <text x="0" y="3" textAnchor="middle" fontSize="6.5" fontWeight="bold" fill="#ffffff" fontFamily="var(--mono)">A</text>

              {/* Grasp / Gripper indicator brackets on cube */}
              <g className="nsv-grippers">
                <path d="M -18 0 L -15 0 L -15 12 L -18 12" fill="none" stroke="#00e6c8" strokeWidth="1.2" strokeLinecap="round" />
                <path d="M 18 0 L 15 0 L 15 12 L 18 12" fill="none" stroke="#00e6c8" strokeWidth="1.2" strokeLinecap="round" />
              </g>
            </g>
          </g>
        )}

        {stepType === "PLACE" && (
          <g className="nsv-place-group">
            {/* Ground shadow for Base Block */}
            <ellipse cx="50" cy="79" rx="19" ry="7" fill="rgba(0, 0, 0, 0.55)" />

            {/* Base Object B (Wooden/Gold Cube) fixed on table */}
            <g transform="translate(50, 56)">
              <polygon points="0,-7 13,0 0,7 -13,0" fill="url(#nsvWoodTop)" stroke="rgba(255,255,255,0.3)" strokeWidth="0.6" />
              <polygon points="-13,0 0,7 0,19 -13,12" fill="url(#nsvWoodLeft)" stroke="rgba(255,255,255,0.15)" strokeWidth="0.6" />
              <polygon points="0,7 13,0 13,13 0,19" fill="url(#nsvWoodRight)" stroke="rgba(255,255,255,0.15)" strokeWidth="0.6" />
              <text x="0" y="3" textAnchor="middle" fontSize="6.5" fontWeight="bold" fill="rgba(255,255,255,0.9)" fontFamily="var(--mono)">B</text>
            </g>

            {/* Target Docking Alignment crosshair on top of B */}
            <g transform="translate(50, 56)">
              <path d="M -9 -1 L -12 0 L -9 1" fill="none" stroke="#38bdf8" strokeWidth="0.8" />
              <path d="M 9 -1 L 12 0 L 9 1" fill="none" stroke="#38bdf8" strokeWidth="0.8" />
            </g>

            {/* Touchdown shockwave wave expanding at landing */}
            <g transform="translate(50, 56)">
              <ellipse cx="0" cy="0" rx="16" ry="7" fill="none" stroke="#00e6c8" strokeWidth="1.5" className="nsv-impact-shockwave" />
            </g>

            {/* Downward guide motion chevrons */}
            <g className="nsv-chevron-down-1">
              <path d="M 46 25 L 50 29 L 54 25" fill="none" stroke="#00e6c8" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </g>
            <g className="nsv-chevron-down-2">
              <path d="M 46 34 L 50 38 L 54 34" fill="none" stroke="#00e6c8" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </g>

            {/* Object A (Red Cube) descending onto B */}
            <g className="nsv-cube-place">
              <polygon points="0,-7 12,0 0,7 -12,0" fill="url(#nsvRedTop)" stroke="rgba(255,255,255,0.4)" strokeWidth="0.6" />
              <polygon points="-12,0 0,7 0,19 -12,12" fill="url(#nsvRedLeft)" stroke="rgba(255,255,255,0.2)" strokeWidth="0.6" />
              <polygon points="0,7 12,0 12,12 0,19" fill="url(#nsvRedRight)" stroke="rgba(255,255,255,0.2)" strokeWidth="0.6" />
              <text x="0" y="3" textAnchor="middle" fontSize="6.5" fontWeight="bold" fill="#ffffff" fontFamily="var(--mono)">A</text>
            </g>
          </g>
        )}

        {stepType === "TRAY" && (
          <g className="nsv-tray-group">
            {/* Origin table marker at bottom-left */}
            <ellipse cx="26" cy="62" rx="14" ry="6" fill="rgba(255,255,255,0.03)" stroke="rgba(255,255,255,0.15)" strokeWidth="0.8" />

            {/* Isometric Apparatus Tray at bottom-right */}
            <g transform="translate(68, 62)">
              {/* Tray base/outer rim */}
              <polygon points="0,-12 24,0 0,12 -24,0" fill="url(#nsvTrayBed)" stroke="url(#nsvTrayRim)" strokeWidth="1.4" className="nsv-tray-pulse" />
              {/* Tray inner receptacle lip */}
              <polygon points="0,-8 17,0 0,8 -17,0" fill="rgba(28, 14, 45, 0.85)" stroke="rgba(192, 132, 252, 0.4)" strokeWidth="0.8" />
              {/* Receptacle target alignment ticks */}
              <path d="M -7 0 L 7 0" stroke="rgba(192, 132, 252, 0.5)" strokeWidth="0.7" strokeDasharray="1 1" />
              {/* Tech corner status LED */}
              <circle cx="21" cy="-1" r="1.5" fill="#10b981" filter="url(#nsvGlow)" />
            </g>

            {/* Parabolic flight trajectory curve from table to tray */}
            <path
              d="M 26 42 Q 47 14 68 56"
              fill="none"
              stroke="#c084fc"
              strokeWidth="1.5"
              strokeDasharray="3 3"
              className="nsv-tray-curve"
            />
            {/* Flowing arrow head near the end */}
            <polygon points="68,54 64,48 70,49" fill="#c084fc" />

            {/* Object A (Red Cube) moving along the arc into tray */}
            <g className="nsv-cube-tray">
              <polygon points="0,-6 10,0 0,6 -10,0" fill="url(#nsvRedTop)" stroke="rgba(255,255,255,0.4)" strokeWidth="0.6" />
              <polygon points="-10,0 0,6 0,16 -10,10" fill="url(#nsvRedLeft)" stroke="rgba(255,255,255,0.2)" strokeWidth="0.6" />
              <polygon points="0,6 10,0 10,10 0,16" fill="url(#nsvRedRight)" stroke="rgba(255,255,255,0.2)" strokeWidth="0.6" />
              <text x="0" y="2.5" textAnchor="middle" fontSize="5.5" fontWeight="bold" fill="#ffffff" fontFamily="var(--mono)">A</text>
            </g>
          </g>
        )}

        {stepType === "PRESS" && (
          <g className="nsv-press-group">
            {/* Console button base platform */}
            <ellipse cx="50" cy="56" rx="28" ry="14" fill="rgba(15, 23, 42, 0.7)" stroke="rgba(255, 255, 255, 0.12)" strokeWidth="1" />

            {/* Beveled outer metallic bezel */}
            <circle cx="50" cy="54" r="22" fill="url(#nsvBtnBezel)" stroke="rgba(255, 255, 255, 0.22)" strokeWidth="1.2" />
            <circle cx="50" cy="54" r="17" fill="#0f172a" stroke="rgba(245, 158, 11, 0.35)" strokeWidth="0.8" />

            {/* Radiating sonar ripples on button press */}
            <circle cx="50" cy="54" r="14" fill="none" stroke="#00e6c8" strokeWidth="1.5" className="nsv-press-ripple-1" />
            <circle cx="50" cy="54" r="14" fill="none" stroke="#f59e0b" strokeWidth="1.2" className="nsv-press-ripple-2" />

            {/* Animated Gold Plunger (compresses when pressed) */}
            <g className="nsv-btn-plunger">
              <circle cx="50" cy="54" r="12" fill="url(#nsvBtnPlunger)" stroke="rgba(255, 255, 255, 0.4)" strokeWidth="0.8" />
              {/* Tactile concentric ring & finish power symbol */}
              <circle cx="50" cy="54" r="7" fill="none" stroke="rgba(0, 0, 0, 0.35)" strokeWidth="1" />
              <path d="M 50 49 L 50 54" stroke="#ffffff" strokeWidth="1.2" strokeLinecap="round" />
              <path d="M 47.5 51.5 A 3.5 3.5 0 1 0 52.5 51.5" fill="none" stroke="#ffffff" strokeWidth="1.2" strokeLinecap="round" />
            </g>

            {/* Automated Press Probe / Finger Reticle */}
            <g className="nsv-press-probe">
              {/* Stylized mechanical actuator descending */}
              <rect x="47" y="10" width="6" height="18" rx="2" fill="#334155" stroke="#64748b" strokeWidth="0.8" />
              <polygon points="46,28 54,28 50,34" fill="#00e6c8" />
              <circle cx="50" cy="28" r="1" fill="#ffffff" />
            </g>
          </g>
        )}

        {stepType === "COMPLETED" && (
          <g className="nsv-completed-group" transform="translate(50, 52)">
            {/* Rotating Tech HUD Hexagon / Compass Rim */}
            <g className="nsv-completed-bezel">
              <polygon
                points="0,-28 24,-14 24,14 0,28 -24,14 -24,-14"
                fill="none"
                stroke="rgba(16, 185, 129, 0.35)"
                strokeWidth="1.2"
                strokeDasharray="4 2"
              />
              <circle cx="0" cy="0" r="21" fill="none" stroke="rgba(0, 230, 200, 0.2)" strokeWidth="0.8" />
            </g>

            {/* Pulsing Success Glow Core */}
            <circle cx="0" cy="0" r="16" fill="rgba(16, 185, 129, 0.15)" stroke="#10b981" strokeWidth="1.5" className="nsv-success-core" />

            {/* Checkmark */}
            <path
              d="M -7 -1 L -2 4 L 7 -5"
              fill="none"
              stroke="#34d399"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              filter="url(#nsvGlow)"
            />
          </g>
        )}
      </svg>

      {/* Floating Action Badge at bottom-right */}
      <span
        className="next-step-badge"
        style={{
          color: badgeMeta.color,
          borderColor: `${badgeMeta.color}55`,
        }}
      >
        <span style={{ fontSize: 7, marginRight: 2 }}>{badgeMeta.icon}</span>
        {badgeMeta.text}
      </span>
    </div>
  );
}
