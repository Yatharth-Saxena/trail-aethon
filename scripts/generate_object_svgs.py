"""
Generates clean, aerospace-styled SVG graphics for objects detected by AETHON.
Saves them in client/public/objects/
"""
from pathlib import Path

OUT_DIR = Path("/Users/joshi2008/Desktop/trail-aethon/client/public/objects")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SVGS = {}

# 1. Phone
SVGS["phone.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <defs>
    <linearGradient id="body" x1="20" y1="10" x2="100" y2="110" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#2d3748"/>
      <stop offset="100%" stop-color="#0f172a"/>
    </linearGradient>
    <linearGradient id="screen" x1="34" y1="18" x2="86" y2="102" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#0284c7"/>
      <stop offset="40%" stop-color="#0369a1"/>
      <stop offset="100%" stop-color="#082f49"/>
    </linearGradient>
    <linearGradient id="glass" x1="30" y1="15" x2="90" y2="105" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#ffffff" stop-opacity="0.35"/>
      <stop offset="50%" stop-color="#ffffff" stop-opacity="0.05"/>
      <stop offset="100%" stop-color="#ffffff" stop-opacity="0"/>
    </linearGradient>
  </defs>
  <!-- Outer Body -->
  <rect x="30" y="12" width="60" height="96" rx="14" fill="url(#body)" stroke="#38bdf8" stroke-width="2"/>
  <!-- Screen -->
  <rect x="34" y="16" width="52" height="88" rx="10" fill="url(#screen)"/>
  <!-- Glass Reflection Sheen -->
  <path d="M34 26 L86 16 L86 45 L34 75 Z" fill="url(#glass)"/>
  <!-- Speaker / Dynamic Island -->
  <rect x="50" y="20" width="20" height="5" rx="2.5" fill="#030712"/>
  <circle cx="66" cy="22.5" r="1.5" fill="#0284c7"/>
  <!-- Screen HUD elements -->
  <circle cx="60" cy="54" r="14" stroke="rgba(255,255,255,0.4)" stroke-dasharray="3 3" stroke-width="1.5" fill="none"/>
  <path d="M54 54 L66 54 M60 48 L60 60" stroke="#38bdf8" stroke-width="1.5" stroke-linecap="round"/>
  <rect x="42" y="76" width="36" height="4" rx="2" fill="rgba(255,255,255,0.25)"/>
  <rect x="46" y="83" width="28" height="3" rx="1.5" fill="rgba(255,255,255,0.15)"/>
  <!-- Home Indicator -->
  <rect x="48" y="97" width="24" height="2.5" rx="1.25" fill="rgba(255,255,255,0.7)"/>
</svg>"""

# 2. Bottle
SVGS["bottle.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <defs>
    <linearGradient id="flask" x1="38" y1="20" x2="82" y2="110" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#0284c7"/>
      <stop offset="60%" stop-color="#0369a1"/>
      <stop offset="100%" stop-color="#0c4a6e"/>
    </linearGradient>
    <linearGradient id="cap" x1="48" y1="12" x2="72" y2="28" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#94a3b8"/>
      <stop offset="100%" stop-color="#475569"/>
    </linearGradient>
  </defs>
  <!-- Cap -->
  <rect x="50" y="12" width="20" height="14" rx="3" fill="url(#cap)" stroke="#cbd5e1" stroke-width="1.5"/>
  <line x1="53" y1="17" x2="67" y2="17" stroke="#334155" stroke-width="1"/>
  <line x1="53" y1="21" x2="67" y2="21" stroke="#334155" stroke-width="1"/>
  <!-- Neck -->
  <path d="M53 26 L67 26 L67 34 L53 34 Z" fill="#64748b"/>
  <!-- Bottle Body -->
  <path d="M53 34 C44 40 40 48 40 60 L40 98 C40 104 45 108 52 108 L68 108 C75 108 80 104 80 98 L80 60 C80 48 76 40 67 34 Z" fill="url(#flask)" stroke="#38bdf8" stroke-width="2"/>
  <!-- Measurement ticks -->
  <line x1="68" y1="55" x2="74" y2="55" stroke="rgba(255,255,255,0.6)" stroke-width="1.5"/>
  <line x1="70" y1="65" x2="74" y2="65" stroke="rgba(255,255,255,0.6)" stroke-width="1.5"/>
  <line x1="66" y1="75" x2="74" y2="75" stroke="rgba(255,255,255,0.6)" stroke-width="1.5"/>
  <line x1="70" y1="85" x2="74" y2="85" stroke="rgba(255,255,255,0.6)" stroke-width="1.5"/>
  <!-- Liquid highlight -->
  <path d="M44 65 C48 68 54 62 60 65 C66 68 72 62 76 65 L76 98 C76 102 72 104 68 104 L52 104 C48 104 44 102 44 98 Z" fill="#00e6c8" fill-opacity="0.25"/>
</svg>"""

# 3. Cup / Mug
SVGS["cup.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <defs>
    <linearGradient id="mug" x1="28" y1="36" x2="84" y2="104" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#059669"/>
      <stop offset="100%" stop-color="#064e3b"/>
    </linearGradient>
  </defs>
  <!-- Steam -->
  <path d="M44 26 C42 20 48 16 46 12" stroke="#6ee7b7" stroke-width="2" stroke-linecap="round" stroke-dasharray="3 3"/>
  <path d="M56 24 C54 18 60 14 58 10" stroke="#6ee7b7" stroke-width="2" stroke-linecap="round" stroke-dasharray="3 3"/>
  <path d="M68 26 C66 20 72 16 70 12" stroke="#6ee7b7" stroke-width="2" stroke-linecap="round" stroke-dasharray="3 3"/>
  <!-- Handle -->
  <path d="M78 48 C94 48 94 82 78 84" fill="none" stroke="#34d399" stroke-width="6" stroke-linecap="round"/>
  <!-- Mug Body -->
  <path d="M30 36 L82 36 L76 96 C76 101 70 106 63 106 L49 106 C42 106 36 101 36 96 Z" fill="url(#mug)" stroke="#10b981" stroke-width="2"/>
  <!-- Thermal Rim -->
  <ellipse cx="56" cy="36" rx="26" ry="6" fill="#065f46" stroke="#34d399" stroke-width="2"/>
  <ellipse cx="56" cy="36" rx="20" ry="4" fill="#312e81"/>
  <!-- Emblem -->
  <circle cx="56" cy="68" r="8" fill="none" stroke="#a7f3d0" stroke-width="1.5"/>
  <polygon points="56,63 58.5,67.5 63,68 59.5,71.5 60.5,76 56,73.5 51.5,76 52.5,71.5 49,68 53.5,67.5" fill="#34d399"/>
</svg>"""

# 4. Laptop
SVGS["laptop.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <defs>
    <linearGradient id="lapScreen" x1="30" y1="20" x2="90" y2="76" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#0f172a"/>
      <stop offset="100%" stop-color="#0284c7"/>
    </linearGradient>
  </defs>
  <!-- Lid -->
  <rect x="24" y="20" width="72" height="52" rx="4" fill="#1e293b" stroke="#38bdf8" stroke-width="1.5"/>
  <!-- Display Area -->
  <rect x="28" y="24" width="64" height="44" rx="2" fill="url(#lapScreen)"/>
  <!-- Screen Graphic (Telemetry graph) -->
  <path d="M34 56 L46 44 L54 50 L64 36 L72 44 L86 34" stroke="#00e6c8" stroke-width="2" fill="none" stroke-linecap="round"/>
  <circle cx="64" cy="36" r="2.5" fill="#00e6c8"/>
  <circle cx="86" cy="34" r="2.5" fill="#38bdf8"/>
  <!-- Base -->
  <path d="M14 76 L106 76 L100 94 C99 96 97 98 94 98 L26 98 C23 98 21 96 20 94 Z" fill="#334155" stroke="#94a3b8" stroke-width="1.5"/>
  <!-- Trackpad -->
  <rect x="50" y="86" width="20" height="9" rx="1.5" fill="#1e293b" stroke="#475569" stroke-width="1"/>
  <!-- Keyboard area -->
  <rect x="26" y="79" width="68" height="5" rx="1" fill="#0f172a"/>
</svg>"""

# 5. Mouse
SVGS["mouse.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <defs>
    <linearGradient id="mouseGrad" x1="40" y1="18" x2="80" y2="102" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#334155"/>
      <stop offset="100%" stop-color="#0f172a"/>
    </linearGradient>
  </defs>
  <!-- Mouse Body -->
  <path d="M40 44 C40 26 50 16 60 16 C70 16 80 26 80 44 L80 74 C80 92 70 104 60 104 C50 104 40 92 40 74 Z" fill="url(#mouseGrad)" stroke="#38bdf8" stroke-width="2"/>
  <!-- Split line -->
  <line x1="60" y1="16" x2="60" y2="48" stroke="#1e293b" stroke-width="2"/>
  <!-- Scroll Wheel -->
  <rect x="57" y="28" width="6" height="14" rx="3" fill="#00e6c8" stroke="#0284c7" stroke-width="1"/>
  <!-- Accent LED lines -->
  <path d="M48 64 C48 76 53 84 60 84 C67 84 72 76 72 64" fill="none" stroke="rgba(0,230,200,0.5)" stroke-width="1.5"/>
</svg>"""

# 6. Keyboard
SVGS["keyboard.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <!-- Body -->
  <rect x="14" y="34" width="92" height="52" rx="6" fill="#1e293b" stroke="#38bdf8" stroke-width="2"/>
  <!-- Key rows -->
  <g fill="#334155" stroke="#475569" stroke-width="0.75">
    <!-- Row 1 -->
    <rect x="20" y="40" width="8" height="6" rx="1"/>
    <rect x="31" y="40" width="8" height="6" rx="1"/>
    <rect x="42" y="40" width="8" height="6" rx="1"/>
    <rect x="53" y="40" width="8" height="6" rx="1"/>
    <rect x="64" y="40" width="8" height="6" rx="1"/>
    <rect x="75" y="40" width="8" height="6" rx="1"/>
    <rect x="86" y="40" width="14" height="6" rx="1" fill="#0284c7"/>
    <!-- Row 2 -->
    <rect x="20" y="49" width="12" height="6" rx="1"/>
    <rect x="35" y="49" width="8" height="6" rx="1"/>
    <rect x="46" y="49" width="8" height="6" rx="1"/>
    <rect x="57" y="49" width="8" height="6" rx="1"/>
    <rect x="68" y="49" width="8" height="6" rx="1"/>
    <rect x="79" y="49" width="8" height="6" rx="1"/>
    <rect x="90" y="49" width="10" height="6" rx="1"/>
    <!-- Row 3 -->
    <rect x="20" y="58" width="14" height="6" rx="1"/>
    <rect x="37" y="58" width="8" height="6" rx="1"/>
    <rect x="48" y="58" width="8" height="6" rx="1" fill="#00e6c8"/>
    <rect x="59" y="58" width="8" height="6" rx="1"/>
    <rect x="70" y="58" width="8" height="6" rx="1"/>
    <rect x="81" y="58" width="19" height="6" rx="1" fill="#0284c7"/>
    <!-- Row 4 -->
    <rect x="20" y="67" width="16" height="6" rx="1"/>
    <rect x="39" y="67" width="42" height="6" rx="1.5" fill="#475569"/>
    <rect x="84" y="67" width="16" height="6" rx="1"/>
  </g>
  <!-- Indicator dots -->
  <circle cx="94" cy="78" r="1.5" fill="#00e6c8"/>
  <circle cx="99" cy="78" r="1.5" fill="#38bdf8"/>
</svg>"""

# 7. Scissors
SVGS["scissors.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <!-- Blades -->
  <path d="M40 48 L86 16 C90 14 92 18 88 22 L54 62 Z" fill="#cbd5e1" stroke="#94a3b8" stroke-width="1.5"/>
  <path d="M40 72 L86 104 C90 106 92 102 88 98 L54 58 Z" fill="#94a3b8" stroke="#cbd5e1" stroke-width="1.5"/>
  <!-- Pivot Screw -->
  <circle cx="52" cy="60" r="4.5" fill="#f59e0b" stroke="#d97706" stroke-width="1.5"/>
  <circle cx="52" cy="60" r="1.5" fill="#1e293b"/>
  <!-- Handles -->
  <circle cx="30" cy="46" r="14" fill="none" stroke="#f59e0b" stroke-width="5"/>
  <circle cx="30" cy="74" r="14" fill="none" stroke="#f59e0b" stroke-width="5"/>
</svg>"""

# 8. Book
SVGS["book.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <defs>
    <linearGradient id="cover" x1="24" y1="20" x2="96" y2="100" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#4338ca"/>
      <stop offset="100%" stop-color="#1e1b4b"/>
    </linearGradient>
  </defs>
  <!-- Book Cover -->
  <rect x="28" y="16" width="64" height="88" rx="4" fill="url(#cover)" stroke="#818cf8" stroke-width="2"/>
  <!-- Spine crease -->
  <line x1="38" y1="16" x2="38" y2="104" stroke="#6366f1" stroke-width="2"/>
  <!-- Pages edge -->
  <rect x="88" y="20" width="5" height="80" fill="#e2e8f0" stroke="#94a3b8" stroke-width="1"/>
  <!-- Bookmark ribbon -->
  <path d="M52 16 L52 46 L58 40 L64 46 L64 16 Z" fill="#f43f5e"/>
  <!-- Emblem on cover -->
  <circle cx="64" cy="60" r="14" fill="none" stroke="#a5b4fc" stroke-width="1.5"/>
  <path d="M64 52 L64 68 M56 60 L72 60" stroke="#a5b4fc" stroke-width="1.5"/>
  <rect x="48" y="80" width="32" height="3" rx="1.5" fill="#c7d2fe"/>
  <rect x="52" y="86" width="24" height="2" rx="1" fill="#a5b4fc"/>
</svg>"""

# 9. Clock
SVGS["clock.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <defs>
    <linearGradient id="clockFace" x1="20" y1="20" x2="100" y2="100" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#1e293b"/>
      <stop offset="100%" stop-color="#020617"/>
    </linearGradient>
  </defs>
  <!-- Rim -->
  <circle cx="60" cy="60" r="46" fill="url(#clockFace)" stroke="#38bdf8" stroke-width="3"/>
  <circle cx="60" cy="60" r="41" stroke="rgba(56, 189, 248, 0.3)" stroke-width="1" fill="none"/>
  <!-- Hour ticks -->
  <line x1="60" y1="23" x2="60" y2="28" stroke="#00e6c8" stroke-width="2.5" stroke-linecap="round"/>
  <line x1="60" y1="92" x2="60" y2="97" stroke="#00e6c8" stroke-width="2.5" stroke-linecap="round"/>
  <line x1="23" y1="60" x2="28" y2="60" stroke="#00e6c8" stroke-width="2.5" stroke-linecap="round"/>
  <line x1="92" y1="60" x2="97" y2="60" stroke="#00e6c8" stroke-width="2.5" stroke-linecap="round"/>
  <!-- Hands -->
  <line x1="60" y1="60" x2="60" y2="36" stroke="#ffffff" stroke-width="3" stroke-linecap="round"/>
  <line x1="60" y1="60" x2="78" y2="60" stroke="#38bdf8" stroke-width="2.5" stroke-linecap="round"/>
  <line x1="60" y1="64" x2="48" y2="76" stroke="#f43f5e" stroke-width="1.5" stroke-linecap="round"/>
  <!-- Center Pin -->
  <circle cx="60" cy="60" r="3.5" fill="#f43f5e"/>
</svg>"""

# 10. Remote
SVGS["remote.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <rect x="42" y="14" width="36" height="92" rx="10" fill="#1e293b" stroke="#64748b" stroke-width="2"/>
  <circle cx="60" cy="24" r="3" fill="#ef4444"/>
  <!-- D-pad -->
  <circle cx="60" cy="46" r="12" fill="#0f172a" stroke="#38bdf8" stroke-width="1.5"/>
  <polygon points="60,37 57,41 63,41" fill="#38bdf8"/>
  <polygon points="60,55 57,51 63,51" fill="#38bdf8"/>
  <polygon points="51,46 55,43 55,49" fill="#38bdf8"/>
  <polygon points="69,46 65,43 65,49" fill="#38bdf8"/>
  <!-- Number buttons -->
  <circle cx="51" cy="66" r="3" fill="#334155"/>
  <circle cx="60" cy="66" r="3" fill="#334155"/>
  <circle cx="69" cy="66" r="3" fill="#334155"/>
  <circle cx="51" cy="76" r="3" fill="#334155"/>
  <circle cx="60" cy="76" r="3" fill="#334155"/>
  <circle cx="69" cy="76" r="3" fill="#334155"/>
  <circle cx="51" cy="86" r="3" fill="#334155"/>
  <circle cx="60" cy="86" r="3" fill="#334155"/>
  <circle cx="69" cy="86" r="3" fill="#334155"/>
  <rect x="52" y="94" width="16" height="4" rx="2" fill="#0284c7"/>
</svg>"""

# 11. Pen
SVGS["pen.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <!-- Pen Body angled -->
  <g transform="rotate(45 60 60)">
    <rect x="56" y="24" width="8" height="68" rx="2" fill="#38bdf8" stroke="#0284c7" stroke-width="1.5"/>
    <path d="M56 92 L64 92 L60 106 Z" fill="#cbd5e1" stroke="#64748b" stroke-width="1"/>
    <circle cx="60" cy="105" r="1" fill="#0f172a"/>
    <rect x="55" y="20" width="10" height="6" rx="1.5" fill="#94a3b8"/>
    <!-- Clip -->
    <rect x="64" y="26" width="3" height="24" rx="1" fill="#cbd5e1"/>
  </g>
</svg>"""

# 12. Bowl
SVGS["bowl.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <defs>
    <linearGradient id="bowlGrad" x1="20" y1="40" x2="100" y2="94" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#0284c7"/>
      <stop offset="100%" stop-color="#0c4a6e"/>
    </linearGradient>
  </defs>
  <!-- Bowl Base -->
  <path d="M22 50 C22 84 40 96 60 96 C80 96 98 84 98 50 Z" fill="url(#bowlGrad)" stroke="#38bdf8" stroke-width="2"/>
  <ellipse cx="60" cy="50" rx="38" ry="12" fill="#0369a1" stroke="#7dd3fc" stroke-width="2"/>
  <ellipse cx="60" cy="50" rx="30" ry="8" fill="#082f49"/>
  <rect x="46" y="96" width="28" height="6" rx="2" fill="#0369a1" stroke="#38bdf8" stroke-width="1"/>
</svg>"""

# 13. Knife
SVGS["knife.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <g transform="rotate(-35 60 60)">
    <!-- Blade -->
    <path d="M57 16 C60 16 63 30 63 60 L57 60 Z" fill="#e2e8f0" stroke="#94a3b8" stroke-width="1.5"/>
    <!-- Guard -->
    <rect x="53" y="60" width="14" height="4" rx="1" fill="#f59e0b"/>
    <!-- Handle -->
    <rect x="56" y="64" width="8" height="40" rx="3" fill="#1e293b" stroke="#475569" stroke-width="1.5"/>
    <circle cx="60" cy="74" r="1.5" fill="#94a3b8"/>
    <circle cx="60" cy="84" r="1.5" fill="#94a3b8"/>
    <circle cx="60" cy="94" r="1.5" fill="#94a3b8"/>
  </g>
</svg>"""

# 14. Fork
SVGS["fork.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <g transform="rotate(25 60 60)">
    <!-- Tines -->
    <path d="M50 20 L50 44 C50 52 56 56 58 56 L58 100 C58 102 62 102 62 100 L62 56 C64 56 70 52 70 44 L70 20" stroke="#cbd5e1" stroke-width="2.5" fill="none" stroke-linecap="round"/>
    <line x1="56" y1="20" x2="56" y2="44" stroke="#cbd5e1" stroke-width="2" stroke-linecap="round"/>
    <line x1="64" y1="20" x2="64" y2="44" stroke="#cbd5e1" stroke-width="2" stroke-linecap="round"/>
  </g>
</svg>"""

# 15. Spoon
SVGS["spoon.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <g transform="rotate(30 60 60)">
    <ellipse cx="60" cy="34" rx="14" ry="20" fill="#94a3b8" stroke="#e2e8f0" stroke-width="2"/>
    <path d="M58 52 L58 102 C58 105 62 105 62 102 L62 52" fill="#cbd5e1" stroke="#e2e8f0" stroke-width="2" stroke-linecap="round"/>
  </g>
</svg>"""

# 16. Chair
SVGS["chair.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <!-- Backrest -->
  <rect x="42" y="16" width="36" height="42" rx="6" fill="#1e293b" stroke="#38bdf8" stroke-width="2"/>
  <rect x="48" y="24" width="24" height="26" rx="3" fill="#0f172a" stroke="#0284c7" stroke-width="1"/>
  <!-- Seat -->
  <rect x="34" y="60" width="52" height="12" rx="4" fill="#334155" stroke="#38bdf8" stroke-width="2"/>
  <!-- Center Stem -->
  <rect x="57" y="72" width="6" height="20" fill="#64748b"/>
  <!-- Base legs -->
  <path d="M30 102 L60 92 L90 102" stroke="#64748b" stroke-width="3" stroke-linecap="round"/>
  <circle cx="30" cy="103" r="2.5" fill="#38bdf8"/>
  <circle cx="60" cy="93" r="2.5" fill="#38bdf8"/>
  <circle cx="90" cy="103" r="2.5" fill="#38bdf8"/>
</svg>"""

# 17. Monitor
SVGS["monitor.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <rect x="18" y="20" width="84" height="54" rx="4" fill="#0f172a" stroke="#38bdf8" stroke-width="2"/>
  <rect x="22" y="24" width="76" height="46" rx="2" fill="#0369a1"/>
  <path d="M26 62 L44 46 L58 54 L74 38 L88 48" stroke="#00e6c8" stroke-width="2" fill="none" stroke-linecap="round"/>
  <!-- Stand -->
  <rect x="56" y="74" width="8" height="18" fill="#475569"/>
  <rect x="42" y="92" width="36" height="6" rx="3" fill="#334155" stroke="#64748b" stroke-width="1.5"/>
</svg>"""

# 18. Backpack
SVGS["backpack.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <!-- Body -->
  <path d="M36 40 C36 24 48 18 60 18 C72 18 84 24 84 40 L84 94 C84 100 78 104 70 104 L50 104 C42 104 36 100 36 94 Z" fill="#0f766e" stroke="#14b8a6" stroke-width="2"/>
  <!-- Front Pocket -->
  <rect x="42" y="56" width="36" height="38" rx="6" fill="#115e59" stroke="#2dd4bf" stroke-width="1.5"/>
  <line x1="46" y1="68" x2="74" y2="68" stroke="#5eead4" stroke-width="1.5"/>
  <!-- Straps / handle -->
  <path d="M50 18 C50 12 70 12 70 18" stroke="#0d9488" stroke-width="3" fill="none"/>
</svg>"""

# 19. Bag
SVGS["bag.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <rect x="30" y="44" width="60" height="56" rx="8" fill="#475569" stroke="#94a3b8" stroke-width="2"/>
  <!-- Handles -->
  <path d="M46 44 C46 26 74 26 74 44" fill="none" stroke="#cbd5e1" stroke-width="3"/>
  <line x1="30" y1="64" x2="90" y2="64" stroke="#64748b" stroke-width="2"/>
</svg>"""

# 20. Umbrella
SVGS["umbrella.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <path d="M22 62 C22 36 38 20 60 20 C82 20 98 36 98 62 Z" fill="#0284c7" stroke="#38bdf8" stroke-width="2"/>
  <line x1="60" y1="20" x2="60" y2="92" stroke="#94a3b8" stroke-width="3"/>
  <path d="M60 92 C60 100 52 104 46 98" fill="none" stroke="#94a3b8" stroke-width="3" stroke-linecap="round"/>
</svg>"""

# 21. Plant
SVGS["plant.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <!-- Leaves -->
  <path d="M60 62 C50 46 36 44 32 50 C28 56 36 68 60 62 Z" fill="#10b981" stroke="#059669" stroke-width="1.5"/>
  <path d="M60 58 C68 40 84 38 88 44 C92 50 82 62 60 58 Z" fill="#34d399" stroke="#059669" stroke-width="1.5"/>
  <path d="M60 52 C56 32 64 20 68 22 C72 24 70 38 60 52 Z" fill="#6ee7b7" stroke="#059669" stroke-width="1.5"/>
  <!-- Pot -->
  <path d="M42 66 L78 66 L72 102 L48 102 Z" fill="#d97706" stroke="#b45309" stroke-width="2"/>
  <rect x="38" y="62" width="44" height="6" rx="2" fill="#f59e0b" stroke="#b45309" stroke-width="1.5"/>
</svg>"""

# 22. Apple
SVGS["apple.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <path d="M60 38 C42 22 24 44 26 68 C28 88 44 104 60 104 C76 104 92 88 94 68 C96 44 78 22 60 38 Z" fill="#dc2626" stroke="#b91c1c" stroke-width="2"/>
  <!-- Stem -->
  <path d="M60 38 C60 26 66 18 72 16" fill="none" stroke="#78350f" stroke-width="3" stroke-linecap="round"/>
  <!-- Leaf -->
  <path d="M68 24 C76 20 84 22 86 28 C84 32 76 30 68 24 Z" fill="#16a34a"/>
</svg>"""

# 23. Banana
SVGS["banana.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <path d="M30 32 C48 64 74 88 100 78 C80 94 48 86 22 52 C18 46 22 34 30 32 Z" fill="#eab308" stroke="#ca8a04" stroke-width="2"/>
  <!-- Tip -->
  <circle cx="23" cy="46" r="3" fill="#713f12"/>
</svg>"""

# 24. Orange
SVGS["orange.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <circle cx="60" cy="62" r="38" fill="#ea580c" stroke="#c2410c" stroke-width="2"/>
  <circle cx="60" cy="26" r="2.5" fill="#78350f"/>
  <path d="M60 26 C68 20 78 22 80 28 C78 32 70 30 60 26 Z" fill="#16a34a"/>
</svg>"""

# 25. Pizza
SVGS["pizza.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <path d="M60 102 L20 34 C36 24 84 24 100 34 Z" fill="#f59e0b" stroke="#d97706" stroke-width="2"/>
  <!-- Crust -->
  <path d="M18 34 C38 22 82 22 102 34" stroke="#b45309" stroke-width="7" stroke-linecap="round" fill="none"/>
  <!-- Pepperonis -->
  <circle cx="56" cy="54" r="5" fill="#dc2626"/>
  <circle cx="46" cy="42" r="4.5" fill="#dc2626"/>
  <circle cx="72" cy="44" r="4.5" fill="#dc2626"/>
  <circle cx="60" cy="74" r="4" fill="#dc2626"/>
</svg>"""

# 26. Sandwich
SVGS["sandwich.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <!-- Bottom bread -->
  <path d="M24 74 L96 74 C98 84 92 88 60 88 C28 88 22 84 24 74 Z" fill="#d97706" stroke="#b45309" stroke-width="1.5"/>
  <!-- Fillings -->
  <rect x="22" y="66" width="76" height="8" rx="2" fill="#16a34a"/>
  <rect x="24" y="60" width="72" height="6" rx="2" fill="#dc2626"/>
  <rect x="22" y="54" width="76" height="6" rx="2" fill="#eab308"/>
  <!-- Top bread -->
  <path d="M24 54 L96 54 C98 44 90 34 60 34 C30 34 22 44 24 54 Z" fill="#f59e0b" stroke="#b45309" stroke-width="2"/>
</svg>"""

# 27. Donut
SVGS["donut.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <circle cx="60" cy="60" r="38" fill="#d97706" stroke="#b45309" stroke-width="2"/>
  <circle cx="60" cy="60" r="32" fill="#ec4899"/>
  <circle cx="60" cy="60" r="14" fill="#0f172a" stroke="#b45309" stroke-width="2"/>
  <!-- Sprinkles -->
  <line x1="42" y1="42" x2="48" y2="44" stroke="#ffffff" stroke-width="2.5" stroke-linecap="round"/>
  <line x1="72" y1="40" x2="76" y2="46" stroke="#38bdf8" stroke-width="2.5" stroke-linecap="round"/>
  <line x1="78" y1="68" x2="84" y2="66" stroke="#facc15" stroke-width="2.5" stroke-linecap="round"/>
  <line x1="40" y1="74" x2="46" y2="70" stroke="#34d399" stroke-width="2.5" stroke-linecap="round"/>
</svg>"""

# 28. Cake
SVGS["cake.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <path d="M24 74 L96 74 L90 100 L30 100 Z" fill="#f43f5e" stroke="#e11d48" stroke-width="2"/>
  <rect x="22" y="70" width="76" height="6" rx="3" fill="#ffffff"/>
  <path d="M30 46 L90 46 L86 70 L34 70 Z" fill="#ec4899" stroke="#db2777" stroke-width="2"/>
  <rect x="28" y="42" width="64" height="6" rx="3" fill="#ffffff"/>
  <!-- Candle -->
  <rect x="58" y="24" width="4" height="18" fill="#38bdf8"/>
  <circle cx="60" cy="18" r="4" fill="#f59e0b"/>
</svg>"""

# 29. Ball
SVGS["ball.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <circle cx="60" cy="60" r="38" fill="#f97316" stroke="#ea580c" stroke-width="2"/>
  <path d="M22 60 C40 60 52 40 52 22" fill="none" stroke="#0f172a" stroke-width="2"/>
  <path d="M22 60 C40 60 52 80 52 98" fill="none" stroke="#0f172a" stroke-width="2"/>
  <path d="M98 60 C80 60 68 40 68 22" fill="none" stroke="#0f172a" stroke-width="2"/>
  <path d="M98 60 C80 60 68 80 68 98" fill="none" stroke="#0f172a" stroke-width="2"/>
  <line x1="22" y1="60" x2="98" y2="60" stroke="#0f172a" stroke-width="2"/>
  <line x1="60" y1="22" x2="60" y2="98" stroke="#0f172a" stroke-width="2"/>
</svg>"""

# 30. Teddy Bear
SVGS["teddy.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <!-- Ears -->
  <circle cx="40" cy="36" r="10" fill="#b45309"/>
  <circle cx="80" cy="36" r="10" fill="#b45309"/>
  <!-- Head -->
  <circle cx="60" cy="52" r="24" fill="#d97706" stroke="#b45309" stroke-width="2"/>
  <!-- Snout -->
  <ellipse cx="60" cy="58" rx="10" ry="7" fill="#fef3c7"/>
  <circle cx="60" cy="56" r="2.5" fill="#1e293b"/>
  <!-- Eyes -->
  <circle cx="52" cy="48" r="2.5" fill="#1e293b"/>
  <circle cx="68" cy="48" r="2.5" fill="#1e293b"/>
  <!-- Body -->
  <path d="M42 74 C34 84 36 104 60 104 C84 104 86 84 78 74 Z" fill="#d97706" stroke="#b45309" stroke-width="2"/>
  <ellipse cx="60" cy="90" rx="10" ry="8" fill="#fef3c7"/>
</svg>"""

# 31. Toothbrush
SVGS["toothbrush.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <g transform="rotate(35 60 60)">
    <!-- Handle -->
    <rect x="57" y="32" width="6" height="74" rx="3" fill="#0284c7" stroke="#38bdf8" stroke-width="1.5"/>
    <!-- Head -->
    <rect x="56" y="16" width="8" height="18" rx="2" fill="#38bdf8"/>
    <!-- Bristles -->
    <rect x="64" y="16" width="6" height="16" fill="#ffffff" stroke="#94a3b8" stroke-width="0.5"/>
  </g>
</svg>"""

# 32. Hair Drier
SVGS["hairdrier.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <!-- Barrel -->
  <rect x="36" y="30" width="46" height="24" rx="4" fill="#0284c7" stroke="#38bdf8" stroke-width="2"/>
  <rect x="82" y="34" width="12" height="16" fill="#1e293b" stroke="#64748b" stroke-width="1.5"/>
  <circle cx="36" cy="42" r="12" fill="#0369a1"/>
  <!-- Handle -->
  <path d="M46 54 L52 94 C52 98 58 98 60 94 L66 54" fill="#0f172a" stroke="#64748b" stroke-width="2"/>
</svg>"""

# 33. Vase
SVGS["vase.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <path d="M50 20 L70 20 C64 36 82 54 82 82 C82 98 72 102 60 102 C48 102 38 98 38 82 C38 54 56 36 50 20 Z" fill="#7c3aed" stroke="#a78bfa" stroke-width="2"/>
  <ellipse cx="60" cy="20" rx="10" ry="3" fill="#6d28d9" stroke="#a78bfa" stroke-width="1.5"/>
</svg>"""

# 34. Tie
SVGS["tie.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <!-- Knot -->
  <polygon points="54,16 66,16 63,26 57,26" fill="#b91c1c" stroke="#dc2626" stroke-width="1.5"/>
  <!-- Blade -->
  <polygon points="57,26 63,26 70,88 60,104 50,88" fill="#ef4444" stroke="#dc2626" stroke-width="2"/>
  <!-- Stripes -->
  <line x1="56" y1="40" x2="65" y2="44" stroke="#ffffff" stroke-width="2"/>
  <line x1="54" y1="60" x2="67" y2="64" stroke="#ffffff" stroke-width="2"/>
  <line x1="52" y1="80" x2="69" y2="84" stroke="#ffffff" stroke-width="2"/>
</svg>"""

# 35. Red Block (Object A)
SVGS["block_red.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <!-- Isometric Red Cube -->
  <polygon points="60,24 94,42 60,60 26,42" fill="#f87171" stroke="#fca5a5" stroke-width="1.5"/>
  <polygon points="60,60 94,42 94,80 60,98" fill="#dc2626" stroke="#b91c1c" stroke-width="1.5"/>
  <polygon points="60,60 26,42 26,80 60,98" fill="#991b1b" stroke="#7f1d1d" stroke-width="1.5"/>
  <!-- Tech Markings / Payload Cross -->
  <line x1="60" y1="36" x2="60" y2="48" stroke="#ffffff" stroke-width="2" stroke-linecap="round"/>
  <line x1="50" y1="42" x2="70" y2="42" stroke="#ffffff" stroke-width="2" stroke-linecap="round"/>
</svg>"""

# 36. Wooden Block (Object B)
SVGS["block_wood.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <!-- Isometric Gold / Wooden Base Block -->
  <polygon points="60,26 96,44 60,62 24,44" fill="#fbbf24" stroke="#fef08a" stroke-width="1.5"/>
  <polygon points="60,62 96,44 96,82 60,100" fill="#d97706" stroke="#b45309" stroke-width="1.5"/>
  <polygon points="60,62 24,44 24,82 60,100" fill="#92400e" stroke="#78350f" stroke-width="1.5"/>
  <!-- Alignment Socket Target -->
  <ellipse cx="60" cy="44" rx="10" ry="5" fill="#78350f" stroke="#fbbf24" stroke-width="1.5"/>
</svg>"""

# 37. Tray
SVGS["tray.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <!-- Perspective Tray -->
  <polygon points="28,42 92,42 104,82 16,82" fill="#334155" stroke="#64748b" stroke-width="2"/>
  <polygon points="34,48 86,48 94,76 26,76" fill="#0f172a" stroke="#00e6c8" stroke-width="1.5"/>
  <line x1="60" y1="48" x2="60" y2="76" stroke="rgba(0,230,200,0.4)" stroke-width="1.5" stroke-dasharray="2 2"/>
</svg>"""

# 38. Button / Complete Button
SVGS["button.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <circle cx="60" cy="60" r="44" fill="#1e293b" stroke="#475569" stroke-width="2"/>
  <!-- Status Halo -->
  <circle cx="60" cy="60" r="36" fill="none" stroke="#22c55e" stroke-width="3" stroke-dasharray="8 4"/>
  <!-- Push Surface -->
  <circle cx="60" cy="60" r="28" fill="#15803d" stroke="#4ade80" stroke-width="2"/>
  <!-- Checkmark / Power symbol -->
  <path d="M48 60 L56 68 L74 50" stroke="#ffffff" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""

# 39. Generic Payload Component (Fallback)
SVGS["component.svg"] = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <defs>
    <linearGradient id="compGrad" x1="20" y1="20" x2="100" y2="100" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#1e293b"/>
      <stop offset="100%" stop-color="#090d16"/>
    </linearGradient>
  </defs>
  <!-- Main module box -->
  <rect x="24" y="24" width="72" height="72" rx="10" fill="url(#compGrad)" stroke="#00e6c8" stroke-width="2"/>
  <!-- Gold connector pins -->
  <g fill="#f59e0b">
    <rect x="36" y="18" width="6" height="6" rx="1"/>
    <rect x="48" y="18" width="6" height="6" rx="1"/>
    <rect x="60" y="18" width="6" height="6" rx="1"/>
    <rect x="72" y="18" width="6" height="6" rx="1"/>
  </g>
  <!-- Optical lens / Sensor reticle -->
  <circle cx="60" cy="60" r="18" stroke="#38bdf8" stroke-width="2" fill="#0284c7" fill-opacity="0.3"/>
  <circle cx="60" cy="60" r="10" stroke="#00e6c8" stroke-width="1.5" fill="#0369a1"/>
  <circle cx="60" cy="60" r="3" fill="#ffffff"/>
  <!-- Crosshair ticks -->
  <line x1="60" y1="36" x2="60" y2="40" stroke="#00e6c8" stroke-width="2"/>
  <line x1="60" y1="80" x2="60" y2="84" stroke="#00e6c8" stroke-width="2"/>
  <line x1="36" y1="60" x2="40" y2="60" stroke="#00e6c8" stroke-width="2"/>
  <line x1="80" y1="60" x2="84" y2="60" stroke="#00e6c8" stroke-width="2"/>
</svg>"""

for name, content in SVGS.items():
    path = OUT_DIR / name
    path.write_text(content.strip(), encoding="utf-8")
    print(f"Generated: {name}")

print(f"Successfully generated {len(SVGS)} object SVG assets.")
