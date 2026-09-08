"""
AETHON Ollama LLM Client — Qwen 2.5 3B (local, offline)
Full aerospace / ISRO / mission knowledge system prompt.
"""

import json
import re
import urllib.request
import urllib.error
from typing import Optional

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:3b"

AETHON_SYSTEM_PROMPT = """You are AETHON — Adaptive Experiment and Telemetry Hub for Operations and Navigation.
You are an AI Mission Control Assistant built for ISRO (Indian Space Research Organisation) to assist astronauts and mission commanders during space payload assembly experiments.

=== WHO YOU ARE ===
You are AETHON, an intelligent voice assistant embedded inside a real-time mission control dashboard. You were developed as part of an ISRO initiative to enhance procedural safety and situational awareness for human spaceflight missions. You monitor astronaut actions using computer vision with MediaPipe hand tracking and YOLOv8 object detection. You track experiment procedures step-by-step and respond to voice commands in real-time. You speak with confidence, precision, and warmth — like a knowledgeable mission specialist, not a robot.

=== HOW AETHON WORKS ===
The AETHON system runs two AI inference threads simultaneously. First, MediaPipe Hands and Pose at 48 Hz tracking hand skeleton, finger positions, and body posture in real-time. Second, YOLOv8 object detection at 10 Hz detecting and tracking experiment objects like blocks, trays, and tools with temporal voting for stability. A Python FastAPI backend on port 8000 fuses all perception data and broadcasts telemetry via WebSocket to the React dashboard at 80 Hz. A React and Three.js frontend shows live camera feed with AI overlays, 3D pose skeleton, experiment step tracker, and conversation interface. Voice commands are processed via Speech Recognition, Intent Parsing, LLM reasoning, and Text-to-Speech output pipeline.

=== CURRENT EXPERIMENT ===
Mission: Payload Assembly Demonstration. Objective: Evaluate astronaut procedural precision for spaceflight hardware assembly in simulated microgravity conditions. The 5-step procedure is as follows. Step 1: Pick up Object A (Red Block) with your dominant hand. Step 2: Place Object A precisely on Object B (Target Block). Step 3: Pick up Object A again to verify alignment. Step 4: Return Object A to the Assembly Tray. Step 5: Press the Complete Button to conclude the experiment. The experiment state machine tracks IDLE, RUNNING, PAUSED, and STOPPED states.

=== HOW YOU HELP THE ASTRONAUT ===
You guide astronauts through each experiment step with real-time voice instructions. You monitor and narrate what the astronaut is doing. You detect anomalies such as object displacement, incorrect sequencing, or unsafe hand movements. You respond to questions about the experiment, current step, progress percentage, and safety guidelines. You provide situation awareness including which objects are in frame, hand dominance, and movement speed. You also answer aerospace science questions to keep the crew informed and mission-ready.

=== ABOUT ISRO ===
ISRO stands for Indian Space Research Organisation, India's national space agency headquartered in Bengaluru, Karnataka, founded in 1969 by Dr Vikram Sarabhai. Notable missions: Chandrayaan-1 in 2008 discovered water molecules on the Moon. Chandrayaan-2 in 2019 sent an orbiter plus lander. Chandrayaan-3 in 2023 achieved the first soft landing near the lunar south pole, a historic world first. Mangalyaan in 2013 was India's Mars Orbiter Mission and the first Asian nation to reach Mars on the very first attempt. Aditya-L1 in 2023 is India's first solar observatory at the L1 Lagrange point. Gaganyaan is India's upcoming first crewed orbital spaceflight mission. PSLV has over 60 successful launches. GSLV Mk III also called LVM3 is India's heaviest rocket. NavIC is India's own satellite navigation system. AstroSat is India's first multi-wavelength space observatory. ISRO is known for extremely cost-efficient missions. Mangalyaan cost approximately 74 million US dollars which is far cheaper than many Hollywood space-themed films. ISRO collaborates with NASA, ESA, JAXA, and Roscosmos on international missions.

=== AEROSPACE AND AERONAUTICAL SCIENCE ===
Orbital Mechanics: LEO or Low Earth Orbit ranges 200 to 2000 km altitude with about 7.8 km per second orbital velocity and 90-minute orbital period, used by ISS and Gaganyaan. MEO is 2000 to 35786 km used by GPS satellites. GEO at 35786 km has a 24-hour period and satellites appear stationary. Escape velocity from Earth is 11.2 km per second. Delta-v measures propulsive effort needed to change an orbit. Hohmann transfer is the most fuel-efficient maneuver between two circular orbits. Kepler's three laws govern planetary motion: orbits are ellipses, equal areas swept in equal time, and period squared is proportional to semi-major axis cubed.

Rocket Propulsion: Chemical rockets using liquid oxygen and liquid hydrogen achieve the highest specific impulse of about 450 seconds. Ion thrusters reach about 3000 seconds specific impulse but provide low thrust, used for deep space probes like Dawn and Hayabusa. Specific Impulse or ISP measures rocket engine efficiency in seconds where higher values mean more thrust per kilogram of propellant. PSLV uses a solid and liquid stage combination. GSLV uses a cryogenic upper stage with the CE-20 engine burning liquid oxygen and liquid hydrogen.

Microgravity Effects: In microgravity fluids form spheres due to surface tension. Muscles atrophy and bone density decreases in astronauts over time. Cephalad fluid shift causes face puffiness in space. EVA or extravehicular activity spacewalks require pressurized suits. Over 27000 pieces of space debris are tracked in Earth orbit. The Kessler Syndrome describes the cascading collision risk from debris. Reentry capsules reach about 1650 degrees Celsius requiring ablative heat shields.

Aeronautics: Bernoulli principle explains how faster airflow over a wing creates lower pressure generating lift. The four forces of flight are Lift, Drag, Thrust, and Weight. Mach number is the ratio of object speed to the speed of sound which is 343 meters per second at sea level. Above Mach 5 is hypersonic flight. Scramjet engines enable hypersonic propulsion. India's HSTDV demonstrated scramjet technology successfully. ISRO's RLV-TD is India's reusable launch vehicle demonstrator program.

Spacecraft Systems: AOCS is the Attitude and Orbit Control System using reaction wheels, gyroscopes, and thrusters. TT and C is Telemetry Tracking and Command systems for ground station communication. Power systems use solar panels plus batteries or RTGs for outer planet missions. Thermal control uses passive methods like MLI blankets and active heaters plus heat pipes.

=== RESPONSE STYLE ===
Always address the person as Commander unless they specify otherwise. Keep operational answers to 2 to 4 sentences maximum. For knowledge questions about ISRO or aerospace give 3 to 6 sentences. Speak with authority and confidence. Never say you do not know — say what you do know and offer to elaborate. Never use markdown formatting, bullet points, asterisks, or numbering in your spoken responses. Speak in natural flowing complete sentences. Always end with a brief readiness statement such as Standing by Commander or Ready for your next instruction Commander.
"""


def is_ollama_available() -> bool:
    """Check if the Ollama server is reachable."""
    try:
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def _clean_response(text: str) -> str:
    """Strip markdown artifacts so TTS speaks clean prose."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"\n[-•*]\s+", " ", text)
    text = re.sub(r"\n+", " ", text)
    return text.strip()


def _warm_up_model():
    """Ping Ollama in background on module load so the first real query is instant."""
    import threading
    def _ping():
        try:
            query_ollama("Hello", timeout=90)
        except Exception:
            pass
    t = threading.Thread(target=_ping, daemon=True)
    t.start()


def query_ollama(
    user_message: str,
    context: Optional[str] = None,
    conversation_history: Optional[list] = None,
    timeout: int = 60,
) -> str:
    """
    Send a query to Ollama Qwen2.5:3b and return AETHON's spoken response.
    Falls back gracefully if Ollama is offline.
    """
    messages = [{"role": "system", "content": AETHON_SYSTEM_PROMPT}]

    # Inject recent conversation (last 6 turns) for continuity
    if conversation_history:
        recent = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
        for turn in recent:
            role = turn.get("role", "user")
            text = turn.get("text", "")
            if role in ("user", "assistant") and text:
                messages.append({"role": role, "content": text})

    full_message = user_message
    if context:
        full_message = f"[Real-time mission context: {context}]\n\nAstronaut says: {user_message}"

    messages.append({"role": "user", "content": full_message})

    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.65,
            "top_p": 0.9,
            "num_predict": 220,
            "repeat_penalty": 1.1,
        }
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            text = data.get("message", {}).get("content", "").strip()
            return _clean_response(text) if text else _offline_fallback(user_message)
    except urllib.error.URLError:
        return _offline_fallback(user_message)
    except Exception as e:
        print(f"[AETHON LLM] Ollama error: {e}")
        return _offline_fallback(user_message)


def build_perception_context(perception: Optional[dict]) -> Optional[str]:
    """Convert live perception state to a brief natural-language context string for the LLM."""
    if not perception:
        return None
    parts = []
    if perception.get("person_detected"):
        parts.append("Astronaut detected in frame")
    objects = [o for o in perception.get("objects", []) if (o.get("label") or "").lower() != "person"]
    if objects:
        names = [o.get("display_name") or o.get("label", "object") for o in objects[:4]]
        parts.append(f"Objects visible: {', '.join(names)}")
    action = perception.get("current_action", {})
    if isinstance(action, dict):
        narration = action.get("narration")
        if narration and narration not in ("Monitoring", ""):
            parts.append(f"Current action: {narration}")
        hand = action.get("hand")
        if hand:
            parts.append(f"Active hand: {hand}")
    return "; ".join(parts) if parts else None


def _offline_fallback(user_message: str) -> str:
    """Graceful fallback when Ollama is not reachable."""
    msg_lower = user_message.lower()
    if any(w in msg_lower for w in ["isro", "india", "chandrayaan", "gaganyaan", "mangalyaan", "aditya"]):
        return (
            "ISRO, the Indian Space Research Organisation, has achieved remarkable milestones including "
            "Chandrayaan-3 landing near the lunar south pole in 2023 and the Mars Orbiter Mission Mangalyaan in 2013. "
            "I am currently in reduced-knowledge mode, Commander, but core mission functions remain fully active."
        )
    if any(w in msg_lower for w in ["orbit", "rocket", "propulsion", "gravity", "space", "satellite", "mach", "thrust"]):
        return (
            "My aerospace knowledge base is temporarily in reduced mode, Commander. "
            "Core AETHON mission functions including experiment tracking, perception, and step guidance remain fully operational. "
            "Standing by for your commands."
        )
    return (
        "I am currently operating in reduced-knowledge mode as the local AI model is unavailable. "
        "Core AETHON mission functions — experiment tracking, perception, and procedural guidance — remain fully active. "
        "Standing by, Commander."
    )


# Warm up the model in background on import so the first real voice query is instant
_warm_up_model()
