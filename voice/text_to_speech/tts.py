import re
import subprocess
import platform
import threading
import queue
from typing import Optional, AsyncGenerator

try:
    import edge_tts
except ImportError:
    edge_tts = None

DEFAULT_VOICE_EN = "en-IN-NeerjaExpressiveNeural"
DEFAULT_VOICE_HI = "hi-IN-SwaraNeural"

# Phonetic normalization for natural speech
PHONETIC_MAP = {
    "AETHON": "Aython",
    "Aethon": "Aython",
    "aethon": "Aython",
}

def _normalize_for_speech(text: str) -> str:
    for key, val in PHONETIC_MAP.items():
        text = text.replace(key, val)
    return text

def detect_voice_for_text(text: str, preferred_voice: Optional[str] = None) -> str:
    """Select appropriate neural voice: Swara for Hindi text, Neerja for Indian English."""
    if preferred_voice:
        return preferred_voice
    # Check for Devanagari script (Hindi characters)
    if re.search(r"[\u0900-\u097F]", text):
        return DEFAULT_VOICE_HI
    return DEFAULT_VOICE_EN

async def stream_neural_tts(text: str, voice: Optional[str] = None) -> AsyncGenerator[bytes, None]:
    """Stream high-fidelity neural speech audio chunks from Microsoft Edge TTS."""
    if edge_tts is None:
        raise RuntimeError("edge-tts library is not installed")
    normalized = _normalize_for_speech(text)
    selected_voice = detect_voice_for_text(text, voice)
    communicate = edge_tts.Communicate(normalized, selected_voice)
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            yield chunk["data"]

class TextToSpeechService:
    """
    Dedicated background TTS service.
    Muted by default to avoid duplicating the browser's high-quality female speech synthesis.
    """
    def __init__(self, rate: int = 175, female_voice: str = "Samantha"):
        self.rate = rate
        self.voice = female_voice
        self.queue: queue.Queue = queue.Queue()
        self.is_running = True
        self.muted = True  # Muted by default: browser client plays natural female voice without echo
        self.is_speaking = False
        self._current_proc: Optional[subprocess.Popen] = None
        self._proc_lock = threading.Lock()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def _worker(self):
        system = platform.system()
        engine = None
        if system != "Darwin":
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty('rate', self.rate)
                voices = engine.getProperty('voices')
                if voices:
                    for v in voices:
                        if "zira" in v.name.lower() or "female" in v.name.lower():
                            engine.setProperty('voice', v.id)
                            break
            except Exception:
                engine = None

        while self.is_running:
            try:
                text = self.queue.get(timeout=0.5)
                if text is None:
                    break
                if not self.muted and text.strip():
                    self.is_speaking = True
                    norm_text = _normalize_for_speech(text)
                    if system == "Darwin":
                        try:
                            proc = subprocess.Popen(
                                ["say", "-v", self.voice, "-r", str(self.rate), norm_text],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL
                            )
                            with self._proc_lock:
                                self._current_proc = proc
                            proc.wait()
                        except Exception:
                            pass
                        finally:
                            with self._proc_lock:
                                self._current_proc = None
                    elif engine:
                        try:
                            engine.say(norm_text)
                            engine.runAndWait()
                        except Exception:
                            pass
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[TTS Loop Error] {e}")
            finally:
                self.is_speaking = False

    @property
    def is_active(self) -> bool:
        return self.is_speaking or not self.queue.empty()

    def speak(self, text: str):
        if not text or self.muted:
            return
        # Interrupt previous output
        with self._proc_lock:
            if self._current_proc and self._current_proc.poll() is None:
                try:
                    self._current_proc.terminate()
                except Exception:
                    pass
        # Clear backlog
        try:
            while not self.queue.empty():
                self.queue.get_nowait()
                self.queue.task_done()
        except Exception:
            pass
        self.queue.put(text)

    def stop(self):
        self.is_running = False
        with self._proc_lock:
            if self._current_proc and self._current_proc.poll() is None:
                try:
                    self._current_proc.terminate()
                except Exception:
                    pass
        self.queue.put(None)

# Singleton instance
tts_service = TextToSpeechService()
