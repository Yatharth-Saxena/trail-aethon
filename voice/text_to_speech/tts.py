import pyttsx3
import threading
import queue
import time
from typing import Optional

class TextToSpeechService:
    def __init__(self, rate: int = 165, volume: float = 0.95):
        self.rate = rate
        self.volume = volume
        self.queue = queue.Queue()
        self.is_running = True
        self.muted = False
        self.is_speaking = False
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def _worker(self):
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except Exception:
            pass
        engine = None
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', self.rate)
            engine.setProperty('volume', self.volume)
            # Pick a natural voice if available
            voices = engine.getProperty('voices')
            if voices:
                # Prefer English female voice (Microsoft Zira)
                for v in voices:
                    if "zira" in v.name.lower():
                        engine.setProperty('voice', v.id)
                        break
        except Exception as e:
            print(f"[TTS Worker Warning] Engine init warning: {e}")

        while self.is_running:
            try:
                text = self.queue.get(timeout=0.5)
                if text is None:
                    break
                if not self.muted and text.strip():
                    if engine:
                        try:
                            self.is_speaking = True
                            engine.say(text)
                            engine.runAndWait()
                        except Exception as e:
                            print(f"[TTS Error] Engine playback error: {e}")
                            # Re-initialize engine if it got stuck
                            try:
                                engine = pyttsx3.init()
                            except Exception:
                                pass
                        finally:
                            self.is_speaking = False
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[TTS Loop Error] {e}")
                self.is_speaking = False

    @property
    def is_active(self) -> bool:
        return self.is_speaking or not self.queue.empty()

    def speak(self, text: str):
        if not text or self.muted:
            return
        # Avoid queuing huge backlogs of repetitive messages
        if self.queue.qsize() > 2:
            try:
                while not self.queue.empty():
                    self.queue.get_nowait()
                    self.queue.task_done()
            except Exception:
                pass
        self.queue.put(text)

    def stop(self):
        self.is_running = False
        self.queue.put(None)

# Singleton instance
tts_service = TextToSpeechService()
