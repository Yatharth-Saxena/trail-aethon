import os
import io
import wave
import tempfile
import threading
import time
from typing import Optional, Callable
import speech_recognition as sr

class SpeechToTextService:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.is_listening = False
        self.listener_thread: Optional[threading.Thread] = None
        self.on_command_callback: Optional[Callable[[str], None]] = None
        self.last_command: str = ""

    def transcribe_audio_data(self, audio_data: sr.AudioData) -> Optional[str]:
        """Transcribes sr.AudioData using available engines (Google Speech API first, Sphinx fallback)."""
        # 1. Try Google Speech Recognition
        try:
            text = self.recognizer.recognize_google(audio_data)
            if text and text.strip():
                return text.strip()
        except sr.UnknownValueError:
            return None
        except Exception as e:
            # Fall through if Google is offline or unavailable
            pass

        # 2. Try Sphinx if pocketsphinx is actually installed
        try:
            import pocketsphinx  # type: ignore
            text = self.recognizer.recognize_sphinx(audio_data)
            if text and text.strip():
                return text.strip()
        except Exception:
            pass

        return None

    def transcribe_audio_file(self, file_path: str) -> Optional[str]:
        try:
            with sr.AudioFile(file_path) as source:
                audio = self.recognizer.record(source)
                return self.transcribe_audio_data(audio)
        except Exception as e:
            print(f"[STT Error] Failed to transcribe file: {e}")
            return None

    def transcribe_audio_bytes(self, audio_bytes: bytes) -> Optional[str]:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            try:
                res = self.transcribe_audio_file(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            return res
        except Exception as e:
            print(f"[STT Error] Transcribe bytes error: {e}")
            return None

    def start_background_listener(self, on_command_callback: Callable[[str], None]):
        """Starts live system microphone listening in a background daemon thread."""
        if self.is_listening:
            return
        self.is_listening = True
        self.on_command_callback = on_command_callback
        self.listener_thread = threading.Thread(target=self._mic_listen_loop, daemon=True)
        self.listener_thread.start()
        print("[STT Service] Background microphone listener started.")

    def stop_background_listener(self):
        self.is_listening = False

    def _mic_listen_loop(self):
        try:
            mic = sr.Microphone()
            with mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.8)
            print("[STT Service] Physical microphone calibrated and listening.")
            
            while self.is_listening:
                try:
                    with mic as source:
                        audio = self.recognizer.listen(source, timeout=3.0, phrase_time_limit=8.0)
                    
                    text = self.transcribe_audio_data(audio)
                    if text and text.strip():
                        print(f"[STT Service] Physical mic voice command detected: '{text}'")
                        self.last_command = text.strip()
                        if self.on_command_callback:
                            self.on_command_callback(self.last_command)
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    time.sleep(0.5)
        except Exception as e:
            print(f"[STT Service] Physical microphone listener notice: {e}")
            self.is_listening = False

stt_service = SpeechToTextService()

