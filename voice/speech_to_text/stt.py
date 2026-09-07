import os
import io
import wave
import tempfile
from typing import Optional
import speech_recognition as sr

class SpeechToTextService:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True

    def transcribe_audio_file(self, file_path: str) -> Optional[str]:
        try:
            with sr.AudioFile(file_path) as source:
                audio = self.recognizer.record(source)
                # Try offline recognition if available, or standard local recognizer
                text = self.recognizer.recognize_sphinx(audio) if hasattr(self.recognizer, "recognize_sphinx") else None
                if not text:
                    # Fallback to default recognize_google if network available or internal fallback
                    try:
                        text = self.recognizer.recognize_google(audio)
                    except Exception:
                        pass
                return text
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

stt_service = SpeechToTextService()
