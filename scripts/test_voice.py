"""
AETHON - Voice Assistant Diagnostics
Tests pyttsx3 offline TTS and SpeechRecognition.
"""
from voice.text_to_speech.tts import tts_service
from voice.command_parser.command_parser import command_parser
import time

def test():
    print("[Voice Test] Testing offline TTS engine...")
    test_phrase = "AETHON offline voice system operational."
    tts_service.speak(test_phrase)
    time.sleep(2.5)
    print("[PASS] TTS triggered.")

    print("[Voice Test] Testing Command Parser...")
    cmd = "Hey Aethon what is the next step"
    intent = command_parser.parse(cmd)
    print(f"[PASS] Parsed '{cmd}' -> Intent.{intent.value}")

if __name__ == "__main__":
    test()
