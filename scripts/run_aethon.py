"""
AETHON - Unified Application Launcher
Starts the FastAPI backend on port 8000.
"""
import sys
import os
from pathlib import Path

# Add root directory to python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import uvicorn
from backend.config import SERVER_HOST, SERVER_PORT

def run():
    print("=" * 60)
    print("LAUNCHING AETHON - Offline AI Experiment Assistance System")
    print(f"Backend Server: http://localhost:{SERVER_PORT}")
    print("Webcam & AI Perception Pipeline initializing...")
    print("=" * 60)
    uvicorn.run("backend.main:app", host=SERVER_HOST, port=SERVER_PORT, reload=False)

if __name__ == "__main__":
    run()
