import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from backend.config import SESSIONS_DIR

class SessionManager:
    def __init__(self):
        self.active_session_id: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.active_session_data: Dict[str, Any] = {
            "session_id": self.active_session_id,
            "experiment_id": "payload_assembly_demo",
            "experiment_name": "Payload Assembly (Demo)",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "status": "INITIALIZED",
            "completed_steps": [],
            "violations_count": 0,
            "snapshots_count": 0,
            "has_recording": False
        }
        self._ensure_session_dir(self.active_session_id)
        self.save_active_session()

    def _ensure_session_dir(self, session_id: str) -> Path:
        s_dir = SESSIONS_DIR / session_id
        s_dir.mkdir(parents=True, exist_ok=True)
        (s_dir / "snapshots").mkdir(parents=True, exist_ok=True)
        return s_dir

    def create_new_session(self, experiment_id: str = "payload_assembly_demo") -> str:
        self.active_session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.active_session_data = {
            "session_id": self.active_session_id,
            "experiment_id": experiment_id,
            "experiment_name": "Payload Assembly (Demo)",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "status": "INITIALIZED",
            "completed_steps": [],
            "violations_count": 0,
            "snapshots_count": 0,
            "has_recording": False
        }
        self._ensure_session_dir(self.active_session_id)
        self.save_active_session()
        return self.active_session_id

    def save_active_session(self):
        s_dir = SESSIONS_DIR / self.active_session_id
        meta_file = s_dir / "session.json"
        try:
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(self.active_session_data, f, indent=2)
        except Exception as e:
            print(f"[SessionManager Error] Failed to save session: {e}")

    def update_session(self, **kwargs):
        self.active_session_data.update(kwargs)
        self.save_active_session()

    def list_sessions(self) -> List[Dict[str, Any]]:
        sessions = []
        if not SESSIONS_DIR.exists():
            return sessions
        for p in sorted(SESSIONS_DIR.iterdir(), reverse=True):
            if p.is_dir():
                meta_file = p / "session.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            sessions.append(json.load(f))
                    except Exception:
                        pass
                else:
                    sessions.append({
                        "session_id": p.name,
                        "experiment_name": "Payload Assembly (Demo)",
                        "start_time": p.name
                    })
        return sessions

session_manager = SessionManager()
