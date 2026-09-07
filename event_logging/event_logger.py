import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from backend.config import LOGS_DIR, SESSIONS_DIR

class EventLogger:
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir = SESSIONS_DIR / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "snapshots").mkdir(parents=True, exist_ok=True)
        
        self.session_log_file = self.session_dir / "events.jsonl"
        self.global_log_file = LOGS_DIR / "aethon.jsonl"
        self.in_memory_logs: List[Dict[str, Any]] = []

    def set_session(self, session_id: str):
        self.session_id = session_id
        self.session_dir = SESSIONS_DIR / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "snapshots").mkdir(parents=True, exist_ok=True)
        self.session_log_file = self.session_dir / "events.jsonl"

    def log(self, event_type: str, message: str, details: Optional[Dict[str, Any]] = None, level: str = "INFO") -> Dict[str, Any]:
        now = datetime.now()
        entry = {
            "timestamp": now.isoformat(),
            "time_str": now.strftime("%H:%M:%S"),
            "session_id": self.session_id,
            "event_type": event_type,
            "level": level,
            "message": message,
            "details": details or {}
        }

        self.in_memory_logs.append(entry)
        if len(self.in_memory_logs) > 500:
            self.in_memory_logs.pop(0)

        line = json.dumps(entry) + "\n"
        try:
            with open(self.session_log_file, "a", encoding="utf-8") as f:
                f.write(line)
            with open(self.global_log_file, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception as e:
            print(f"[EventLogger Error] Failed to write log: {e}")

        return entry

    def get_recent_logs(self, limit: int = 50, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        logs = self.in_memory_logs
        if event_type:
            logs = [l for l in logs if l["event_type"] == event_type]
        return logs[-limit:]

    def get_session_logs(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        path = SESSIONS_DIR / session_id / "events.jsonl"
        if not path.exists():
            return []
        entries = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line.strip()))
        except Exception as e:
            print(f"[EventLogger Error] reading session logs: {e}")
        return entries[-limit:]

# Singleton instance
event_logger = EventLogger()
