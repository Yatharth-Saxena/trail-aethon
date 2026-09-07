# Cursor Prompt — AETHON Feature Pass

Paste everything below into Cursor (ideally in Composer/Agent mode with the whole repo indexed).

---

You are working in the **AETHON** repo — a FastAPI + OpenCV/YOLOv8/MediaPipe backend (`ai/`, `backend/`, `camera/`, `experiment/`, `voice/`) with a React 19 + Vite dashboard (`client/src/pages/Home.jsx`). Implement the six changes below. Work file-by-file, don't refactor unrelated code, and after each change explain what you changed and why in a short comment block at the top of the diff.

## 1. Real object-movement detection, not just hand-action detection

Today `ai/actions/action_recognizer.py` classifies actions (`PICK_UP`, `PLACE`, `PRESS`, …) purely from **hand** proximity/IoU signals coming out of `ai/interaction/hand_object_interaction.py`. There is no notion of an object moving on its own (e.g., sliding, tipping, vibrating, or being moved by something other than the tracked hands).

- Add a lightweight **per-object motion tracker**: for each YOLOv8 detection returned by `ai/detection/object_detector.py`, track its centroid/bbox across frames (a simple centroid-distance or IoU-based tracker is fine — doesn't need to be DeepSORT) and compute a velocity/displacement-per-second value.
- Feed this into `ai/actions/action_recognizer.py` as a new signal (`object_velocity`, `is_moving`) alongside the existing hand-based action so the temporal buffer can distinguish:
  - a hand-driven action (current `PICK_UP`/`PLACE`/`PRESS`),
  - vs. independent object movement not caused by a tracked hand (flag as `OBJECT_DISPLACED` or similar),
  - vs. a static/idle object.
- Expose this per-object in the WebSocket telemetry payload (`backend/main.py`'s `/ws` broadcast, inside each `perception.objects[]` entry — currently has `held`/`held_by`; add `moving: bool` and `velocity`).
- Add debounce/smoothing (reuse the existing `ACTION_DEBOUNCE_SECONDS` / `TEMPORAL_BUFFER_SIZE` pattern from `backend/config.py`) so camera shake or detector jitter isn't misread as movement.

## 2. Stable 60 FPS capture with minimal latency

`camera/camera_service.py` already paces frames at `target_fps` (60) in `_stream_pipeline_loop()` and sets `CAP_PROP_BUFFERSIZE=1`, but there are several latency/stability leaks — fix them:

- **Reduce redundant copies**: `get_latest_frame()`, `_stream_pipeline_loop()`, and `inject_client_frame()` each do `frame.copy()` under a single global `self.lock`. Under load this serializes capture, overlay drawing, recording, and streaming on one lock. Split into separate locks (or a lock-free double-buffer / `queue.Queue(maxsize=1)` pattern) so JPEG encoding for `/video_feed` doesn't block the capture thread.
- **Decouple perception inference from the frame-pacing loop.** Confirm `ai/pipeline/perception_pipeline.py` runs YOLOv8/MediaPipe on its own thread/interval and never blocks `_hw_capture_loop()` or `_stream_pipeline_loop()`. If it currently runs inline per-frame at 60 Hz, throttle inference to a sane rate (e.g. 15–30 Hz) and just carry forward the last known detections for the frames in between, so the *video* stays at a smooth 60 FPS even if AI inference can't keep up.
- **MJPEG encode cost**: `generate_mjpeg_stream()` re-encodes JPEG per client per frame at quality 75. If more than one client can connect, encode once per frame and fan out the same bytes to all `/video_feed` consumers instead of re-encoding per request.
- **Measure it**: add a rolling end-to-end latency metric (capture timestamp → WebSocket send timestamp) and surface it in the telemetry payload (`camera.latency_ms`) so the dashboard can display real capture→display latency, not just FPS.
- Verify `CAMERA_FPS=60` in `backend/config.py` is actually achievable on the target camera (`cap.set(cv2.CAP_PROP_FPS, ...)` silently no-ops on many USB webcams) — add a startup log that reads back the actual negotiated FPS/resolution from the device and warns if it doesn't match config.

## 3. Snapshot / Record buttons must not route through the chatbot

Root cause: in `client/src/pages/Home.jsx`, `handleSnapshot()` (~line 1183) and `handleRecordToggle()` (~line 1193) correctly `fetch()` `/api/camera/snapshot` and `/api/camera/record/start|stop`, **but then call `sendCommand(...)`** to announce the result. `sendCommand()` (~line 889) posts to `/api/voice/command`, appends a user+assistant chat bubble, scrolls the chat panel into view, and triggers TTS — i.e. it hijacks the UI into "chat mode" even though the user just clicked a camera button.

Fix:
- Remove the `sendCommand(...)` calls from `handleSnapshot` and `handleRecordToggle`.
- Replace with a **local, non-chat notification** — e.g. a toast/snackbar component (or a small transient status pill near the button) showing "Snapshot saved" / "Recording started" / "Recording stopped and saved", driven by local component state, not the conversation/messages state.
- Keep the actual REST calls and `setIsRecording(...)` state updates unchanged — only the confirmation channel changes.
- Do the same audit for any other dashboard buttons that currently call `sendCommand()` purely to display a confirmation (grep `sendCommand(` in `Home.jsx`) and migrate anything that isn't an actual voice/chat interaction to the same toast mechanism.

## 4. Mic should only animate while actual speech is happening

Root cause: the mic button's CSS (`client/src/pages/Home.jsx`, ~line 1997–2009) applies the `active` class and the `pulse` animation based on `isListening || continuousListening || audioRecording`. Because `continuousListening` is a **mode flag** (mic is armed) rather than a **VAD/speech-detected flag**, the icon pulses nonstop the entire time always-on listening is enabled, even during silence.

Fix:
- Introduce a separate state, e.g. `isSpeechActive` (or reuse/extend interim-result handling already present around lines 1031–1059 where `interim`/`transcript` results come in from the Web Speech API).
- Set `isSpeechActive = true` only while `onresult`/`onspeechstart` events are actively firing with non-empty interim transcript content, and `false` on `onspeechend`/silence timeout (a ~600–800ms debounce after the last interim result is reasonable so it doesn't flicker between words).
- Change the animation binding to `animation: isSpeechActive ? "pulse 1.5s ease-in-out infinite" : "none"`.
- Keep the `active` (armed/highlighted) class tied to `continuousListening || isListening || audioRecording` as before — that's correct for showing "mic is on"; only the **pulsing animation** should be gated to actual detected speech, so there's a clear visual difference between "mic armed, waiting" and "mic hearing you right now."

## 5. Improve detection accuracy (both small/minimalistic objects and major objects)

Current setup: `yolov8n.pt` (nano — smallest/fastest/least accurate Ultralytics model) at a flat `CONFIDENCE_THRESHOLD_OBJECT = 0.50` for everything (`backend/config.py`), inferenced on a downsampled 640×360 tensor per the README.

- Add **per-class confidence thresholds** instead of one global threshold — small/thin payload components typically need a lower threshold than large, high-contrast objects to avoid missed detections, while common/large objects can keep a higher threshold to reduce false positives. Store this as a dict in `backend/config.py` or per-experiment in `experiment/configs/*.json`.
- Add **temporal smoothing/voting** in `ai/pipeline/perception_pipeline.py`: only "commit" a detection (or its loss) after it's been consistent for N of the last M frames, using the existing `TEMPORAL_BUFFER_SIZE` pattern, so single-frame misses on small objects don't flicker the UI/state machine.
- Verify inference resolution: confirm whether the "downsampled 640×360" claim in the README matches what `ai/detection/object_detector.py` actually passes to YOLO — if small objects are the accuracy problem, test at a higher inference resolution (e.g. 960 or native) and profile the FPS cost, since a fixed 60 FPS video pipeline (see #2) doesn't need inference to also run at 60 FPS.
- Wire up `scripts/train_detector.py` / `scripts/collect_dataset.py` to fine-tune on real payload-component imagery instead of relying on a stock `yolov8n.pt` — add a config flag/path so a fine-tuned weight file in `models/` is picked up automatically if present, falling back to `yolov8n.pt` otherwise.
- For MediaPipe pose/hand accuracy, expose `CONFIDENCE_THRESHOLD_POSE` / `CONFIDENCE_THRESHOLD_HAND` as tunable per lighting condition rather than fixed at 0.50, and consider enabling MediaPipe's `model_complexity`/landmark-refinement options if not already set to the highest available.

## 6. "Hey AETHON" wake-word activation (listen → capture command → auto-stop mic)

Today `continuousListening` (`Home.jsx`) is just a manual on/off toggle for the Web Speech API — there is no wake-word gating; whatever is said while it's on gets sent as a command via `sendCommand()`.

Implement a proper wake-word state machine in `Home.jsx` (and `voice/command_parser/command_parser.py` on the backend if wake-word matching should also be validated server-side):

- **State: `IDLE_LISTENING`** — mic is continuously running (low duty), but every recognized transcript is checked **locally** against a wake-word regex (`/\bhey\s+aethon\b/i` and `/\baethon\b/i` as a shorter fallback) before anything is sent anywhere. No chat bubble, no backend call, no TTS while in this state.
- **On wake-word match** → transition to **`CAPTURING_COMMAND`**:
  - Play a short audio/visual cue (a chime + the mic UI switching to its "listening for command" animation from #4).
  - Strip the wake phrase itself out of the transcript (reuse/extend `normalizeAethonWakeWord()` at ~line 524, which already exists for display cleanup — extend it to also be used for wake-word *stripping*, not just cosmetic normalization).
  - Start capturing everything said **after** the wake word as the actual command text.
- **End of command** → detect via the Web Speech API's `onspeechend`/pause-based finalization (same debounce idea as #4) — once the user stops talking for ~1–1.5s after the wake word triggered, take the accumulated transcript and call `sendCommand(capturedText)` exactly once.
- **After sending the command** → transition back to `IDLE_LISTENING` (mic stays armed for the next "Hey AETHON") **and explicitly stop/suspend active audio capture** in between (`recognition.stop()`) rather than leaving the raw mic stream open — the mic should be "listening but not transcribing/sending" between wake-word triggers if you want it truly off between commands, or fully stopped if `continuousListening` was already false before the wake-word session started.
- Guard against false triggers: ignore matches that are part of the assistant's own TTS output being picked up by the mic (check `voiceStatus !== 'speaking'` / mute recognition while `speakAssistantResponse` is playing, if not already handled).
- Add a small set of unit tests in `tests/test_command_parser.py` for the wake-word stripping/matching logic (e.g. "hey aethon what's the next step" → strips to "what's the next step").

---

### General instructions for Cursor
- Don't touch `experiment/`, `event_logging/`, `data/`, `streaming/`, or `recording/` unless a change above explicitly requires it.
- Keep all new config values in `backend/config.py` (backend) or as `const`s near the top of `Home.jsx` (frontend) — don't hardcode magic numbers inline.
- After implementing, list every file you touched and a one-line summary of the change per file.
