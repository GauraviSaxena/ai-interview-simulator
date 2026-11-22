"""Live eye-contact and posture coaching with MediaPipe FaceMesh, Pose, and real-time visual alerts."""
from __future__ import annotations

import math
import sys
import threading
import time
import types
from collections import deque
from typing import Any, Optional

import cv2
import numpy as np

# Stub out tensorflow docs to prevent Windows AppLocker / DLL conflicts when mediapipe loads
if "tensorflow" not in sys.modules:
    dummy_tf = types.ModuleType("tensorflow")
    dummy_tools = types.ModuleType("tensorflow.tools")
    dummy_docs = types.ModuleType("tensorflow.tools.docs")
    dummy_docs.doc_controls = types.SimpleNamespace(
        do_not_generate_docs=lambda x: x,
        doc_private=lambda x: x,
    )
    sys.modules["tensorflow"] = dummy_tf
    sys.modules["tensorflow.tools"] = dummy_tools
    sys.modules["tensorflow.tools.docs"] = dummy_docs

# Ideal Posture Benchmarks
MAX_SHOULDER_TILT_DEG = 5.0  # Shoulders aligned within ±5°
GUIDE_BOX_WIDTH_RATIO = 0.60  # Comfortable 60% width of frame
GUIDE_BOX_HEIGHT_RATIO = 0.70  # Comfortable 70% height of frame
DRIFT_ALERT_THRESHOLD_SEC = 2.0  # Alert persistence window


class MediaBuffer:
    """Thread-safe processor capturing audio chunks and running live MediaPipe FaceMesh & Pose tracking."""

    def __init__(self, window: int = 150):
        self.lock = threading.Lock()
        
        # Audio accumulator
        self.audio_chunks: list[np.ndarray] = []
        self.audio_samples: int = 0
        self.sample_rate: int = 48000
        
        # Real frame counters (strictly NO fake defaults: starts at 0.0)
        self.total_frames_processed: int = 0
        self.face_detected_frames: int = 0
        self.gaze_aligned_frames: int = 0
        self.posture_aligned_frames: int = 0

        # Vision metrics rolling buffers
        self.gaze_history = deque(maxlen=window)
        self.posture_history = deque(maxlen=window)
        self.face_seen_history = deque(maxlen=window)
        self.shoulder_tilt_history = deque(maxlen=window)
        
        # Error tracking & alerts
        self.posture_error_count: int = 0
        self.last_error_time: float = 0.0
        self.violation_start_time: Optional[float] = None
        self.is_in_violation: bool = False
        self.current_alert_message: str = ""
        self.alert_display_until: float = 0.0
        
        # Lazy models
        self._mesh = None
        self._pose = None
        self._models_failed = False
        
        self.started_at = time.monotonic()

    def _init_models(self):
        """Lazily initialize MediaPipe solutions."""
        if self._mesh is None and not self._models_failed:
            try:
                import mediapipe as mp
                self._mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self._pose = mp.solutions.pose.Pose(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            except Exception:
                self._models_failed = True

    def process_audio(self, frame):
        """WebRTC audio callback: converts PCM float/int to int16 samples for Whisper STT."""
        arr = frame.to_ndarray()
        if np.issubdtype(arr.dtype, np.floating):
            arr = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)
        elif arr.dtype != np.int16:
            info = np.iinfo(arr.dtype)
            scale = float(max(abs(info.min), abs(info.max)))
            arr = (arr.astype(np.float32) / scale * 32767.0).astype(np.int16)

        with self.lock:
            self.sample_rate = int(getattr(frame, "sample_rate", 48000) or 48000)
            self.audio_chunks.append(arr.copy())
            self.audio_samples += int(arr.shape[-1])
        return frame

    def process_video(self, frame):
        """WebRTC video callback: detects shoulder tilt, eye-gaze, renders on-screen alerts."""
        if hasattr(frame, "to_ndarray"):
            image = frame.to_ndarray(format="bgr24")
        else:
            image = np.array(frame, copy=True)
        h, w = image.shape[:2]
        now = time.monotonic()
        
        self._init_models()

        posture_ok = True
        gaze_ok = True
        current_tilt = 0.0
        face_center_x, face_center_y = 0.5, 0.4
        has_face = False
        alert_reason = ""
        banner_text = ""

        # Draw centered 60% width x 70% height framing zone for head & shoulders
        center_x_min = int(w * ((1.0 - GUIDE_BOX_WIDTH_RATIO) / 2.0))  # 20%
        center_x_max = int(w * ((1.0 + GUIDE_BOX_WIDTH_RATIO) / 2.0))  # 80%
        center_y_min = int(h * ((1.0 - GUIDE_BOX_HEIGHT_RATIO) / 2.0))  # 15%
        center_y_max = int(h * ((1.0 + GUIDE_BOX_HEIGHT_RATIO) / 2.0))  # 85%

        if not self._models_failed and self._mesh and self._pose:
            try:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mesh_res = self._mesh.process(rgb)
                pose_res = self._pose.process(rgb)

                # 1. FaceMesh tracking
                if mesh_res.multi_face_landmarks:
                    has_face = True
                    pts = mesh_res.multi_face_landmarks[0].landmark
                    nose = pts[1]
                    face_center_x, face_center_y = float(nose.x), float(nose.y)

                    # Check face centering within 60% width x 70% height guide zone (0.20-0.80 x, 0.15-0.85 y)
                    box_x_min_norm = (1.0 - GUIDE_BOX_WIDTH_RATIO) / 2.0
                    box_x_max_norm = (1.0 + GUIDE_BOX_WIDTH_RATIO) / 2.0
                    box_y_min_norm = (1.0 - GUIDE_BOX_HEIGHT_RATIO) / 2.0
                    box_y_max_norm = (1.0 + GUIDE_BOX_HEIGHT_RATIO) / 2.0

                    if not (box_x_min_norm <= face_center_x <= box_x_max_norm and
                            box_y_min_norm <= face_center_y <= box_y_max_norm):
                        posture_ok = False
                        alert_reason = "⚠️ POSTURE ALERT: Align Shoulders / Face Not Centered"
                        banner_text = "⚠️ POSTURE ALERT: Align Shoulders / Face Not Centered"

                    # Iris eye-gaze tracking
                    iris_ids = (468, 473)
                    corners = ((33, 133), (362, 263))
                    eye_scores = []
                    for iris_idx, (c1, c2) in zip(iris_ids, corners):
                        if len(pts) > iris_idx:
                            lo, hi = sorted((pts[c1].x, pts[c2].x))
                            pos = (pts[iris_idx].x - lo) / max(hi - lo, 1e-5)
                            eye_scores.append(max(0.0, 1.0 - abs(pos - 0.5) * 2.2))
                    
                    gaze_score = float(np.mean(eye_scores) * 100.0) if eye_scores else 0.0
                    if gaze_score < 45.0:
                        gaze_ok = False
                        if not alert_reason:
                            alert_reason = "⚠️ POSTURE ALERT: Maintain direct eye contact with camera"
                            banner_text = "⚠️ POSTURE ALERT: Maintain Direct Eye Contact"
                else:
                    has_face = False
                    gaze_score = 0.0
                    posture_ok = False
                    gaze_ok = False
                    alert_reason = "❌ NO FACE DETECTED"
                    banner_text = "❌ NO FACE DETECTED: Position Face & Shoulders in Frame"

                # 2. Pose tracking for shoulder horizontal alignment
                if pose_res.pose_landmarks:
                    lms = pose_res.pose_landmarks.landmark
                    # Landmarks 11: left shoulder, 12: right shoulder
                    if len(lms) > 12:
                        ls = lms[11]
                        rs = lms[12]
                        if ls.visibility > 0.4 and rs.visibility > 0.4:
                            x1, y1 = int(ls.x * w), int(ls.y * h)
                            x2, y2 = int(rs.x * w), int(rs.y * h)
                            dx = abs(x2 - x1)
                            dy = y2 - y1  # vertical difference
                            if dx > 10:
                                current_tilt = abs(math.atan2(dy, dx) * 180.0 / math.pi)
                                shoulder_color = (0, 200, 0) if current_tilt <= MAX_SHOULDER_TILT_DEG else (0, 0, 230)
                                cv2.line(image, (x1, y1), (x2, y2), shoulder_color, 2, cv2.LINE_AA)
                                cv2.putText(image, f"Tilt: {current_tilt:.1f} deg", (min(x1, x2), max(y1, y2) + 22),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, shoulder_color, 1, cv2.LINE_AA)
                                
                                if current_tilt > MAX_SHOULDER_TILT_DEG:
                                    posture_ok = False
                                    alert_reason = f"⚠️ POSTURE ALERT: Align Shoulders (Tilt: {current_tilt:.1f}° > 5.0°)"
                                    banner_text = f"⚠️ POSTURE ALERT: Align Shoulders (Tilt: {current_tilt:.1f}° > 5.0°)"

            except Exception:
                pass
        else:
            # Fallback when mediapipe is not available or camera covered
            has_face = False
            gaze_score = 0.0
            posture_ok = False
            gaze_ok = False
            alert_reason = "❌ NO FACE DETECTED"
            banner_text = "❌ NO FACE DETECTED"

        # Check violation state
        is_alert = (not posture_ok) or (not gaze_ok) or (not has_face)

        with self.lock:
            self.total_frames_processed += 1
            if has_face:
                self.face_detected_frames += 1
                if gaze_ok:
                    self.gaze_aligned_frames += 1
                if posture_ok:
                    self.posture_aligned_frames += 1

            self.face_seen_history.append(1.0 if has_face else 0.0)
            self.gaze_history.append(gaze_score)
            self.shoulder_tilt_history.append(current_tilt)
            posture_score = max(0.0, 100.0 - current_tilt * 7.0) if (posture_ok and has_face) else 0.0
            self.posture_history.append(posture_score)

            if is_alert:
                if not self.is_in_violation:
                    self.posture_error_count += 1
                    self.is_in_violation = True
                self.current_alert_message = alert_reason
                self.alert_display_until = now + 1.5
            else:
                self.is_in_violation = False
                self.current_alert_message = ""

            err_count = self.posture_error_count

        # Render On-Screen Guide Box & Banner Overlay on Video Frame
        if is_alert:
            # BOLD RED bounding box around 60%x70% guide zone
            cv2.rectangle(image, (center_x_min, center_y_min), (center_x_max, center_y_max), (0, 0, 230), 3, cv2.LINE_AA)
            # Solid RED warning banner at top of frame
            cv2.rectangle(image, (0, 0), (w, 44), (20, 20, 220), -1)
            cv2.putText(
                image,
                banner_text or alert_reason,
                (14, 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            # Clean WHITE guide box around 60%x70% zone
            cv2.rectangle(image, (center_x_min, center_y_min), (center_x_max, center_y_max), (240, 240, 240), 2, cv2.LINE_AA)
            # Calm status banner
            cv2.rectangle(image, (0, 0), (w, 36), (30, 30, 30), -1)
            status_text = f"Posture: Aligned (Tilt: {current_tilt:.1f}°) | Gaze: Good | Alerts: {err_count}"
            cv2.putText(
                image,
                status_text,
                (14, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 230, 120),
                1,
                cv2.LINE_AA,
            )

        if hasattr(frame, "from_ndarray"):
            return frame.from_ndarray(image, format="bgr24")
        return image

    def snapshot_and_reset_question(self) -> dict[str, Any]:
        """Atomically capture metrics for current question and clear buffers for next question."""
        with self.lock:
            audio = [a.copy() for a in self.audio_chunks]
            samples = self.audio_samples
            rate = self.sample_rate
            errors = self.posture_error_count

            # Strictly calculate from actual processed frames; 0.0 if camera off or no face detected
            if self.total_frames_processed == 0 or self.face_detected_frames == 0:
                eye = 0.0
                posture = 0.0
                presence = 0.0
                cam_status = "Camera / Face Not Detected"
            else:
                eye = round((self.gaze_aligned_frames / self.total_frames_processed) * 100.0, 1)
                posture = round((self.posture_aligned_frames / self.total_frames_processed) * 100.0, 1)
                presence = round((self.face_detected_frames / self.total_frames_processed) * 100.0, 1)
                cam_status = "Face Detected & Tracked"

            snapshot = {
                "audio_chunks": audio,
                "audio_duration": float(samples / max(rate, 1)),
                "sample_rate": rate,
                "eye_contact": eye,
                "posture": posture,
                "face_presence": presence,
                "posture_errors": errors,
                "camera_status": cam_status,
            }

            self.audio_chunks.clear()
            self.audio_samples = 0
            self.total_frames_processed = 0
            self.face_detected_frames = 0
            self.gaze_aligned_frames = 0
            self.posture_aligned_frames = 0
            self.gaze_history.clear()
            self.posture_history.clear()
            self.shoulder_tilt_history.clear()
            self.face_seen_history.clear()
            self.violation_start_time = None
            self.is_in_violation = False

            return snapshot

    def current_metrics(self) -> dict[str, Any]:
        """Read rolling metrics for live UI cards without clearing question state."""
        with self.lock:
            if self.total_frames_processed == 0 or self.face_detected_frames == 0:
                eye_pct = 0.0
                posture_pct = 0.0
                cam_status = "Camera / Face Not Detected"
            else:
                eye_pct = round((self.gaze_aligned_frames / self.total_frames_processed) * 100.0, 1)
                posture_pct = round((self.posture_aligned_frames / self.total_frames_processed) * 100.0, 1)
                cam_status = "Face Detected & Tracked"

            return {
                "eye_contact": eye_pct,
                "posture": posture_pct,
                "posture_errors": self.posture_error_count,
                "current_tilt": round(self.shoulder_tilt_history[-1] if self.shoulder_tilt_history else 0.0, 1),
                "camera_status": cam_status,
                "alert": self.current_alert_message if self.is_in_violation else "",
            }


def rtc_configuration() -> dict[str, Any]:
    """Provide standard ICE STUN servers for robust WebRTC video streaming."""
    return {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
        ]
    }


# Compatibility alias
VisionTracker = MediaBuffer
