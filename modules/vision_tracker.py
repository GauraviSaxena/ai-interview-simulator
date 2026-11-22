"""Live eye-contact and posture coaching with MediaPipe FaceMesh/Pose and automatic OpenCV Haar Cascade fallback."""
from __future__ import annotations

import logging
import math
import os
import sys
import threading
import time
import types
from collections import deque
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger("vision_tracker")

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
MAX_SHOULDER_TILT_DEG = 5.0  # Shoulders aligned within +-5 deg
GUIDE_BOX_WIDTH_RATIO = 0.60  # Comfortable 60% width of frame
GUIDE_BOX_HEIGHT_RATIO = 0.70  # Comfortable 70% height of frame
DRIFT_ALERT_THRESHOLD_SEC = 2.0  # Alert persistence window


def _to_ascii_banner(text: str) -> str:
    """Sanitize strings for cv2.putText (OpenCV Hershey font only supports ASCII; unicode turns to '?')."""
    if not text:
        return ""
    clean = (
        text.replace("⚠️", "[!]")
        .replace("❌", "[!]")
        .replace("°", " deg")
        .encode("ascii", "replace")
        .decode("ascii")
        .replace("?", "")
        .strip()
    )
    return clean


def _find_cascade_file(filename: str) -> str:
    """Resolve OpenCV cascade XML from cv2.data, bundled project directory, or system."""
    # 1. Directly in modules/ directory (e.g. modules/haarcascade_*.xml)
    same_dir = Path(__file__).resolve().parent / filename
    if same_dir.is_file():
        return str(same_dir)

    # 2. Bundled in modules/haarcascades/ directory
    bundled = Path(__file__).resolve().parent / "haarcascades" / filename
    if bundled.is_file():
        return str(bundled)

    # 3. Primary: cv2.data.haarcascades
    try:
        cv_data_dir = getattr(cv2.data, "haarcascades", "")
        if cv_data_dir:
            cand = os.path.join(cv_data_dir, filename)
            if os.path.isfile(cand):
                return cand
    except Exception:
        pass

    # 4. Project data/haarcascades directory
    project_bundled = Path(__file__).resolve().parent.parent / "data" / "haarcascades" / filename
    if project_bundled.is_file():
        return str(project_bundled)

    # 5. Project root directory
    root_bundled = Path(__file__).resolve().parent.parent / filename
    if root_bundled.is_file():
        return str(root_bundled)

    return getattr(cv2.data, "haarcascades", "") + filename


class MediaBuffer:
    """Thread-safe processor capturing audio chunks and running live vision & posture tracking."""

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

        # MediaPipe models
        self._mesh = None
        self._pose = None
        self._models_failed = False
        self._mp_init_done = False

        # OpenCV Haar Cascade fallback models
        self._face_cascade: Optional[cv2.CascadeClassifier] = None
        self._eye_cascade: Optional[cv2.CascadeClassifier] = None
        self._cascades_init_done = False

        self.started_at = time.monotonic()

    def _init_models(self):
        """Lazily initialize MediaPipe solutions with clear error logging."""
        if not self._mp_init_done:
            self._mp_init_done = True
            try:
                import mediapipe as mp

                if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh") and hasattr(mp.solutions, "pose"):
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
                    logger.info("MediaPipe FaceMesh and Pose successfully initialized.")
                else:
                    raise AttributeError("mediapipe.solutions is not available on this Python / OS build.")
            except Exception as exc:
                self._models_failed = True
                self._mesh = None
                self._pose = None
                logger.warning(
                    "MediaPipe initialization failed: %s. Enabling automatic OpenCV Haar Cascade fallback.",
                    exc,
                )

    def _init_cascades(self):
        """Lazily initialize OpenCV Haar Cascade classifiers."""
        if not self._cascades_init_done:
            self._cascades_init_done = True
            try:
                face_path = _find_cascade_file("haarcascade_frontalface_default.xml")
                eye_path = _find_cascade_file("haarcascade_eye.xml")

                self._face_cascade = cv2.CascadeClassifier(face_path)
                self._eye_cascade = cv2.CascadeClassifier(eye_path)

                if self._face_cascade.empty():
                    logger.warning("OpenCV face cascade loaded empty from: %s", face_path)
                else:
                    logger.info("OpenCV face cascade loaded from: %s", face_path)

                if self._eye_cascade.empty():
                    logger.warning("OpenCV eye cascade loaded empty from: %s", eye_path)
                else:
                    logger.info("OpenCV eye cascade loaded from: %s", eye_path)
            except Exception as exc:
                logger.error("Failed to initialize OpenCV Haar cascades: %s", exc)

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
        self._init_cascades()

        posture_ok = True
        gaze_ok = True
        current_tilt = 0.0
        face_center_x, face_center_y = 0.5, 0.45
        has_face = False
        face_rect: Optional[tuple[int, int, int, int]] = None
        alert_reason = ""
        banner_text = ""
        detection_source = None

        # Draw centered 60% width x 70% height framing zone for head & shoulders
        center_x_min = int(w * ((1.0 - GUIDE_BOX_WIDTH_RATIO) / 2.0))  # 20%
        center_x_max = int(w * ((1.0 + GUIDE_BOX_WIDTH_RATIO) / 2.0))  # 80%
        center_y_min = int(h * ((1.0 - GUIDE_BOX_HEIGHT_RATIO) / 2.0))  # 15%
        center_y_max = int(h * ((1.0 + GUIDE_BOX_HEIGHT_RATIO) / 2.0))  # 85%

        box_x_min_norm = (1.0 - GUIDE_BOX_WIDTH_RATIO) / 2.0
        box_x_max_norm = (1.0 + GUIDE_BOX_WIDTH_RATIO) / 2.0
        box_y_min_norm = (1.0 - GUIDE_BOX_HEIGHT_RATIO) / 2.0
        box_y_max_norm = (1.0 + GUIDE_BOX_HEIGHT_RATIO) / 2.0

        # Attempt 1: MediaPipe FaceMesh & Pose
        if not self._models_failed and self._mesh:
            try:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mesh_res = self._mesh.process(rgb)

                if mesh_res and mesh_res.multi_face_landmarks:
                    detection_source = "mediapipe"
                    has_face = True
                    pts = mesh_res.multi_face_landmarks[0].landmark
                    nose = pts[1]
                    face_center_x = float(nose.x)
                    face_center_y = float(nose.y)

                    # Bounding box of face from landmarks
                    xs = [p.x for p in pts]
                    ys = [p.y for p in pts]
                    fx = int(min(xs) * w)
                    fy = int(min(ys) * h)
                    fw = int((max(xs) - min(xs)) * w)
                    fh = int((max(ys) - min(ys)) * h)
                    face_rect = (fx, fy, fw, fh)

                    # Check face centering within 60%x70% guide zone
                    if not (box_x_min_norm <= face_center_x <= box_x_max_norm and
                            box_y_min_norm <= face_center_y <= box_y_max_norm):
                        posture_ok = False
                        alert_reason = "POSTURE ALERT: Align Shoulders / Face Not Centered"
                        banner_text = "POSTURE ALERT: Align Shoulders / Face Not Centered"

                    # Iris eye-gaze tracking
                    iris_ids = (468, 473)
                    corners = ((33, 133), (362, 263))
                    eye_scores = []
                    for iris_idx, (c1, c2) in zip(iris_ids, corners):
                        if len(pts) > iris_idx:
                            lo, hi = sorted((pts[c1].x, pts[c2].x))
                            pos = (pts[iris_idx].x - lo) / max(hi - lo, 1e-5)
                            eye_scores.append(max(0.0, 1.0 - abs(pos - 0.5) * 2.2))

                    gaze_score = float(np.mean(eye_scores) * 100.0) if eye_scores else 80.0
                    if gaze_score < 45.0:
                        gaze_ok = False
                        if not alert_reason:
                            alert_reason = "POSTURE ALERT: Maintain direct eye contact with camera"
                            banner_text = "POSTURE ALERT: Maintain Direct Eye Contact"

                    # Pose tracking for shoulder horizontal alignment
                    tilt_found = False
                    if self._pose:
                        try:
                            pose_res = self._pose.process(rgb)
                            if pose_res and pose_res.pose_landmarks:
                                lms = pose_res.pose_landmarks.landmark
                                if len(lms) > 12:
                                    ls = lms[11]
                                    rs = lms[12]
                                    if ls.visibility > 0.4 and rs.visibility > 0.4:
                                        x1, y1 = int(ls.x * w), int(ls.y * h)
                                        x2, y2 = int(rs.x * w), int(rs.y * h)
                                        dx = abs(x2 - x1)
                                        dy = y2 - y1
                                        if dx > 10:
                                            current_tilt = abs(math.atan2(dy, dx) * 180.0 / math.pi)
                                            tilt_found = True
                                            shoulder_color = (0, 200, 100) if current_tilt <= MAX_SHOULDER_TILT_DEG else (0, 0, 230)
                                            cv2.line(image, (x1, y1), (x2, y2), shoulder_color, 2, cv2.LINE_AA)
                                            cv2.putText(image, f"Tilt: {current_tilt:.1f} deg", (min(x1, x2), max(y1, y2) + 22),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, shoulder_color, 1, cv2.LINE_AA)
                        except Exception:
                            pass

                    # If pose didn't provide tilt, compute from eye-line angle (landmarks 33 and 263)
                    if not tilt_found and len(pts) > 263:
                        p1 = pts[33]
                        p2 = pts[263]
                        dx = abs(p2.x - p1.x) * w
                        dy = (p2.y - p1.y) * h
                        if dx > 5:
                            current_tilt = abs(math.atan2(dy, dx) * 180.0 / math.pi)
                            tilt_found = True

                    if current_tilt > MAX_SHOULDER_TILT_DEG:
                        posture_ok = False
                        if not alert_reason:
                            alert_reason = f"POSTURE ALERT: Align Shoulders (Tilt: {current_tilt:.1f} deg > 5.0 deg)"
                            banner_text = f"POSTURE ALERT: Align Shoulders (Tilt: {current_tilt:.1f} deg > 5.0 deg)"
            except Exception as mp_err:
                logger.debug("MediaPipe process frame error: %s", mp_err)
                detection_source = None

        # Attempt 2: OpenCV Haar Cascade Fallback (Linux cloud / headless / MediaPipe failure)
        if detection_source is None and self._face_cascade is not None and not self._face_cascade.empty():
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Histogram equalization ensures robust detection across diverse camera lighting
            gray_eq = cv2.equalizeHist(gray)
            min_face_size = (int(w * 0.12), int(h * 0.12))
            faces = self._face_cascade.detectMultiScale(
                gray_eq,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=min_face_size,
                flags=cv2.CASCADE_SCALE_IMAGE,
            )

            if len(faces) > 0:
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                fx, fy, fw, fh = largest_face
                face_rect = (int(fx), int(fy), int(fw), int(fh))
                has_face = True
                detection_source = "opencv_haar"

                face_center_x = float((fx + fw / 2.0) / w)
                face_center_y = float((fy + fh / 2.0) / h)

                # Check face centering within 60%x70% guide zone
                if not (box_x_min_norm <= face_center_x <= box_x_max_norm and
                        box_y_min_norm <= face_center_y <= box_y_max_norm):
                    posture_ok = False
                    alert_reason = "POSTURE ALERT: Align Shoulders / Face Not Centered"
                    banner_text = "POSTURE ALERT: Align Shoulders / Face Not Centered"

                # Detect eyes in upper 18%-65% of the face ROI
                eye_roi_y1 = int(fy + fh * 0.18)
                eye_roi_y2 = int(fy + fh * 0.65)
                eye_roi = gray_eq[eye_roi_y1:eye_roi_y2, fx:fx + fw]

                detected_eyes = []
                if self._eye_cascade is not None and not self._eye_cascade.empty():
                    min_eye = (int(fw * 0.10), int(fh * 0.08))
                    max_eye = (int(fw * 0.45), int(fh * 0.40))
                    raw_eyes = self._eye_cascade.detectMultiScale(
                        eye_roi,
                        scaleFactor=1.1,
                        minNeighbors=3,
                        minSize=min_eye,
                        maxSize=max_eye,
                    )
                    for (ex, ey, ew, eh) in raw_eyes:
                        detected_eyes.append((fx + ex, eye_roi_y1 + ey, ew, eh))

                # Estimate Shoulder Tilt from eye line angle or face tilt
                tilt_found = False
                if len(detected_eyes) >= 2:
                    sorted_eyes = sorted(detected_eyes, key=lambda e: e[0])
                    e_left = sorted_eyes[0]
                    e_right = sorted_eyes[-1]

                    c1 = (float(e_left[0] + e_left[2] / 2.0), float(e_left[1] + e_left[3] / 2.0))
                    c2 = (float(e_right[0] + e_right[2] / 2.0), float(e_right[1] + e_right[3] / 2.0))

                    dx = abs(c2[0] - c1[0])
                    dy = c2[1] - c1[1]

                    if dx > fw * 0.18:
                        current_tilt = abs(math.atan2(dy, dx) * 180.0 / math.pi)
                        tilt_found = True
                        eye_color = (0, 255, 128) if current_tilt <= MAX_SHOULDER_TILT_DEG else (0, 0, 230)
                        cv2.circle(image, (int(c1[0]), int(c1[1])), 4, eye_color, -1, cv2.LINE_AA)
                        cv2.circle(image, (int(c2[0]), int(c2[1])), 4, eye_color, -1, cv2.LINE_AA)
                        cv2.line(image, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), eye_color, 1, cv2.LINE_AA)
                        cv2.putText(
                            image,
                            f"Tilt: {current_tilt:.1f} deg",
                            (fx, max(20, fy - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            eye_color,
                            1,
                            cv2.LINE_AA,
                        )

                if not tilt_found:
                    # Rolling tilt baseline
                    current_tilt = round(self.shoulder_tilt_history[-1] if self.shoulder_tilt_history else 0.0, 1)

                if current_tilt > MAX_SHOULDER_TILT_DEG:
                    posture_ok = False
                    if not alert_reason:
                        alert_reason = f"POSTURE ALERT: Align Shoulders (Tilt: {current_tilt:.1f} deg > 5.0 deg)"
                        banner_text = f"POSTURE ALERT: Align Shoulders (Tilt: {current_tilt:.1f} deg > 5.0 deg)"

                # Compute Eye-Contact Gaze % from eye detection & symmetry
                if len(detected_eyes) >= 2:
                    gaze_score = 88.0
                    gaze_ok = True
                elif len(detected_eyes) == 1:
                    gaze_score = 65.0
                    gaze_ok = True
                else:
                    if box_x_min_norm <= face_center_x <= box_x_max_norm:
                        gaze_score = 55.0
                        gaze_ok = True
                    else:
                        gaze_score = 30.0
                        gaze_ok = False
                        if not alert_reason:
                            alert_reason = "POSTURE ALERT: Maintain direct eye contact with camera"
                            banner_text = "POSTURE ALERT: Maintain Direct Eye Contact"

        # If no face was detected by either MediaPipe or OpenCV Haar cascade
        if not has_face:
            gaze_score = 0.0
            current_tilt = 0.0
            posture_ok = False
            gaze_ok = False
            alert_reason = "NO FACE DETECTED"
            banner_text = "NO FACE DETECTED: Position Face & Shoulders in Frame"

        # Compute Posture Stability % (centering & stability inside guide box)
        if has_face and posture_ok:
            dx_c = abs(face_center_x - 0.5) / (GUIDE_BOX_WIDTH_RATIO / 2.0)
            dy_c = abs(face_center_y - 0.45) / (GUIDE_BOX_HEIGHT_RATIO / 2.0)
            center_penalty = min(1.0, math.sqrt(dx_c**2 + dy_c**2))
            tilt_penalty = min(1.0, current_tilt / 10.0)
            posture_score = max(50.0, 100.0 - (center_penalty * 20.0 + tilt_penalty * 20.0))
        elif has_face:
            posture_score = max(20.0, 60.0 - current_tilt * 4.0)
        else:
            posture_score = 0.0

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

        # Render On-Screen Guide Box & Status Banner Overlay (Requirement 4)
        if not has_face:
            # 1. NO FACE DETECTED -> BOLD RED bounding box around 60%x70% guide zone & solid RED warning banner
            cv2.rectangle(image, (center_x_min, center_y_min), (center_x_max, center_y_max), (0, 0, 230), 3, cv2.LINE_AA)
            cv2.rectangle(image, (0, 0), (w, 44), (20, 20, 220), -1)
            cv2.putText(
                image,
                _to_ascii_banner(banner_text or "NO FACE DETECTED: Position Face & Shoulders in Frame"),
                (14, 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        elif is_alert:
            # 2. FACE DETECTED BUT OUTSIDE GUIDE BOX OR TILT > 5 deg -> Warning bounding box & alert banner
            cv2.rectangle(image, (center_x_min, center_y_min), (center_x_max, center_y_max), (0, 140, 255), 2, cv2.LINE_AA)
            if face_rect:
                fx, fy, fw, fh = face_rect
                cv2.rectangle(image, (fx, fy), (fx + fw, fy + fh), (0, 140, 255), 2, cv2.LINE_AA)
            cv2.rectangle(image, (0, 0), (w, 44), (20, 100, 220), -1)
            cv2.putText(
                image,
                _to_ascii_banner(banner_text or alert_reason),
                (14, 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            # 3. FACE INSIDE GUIDE BOX & POSTURE ALIGNED -> Green bounding box & "FACE & POSTURE LOCKED" banner (Requirement 4)
            # Green guide box around 60%x70% zone
            cv2.rectangle(image, (center_x_min, center_y_min), (center_x_max, center_y_max), (0, 220, 100), 2, cv2.LINE_AA)
            # Green face bounding box
            if face_rect:
                fx, fy, fw, fh = face_rect
                cv2.rectangle(image, (fx, fy), (fx + fw, fy + fh), (0, 220, 100), 2, cv2.LINE_AA)
                cv2.putText(image, "FACE LOCKED", (fx, max(20, fy - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 100), 1, cv2.LINE_AA)
            # Solid dark-emerald status banner with green "FACE & POSTURE LOCKED" text
            cv2.rectangle(image, (0, 0), (w, 38), (15, 45, 25), -1)
            status_text = f"FACE & POSTURE LOCKED | Tilt: {current_tilt:.1f} deg | Gaze: Good | Alerts: {err_count}"
            cv2.putText(
                image,
                status_text,
                (14, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 128),
                2,
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
