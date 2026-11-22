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
