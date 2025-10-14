"""video_utils.py – Helferfunktionen für generisches Video-Frame Lesen.

Aktuell enthalten:
  read_rgb_frames(...): Liest eine Sequenz von Frames aus einem MP4 (oder anderem
                        von OpenCV unterstützten) Video und gibt ein RGB-Array
                        mit Shape [T,H,W,3] zurück.

Hinweise:
  - OpenCV liefert BGR → Konvertierung zu RGB erfolgt intern.
  - Optionales Downsampling per `stride` sowie Resize auf (H,W).
  - Nutzt lazy-Liste + np.stack am Ende.
  - Für Streaming (Frame-für-Frame) bitte weiterhin direkten VideoCapture Loop nutzen.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import os
import sys
import cv2
import numpy as np

# Stabiler Import von ravi_utils (funktioniert als Paket & Skript):
_THIS_DIR = os.path.dirname(__file__)
_PARENT = os.path.dirname(_THIS_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
try:
    # Paket-Kontext (wenn BYS_XAISwallow als Paket importiert)
    from .ravi_utils import read_first_frame_bgr  # type: ignore
except Exception:  # Direkt / Skript-Kontext
    from ravi_utils import read_first_frame_bgr  # type: ignore

def read_rgb_frames(
    video_path: str,
    start: int = 0,
    max_frames: Optional[int] = None,
    stride: int = 1,
    resize_hw: Optional[Tuple[int, int]] = None  # (H, W)
) -> np.ndarray:
    """Liest RGB-Frames aus einem Video (typischerweise MP4).

    Args:
        video_path: Pfad zur Video-Datei.
        start: Start-Frameindex (0-basiert), wird via CAP_PROP_POS_FRAMES gesetzt.
        max_frames: Max. Anzahl zu lesender Frames (None = bis EOF).
        stride: Nur jedes k-te Frame behalten (z. B. stride=2 => jedes zweite).
        resize_hw: Optionale Zielgröße (H, W) für jedes behaltene Frame.

    Returns:
        np.ndarray mit Shape [T, H, W, 3] (RGB, uint8).

    Raises:
        RuntimeError: Öffnen der Datei fehlgeschlagen.
        ValueError: Keine Frames gelesen.
    """
    if stride <= 0:
        raise ValueError("stride muss >= 1 sein")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Konnte Video nicht öffnen: {video_path}")

    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    frames: List[np.ndarray] = []
    grabbed = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break
        if grabbed % stride == 0:
            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if resize_hw is not None:
                rh, rw = resize_hw
                frame_rgb = cv2.resize(frame_rgb, (rw, rh), interpolation=cv2.INTER_LINEAR)
            frames.append(frame_rgb)
            if max_frames is not None and len(frames) >= max_frames:
                break
        grabbed += 1

    cap.release()

    if not frames:
        raise ValueError(f"Keine Frames gelesen aus: {video_path}")

    return np.stack(frames, axis=0)

__all__ = ["read_rgb_frames"]

def read_first_frame_any(
    path: str,
    resize_hw: Optional[Tuple[int, int]] = None,
    little_endian: bool = True,
) -> Tuple[np.ndarray, Tuple[int, int, int, float]]:
    """Liest das erste Frame einer Datei (RAVI oder MP4/sonstiges) als RGB.

    RAVI:
      - nutzt bestehende Funktion `read_first_frame_bgr` (inkl. 16-bit Decode + Invertierung)
      - wandelt BGR -> RGB
    MP4 / andere:
      - simple Logik (VideoCapture -> read -> BGR->RGB -> optional resize (INTER_LINEAR))
        genau wie vom Nutzer gewünscht

    Args:
      path: Videopfad
      resize_hw: optionale Zielgröße (H,W)
      little_endian: für .ravi Decoding (weitergereicht)

    Returns:
      (frame_rgb, (w,h,n,fps))
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ravi":
        bgr, meta = read_first_frame_bgr(path, little_endian=little_endian)
        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if resize_hw is not None:
            rh, rw = resize_hw
            frame = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)
        return frame, meta
    # MP4 / andere
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Konnte Video nicht öffnen: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise ValueError("Erstes Frame nicht lesbar")
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if resize_hw is not None:
        rh, rw = resize_hw
        frame = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)
    return frame, (w, h, n, fps)

__all__.append("read_first_frame_any")
