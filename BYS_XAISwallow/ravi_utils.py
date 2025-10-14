# ravi_utils.py
import cv2
import numpy as np
import os

def decode_ravi_frame(raw_frame: np.ndarray, w: int, h: int, little_endian: bool = True) -> np.ndarray:
    """
    Robust: konvertiert OpenCV-Rohframe zu uint16 [h,w].
    """
    if raw_frame is None:
        return None

    if raw_frame.ndim == 2 and raw_frame.shape == (h, w) and raw_frame.dtype == np.uint16:
        arr16 = raw_frame
    elif raw_frame.ndim == 2 and raw_frame.shape[0] == 1 and raw_frame.dtype == np.uint8:
        flat = raw_frame.reshape(-1)
        need = w * h * 2
        if flat.size < need:
            flat = np.pad(flat, (0, need - flat.size), mode="edge")
        arr16 = flat[:need].view(np.uint16).reshape(h, w)
    else:
        flat = raw_frame.astype(np.uint8).reshape(-1)
        need = w * h * 2
        if flat.size < need:
            flat = np.pad(flat, (0, need - flat.size), mode="edge")
        arr16 = flat[:need].view(np.uint16).reshape(h, w)

    if not little_endian:
        arr16 = arr16.byteswap().newbyteorder()

    return arr16


def u16_to_bgr8(img_u16: np.ndarray) -> np.ndarray:
    """
    16-bit Graustufe → invertiertes 8-bit BGR; erste Zeile entfernt (wie in deinem Skript).
    """
    if img_u16 is None:
        return None
    f32 = img_u16.astype(np.float32)
    f32 = f32[1:, :]  # ROI ab Zeile 1
    bgr = cv2.normalize(f32, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    bgr8 = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    return 255 - bgr8


def crop_center_square(img_bgr: np.ndarray, center_xy, size: int) -> np.ndarray:
    """
    Quadratisches ROI um (x,y), an Ränder geclippt, auf size skaliert.
    """
    h, w = img_bgr.shape[:2]
    cx, cy = int(center_xy[0]), int(center_xy[1])
    half = size // 2
    x1, x2 = max(0, cx - half), min(w, cx + half)
    y1, y2 = max(0, cy - half), min(h, cy + half)
    roi = img_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros((size, size, 3), np.uint8)
    return cv2.resize(roi, (size, size), interpolation=cv2.INTER_AREA)


def read_first_frame_bgr(path: str, little_endian: bool = True):
    """Liest erstes Frame eines Videos und liefert ein 8-bit BGR Bild + Meta.

    Verhalten:
      - .ravi: Rohframe → uint16 Decoding → invertiertes 8-bit BGR (wie Export & Live Pipeline)
      - andere (z.B. .mp4): Roh-BGR direkt (keine Invertierung / kein künstliches Decoding)
    """
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_FORMAT, -1)
    if not cap.isOpened():
        raise RuntimeError(f"Kann Datei nicht öffnen: {path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0

    ok, raw = cap.read()
    if not ok or raw is None:
        cap.release()
        raise RuntimeError("Erstes Frame nicht lesbar.")

    ext = os.path.splitext(path)[1].lower()
    if ext == '.ravi':
        u16 = decode_ravi_frame(raw, w, h, little_endian=little_endian)
        bgr = u16_to_bgr8(u16)
    else:
        # MP4: Falls als flache Zeile geliefert (1,N), zuerst immer als RGB interpretieren
        if raw.ndim == 2 and raw.shape[0] == 1:
            flat = raw.reshape(-1)
            expected3 = w * h * 3
            if w > 0 and h > 0 and flat.size == expected3:
                bgr = flat.reshape(h, w, 3)
            else:
                # Falls keine exakte Übereinstimmung: fallback Graustufe oder heuristik
                expected1 = w * h
                if w > 0 and h > 0 and flat.size == expected1:
                    gray = flat.reshape(h, w)
                    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                else:
                    # Letzter Versuch: nehme Breite an und teile durch 3
                    if w > 0 and flat.size % (w * 3) == 0:
                        h_guess = flat.size // (w * 3)
                        try:
                            bgr = flat.reshape(h_guess, w, 3)
                            h = h_guess
                        except Exception:
                            bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
                    else:
                        bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        else:
            # Normalfall: raw ist bereits BGR8 / Graustufe / BGRA
            if raw.ndim == 2 or (raw.ndim == 3 and raw.shape[2] == 1):
                bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
            elif raw.ndim == 3 and raw.shape[2] == 3:
                bgr = raw
            elif raw.ndim == 3 and raw.shape[2] == 4:  # BGRA → BGR
                bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
            else:
                # Fallback: best effort crop
                bgr = raw[:, :, :3].copy()

    # Sicherheits-Resize, falls Decoder andere Größe liefert
    if bgr.shape[1] != w or bgr.shape[0] != h:
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    cap.release()
    return bgr, (w, h, n, fps)
