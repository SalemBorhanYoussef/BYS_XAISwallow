# vision_utils.py – Face-Ellipse/Mask, ROI-Crop, Neck-Schätzer
from typing import Optional, Tuple, Dict
import numpy as np
import cv2

# COCO Indizes (Ultralytics)
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHO, R_SHO = 0, 1, 2, 3, 4, 5, 6

def ellipse_from_face_points(
    kp: Optional[np.ndarray],
    min_vis_eye=0.10,
    min_vis_nose=0.20,
    min_vis_ear=0.10,
    profile_boost=1.25
) -> Optional[Tuple[Tuple[int,int], Tuple[int,int], float]]:
    if kp is None or kp.shape[0] <= R_EAR:
        return None
    nx, ny, ns = kp[NOSE]
    lx, ly, ls = kp[L_EYE]
    rx, ry, rs = kp[R_EYE]
    lex, ley, les = kp[L_EAR]
    rex, rey, res = kp[R_EAR]
    if ns < min_vis_nose:
        return None

    have_Leye, have_Reye = ls >= min_vis_eye, rs >= min_vis_eye
    have_Lear, have_Rear = les >= min_vis_ear, res >= min_vis_ear

    if have_Leye and have_Reye:
        ex = (lx + rx) / 2.0
        ey = (ly + ry) / 2.0
        eye_vec = np.array([rx - lx, ry - ly], float)
        eye_dist = float(np.hypot(*eye_vec))
        if eye_dist < 2.0:
            return None
        angle = float(np.degrees(np.arctan2(eye_vec[1], eye_vec[0])))
        width = 1.9 * eye_dist
        height = 2.5 * eye_dist
        cx = int(round(ex + 0.35 * (nx - ex)))
        cy = int(round(ey + 0.35 * (ny - ey)))
        axes = (int(round(width / 2)), int(round(height / 2)))
        return (cx, cy), axes, angle

    ear_cands = []
    if have_Lear:
        ear_cands.append((lex, ley))
    if have_Rear:
        ear_cands.append((rex, rey))
    if ear_cands:
        dists = [np.hypot(nx - ex, ny - ey) for (ex, ey) in ear_cands]
        ex, ey = ear_cands[int(np.argmin(dists))]
        head_vec = np.array([ex - nx, ey - ny], float)
        head_len = float(np.hypot(*head_vec))
        if head_len < 2.0:
            return None
        angle = float(np.degrees(np.arctan2(head_vec[1], head_vec[0])))
        width = 1.6 * head_len * profile_boost
        height = 1.8 * head_len
        cx = int(round(nx + 0.45 * head_vec[0]))
        cy = int(round(ny + 0.45 * head_vec[1]))
        axes = (int(round(width / 2)), int(round(height / 2)))
        return (cx, cy), axes, angle

    if have_Leye or have_Reye:
        if have_Leye and not have_Reye:
            rx = nx + (nx - lx)
            ry = ly
        elif have_Reye and not have_Leye:
            lx = nx + (nx - rx)
            ly = ry
        ex = (lx + rx) / 2.0
        ey = (ly + ry) / 2.0
        eye_vec = np.array([rx - lx, ry - ly], float)
        eye_dist = float(np.hypot(*eye_vec))
        if eye_dist < 2.0:
            return None
        angle = float(np.degrees(np.arctan2(eye_vec[1], eye_vec[0])))
        width = 1.9 * eye_dist
        height = 2.5 * eye_dist
        cx = int(round(ex + 0.35 * (nx - ex)))
        cy = int(round(ey + 0.35 * (ny - ey)))
        axes = (int(round(width / 2)), int(round(height / 2)))
        return (cx, cy), axes, angle

    return None

def apply_ellipse_mask_inplace(bgr: np.ndarray, ellipse_params, fill=(0, 0, 0), mode: str = "inside"):
    """Apply ellipse mask.

    mode:
      'inside'  -> nur die Ellipse wird mit 'fill' überschrieben (außen bleibt original)
      'outside' -> außerhalb der Ellipse füllen (invertiert)
    """
    if ellipse_params is None:
        return
    H, W = bgr.shape[:2]
    mask = np.zeros((H, W), np.uint8)
    center, axes, angle = ellipse_params
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 2)
    if mode == "outside":
        mask = 255 - mask
    m = (mask.astype(np.float32) / 255.0)[..., None]
    bgr[:] = (bgr.astype(np.float32) * (1.0 - m) + np.array(fill, np.float32) * m).astype(np.uint8)

def estimate_neck_xy(kp: Optional[np.ndarray], box: Optional[Dict[str, float]], vis_thr=0.15):
    if kp is not None and kp.shape[0] > R_SHO:
        lx, ly, ls = kp[L_SHO]
        rx, ry, rs = kp[R_SHO]
        nx, ny, ns = kp[NOSE]
        if ls >= vis_thr and rs >= vis_thr:
            return int(round((lx + rx)/2)), int(round((ly + ry)/2))
        if ls >= vis_thr and ns >= vis_thr:
            return int(round((lx + nx)/2)), int(round((ly + ny)/2 + 4))
        if rs >= vis_thr and ns >= vis_thr:
            return int(round((rx + nx)/2)), int(round((ry + ny)/2 + 4))
        if ns >= vis_thr:
            return int(round(nx)), int(round(ny + 20))
    if box is not None:
        x1,y1,x2,y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        return int(round((x1 + x2)/2)), int(round((y1 + y2)*0.55))
    return None

def crop_square(bgr: np.ndarray, center_xy, size: int):
    h, w = bgr.shape[:2]
    cx, cy = center_xy
    half = size // 2
    x1, x2 = max(0, cx - half), min(w, cx + half)
    y1, y2 = max(0, cy - half), min(h, cy + half)
    roi = bgr[y1:y2, x1:x2].copy()
    if roi.size == 0:
        roi = np.zeros((size, size, 3), np.uint8)
    else:
        roi = cv2.resize(roi, (size, size))
    return roi, (x1, y1, x2, y2)
