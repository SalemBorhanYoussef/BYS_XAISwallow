"""dataproc.py – erweitert: Auto-Neck (YOLO Pose) & optionale Ellipse-Maske beim Export.

Routes:
  GET  /preprocess/                -> Seite
  POST /preprocess/upload          -> (optional) Upload (Fallback)
  POST /preprocess/first_frame     -> Erstes Frame + Meta
  POST /preprocess/auto_point      -> YOLO Pose auf erstem Frame (Neck + Face-Ellipse)
  POST /preprocess/run             -> Export Full & ROI (optional Ellipse-Mask)
"""

import os
import base64
import time
from typing import Optional, Dict, Any
import cv2
import numpy as np
from flask import Blueprint, current_app, request, jsonify, render_template

from ravi_utils import (
    decode_ravi_frame,
    u16_to_bgr8,
)
from video_utils import read_first_frame_any
from yolo_pose_utils import run_yolo_pose
from vision_utils import estimate_neck_xy, ellipse_from_face_points

bp = Blueprint("preprocess", __name__, url_prefix="/preprocess")

# --------------------------------------------------
# YOLO Lazy Cache
# --------------------------------------------------
_YOLO_MODEL = None
_YOLO_WEIGHTS_LAST = None


def _get_yolo(weights: str):
    global _YOLO_MODEL, _YOLO_WEIGHTS_LAST
    if _YOLO_MODEL is None or _YOLO_WEIGHTS_LAST != weights:
        from ultralytics import YOLO

        _YOLO_MODEL = YOLO(weights)
        _YOLO_WEIGHTS_LAST = weights
    return _YOLO_MODEL


def _outputs_dir() -> str:
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


@bp.route("/browse", methods=["GET"])
def browse_fs():
    """Einfache Server-seitige Dateibrowsing-API.

    Query Params:
      path: optional – Startpfad. Default: aktuelles Arbeitsverzeichnis.
      ext: optional – Filter (z.B. ".ravi,.mp4").
    """
    root = request.args.get("path") or os.getcwd()
    exts_raw = request.args.get("ext", "")
    filters = [e.strip().lower() for e in exts_raw.split(",") if e.strip()]
    try:
        root = os.path.abspath(root)
        if not os.path.exists(root):
            return jsonify({"ok": False, "error": "path not exists"}), 400
        # Wenn root eine Datei ist → Liste vom Parent + selection
        if os.path.isfile(root):
            dir_path = os.path.dirname(root)
        else:
            dir_path = root
        entries = []
        for name in sorted(os.listdir(dir_path)):
            full = os.path.join(dir_path, name)
            try:
                is_dir = os.path.isdir(full)
            except OSError:
                continue
            if not is_dir:
                if filters:
                    ext = os.path.splitext(name)[1].lower()
                    if ext not in filters:
                        continue
                size = None
                try:
                    size = os.path.getsize(full)
                except OSError:
                    size = None
                entries.append({"name": name, "path": full, "dir": False, "size": size})
            else:
                entries.append({"name": name, "path": full, "dir": True})
        parent = os.path.dirname(dir_path) if dir_path != os.path.dirname(dir_path) else dir_path
        return jsonify({
            "ok": True,
            "path": dir_path,
            "parent": parent,
            "entries": entries,
        })
    except Exception as ex:  # pragma: no cover
        return jsonify({"ok": False, "error": str(ex)}), 500


@bp.route("/", methods=["GET"])
def page():
    return render_template("preprocess.html")


@bp.route("/upload", methods=["POST"])
def upload_ravi():
    f = request.files.get("video")
    if not f:
        return jsonify({"ok": False, "error": "no file"}), 400
    updir = current_app.config.get("UPLOAD_FOLDER")
    os.makedirs(updir, exist_ok=True)
    fname = f.filename
    save_path = os.path.join(updir, fname)
    f.save(save_path)
    return jsonify({"ok": True, "path": save_path, "url": f"/uploads/{fname}"})


@bp.route("/first_frame", methods=["POST"])
def first_frame():
    data = request.get_json(force=True, silent=True) or {}
    path = data.get("path")
    little = bool(data.get("little_endian", True))
    if not path or not os.path.exists(path):
        return jsonify({"ok": False, "error": "path missing"}), 400
    try:
        # unified first-frame loader (mp4 vs ravi)
        frame_rgb, (w, h, n, fps) = read_first_frame_any(path, little_endian=little, resize_hw=None)
        # für PNG Encoding BGR benötigt
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        ok, png = cv2.imencode(".png", bgr)
        if not ok:
            raise RuntimeError("encode failed")
        b64 = base64.b64encode(png.tobytes()).decode("ascii")
        return jsonify({
            "ok": True,
            "meta": {"w": w, "h": h, "frames": n, "fps": fps},
            "image": f"data:image/png;base64,{b64}",
        })
    except Exception as ex:  # pragma: no cover - robust fallback
        return jsonify({"ok": False, "error": str(ex)}), 500


@bp.route("/auto_point", methods=["POST"])
def auto_point():
    """YOLO Pose auf erstem Frame: liefert Neck und optionale Face-Ellipse.

    Request JSON:
      path: str
      little_endian: bool
      yolo_weights: str
      yolo_conf: float
    Response JSON:
      ok: bool
      neck: {x,y} | null
      ellipse: {center:[x,y], axes:[a,b], angle:deg} | null
      preview: dataURL (annotiertes Bild)
    """

    data = request.get_json(force=True, silent=True) or {}
    path = data.get("path")
    little = bool(data.get("little_endian", True))
    weights = data.get("yolo_weights", "yolov8n-pose.pt")
    conf = float(data.get("yolo_conf", 0.25))
    if not path or not os.path.exists(path):
        return jsonify({"ok": False, "error": "path missing"}), 400
    try:
        frame_rgb, (w, h, n, fps) = read_first_frame_any(path, little_endian=little, resize_hw=None)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        yolo = _get_yolo(weights)
        kp, box, annotated = run_yolo_pose(yolo, frame_bgr.copy(), conf=conf)
        neck_xy = estimate_neck_xy(kp, box) if (kp is not None or box is not None) else None
        ell = ellipse_from_face_points(kp) if kp is not None else None

        # Overlay zeichnen
        if ell is not None:
            center, axes, angle = ell
            cv2.ellipse(annotated, center, axes, angle, 0, 360, (0, 255, 0), 2)
        if neck_xy is not None:
            cv2.circle(annotated, neck_xy, 5, (255, 0, 255), -1)

        img_b64 = None
        ok_png, png = cv2.imencode(".png", annotated)
        if ok_png:
            img_b64 = "data:image/png;base64," + base64.b64encode(png.tobytes()).decode(
                "ascii"
            )

        resp: Dict[str, Any] = {
            "ok": True,
            "neck": {"x": neck_xy[0], "y": neck_xy[1]} if neck_xy else None,
            "ellipse": None,
            "preview": img_b64,
        }
        if ell is not None:
            (cx, cy), (ax1, ax2), angle = ell
            resp["ellipse"] = {
                "center": [cx, cy],
                "axes": [ax1, ax2],
                "angle": angle,
            }
        return jsonify(resp)
    except Exception as ex:  # pragma: no cover
        return jsonify({"ok": False, "error": str(ex)}), 500


@bp.route("/run", methods=["POST"])
def run_export():
    """Export Full & ROI (optional statische Ellipse-Mask).

    Request JSON:
      path: str
      cx, cy: int  (Neck/Center)
      roi_size: int
      little_endian: bool
      ellipse: {center:[x,y], axes:[a,b], angle:deg} (optional)
    """

    data = request.get_json(force=True, silent=True) or {}
    path = data.get("path")
    cx, cy = int(data.get("cx", -1)), int(data.get("cy", -1))
    roi_size = int(data.get("roi_size", 224))
    little = bool(data.get("little_endian", True))
    ellipse = data.get("ellipse")
    roi_mask = bool(data.get("roi_mask", False))

    if not path or not os.path.exists(path):
        return jsonify({"ok": False, "error": "path missing"}), 400
    if cx < 0 or cy < 0:
        return jsonify({"ok": False, "error": "center not set"}), 400

    out_dir = _outputs_dir()
    stem = os.path.splitext(os.path.basename(path))[0]
    out_full = os.path.join(out_dir, f"{stem}.mp4")
    out_roi = os.path.join(out_dir, f"{stem}_roi.mp4")

    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_FORMAT, -1)
    if not cap.isOpened():
        return jsonify({"ok": False, "error": "cannot open video"}), 400

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw_full = cv2.VideoWriter(out_full, fourcc, fps, (w, h), True)
    vw_roi = cv2.VideoWriter(out_roi, fourcc, fps, (roi_size, roi_size), True)
    if not vw_full.isOpened() or not vw_roi.isOpened():
        cap.release()
        return jsonify({"ok": False, "error": "VideoWriter open failed"}), 500

    # Ellipse vorbereiten + Mask einmalig vorrechnen
    ellipse_params: Optional[Any] = None
    mask_full: Optional[np.ndarray] = None
    if ellipse and isinstance(ellipse, dict):
        try:
            c = ellipse.get("center")
            a = ellipse.get("axes")
            ang = float(ellipse.get("angle", 0.0))
            if c and a:
                ellipse_params = ((int(c[0]), int(c[1])), (int(a[0]), int(a[1])), float(ang))
        except Exception:  # pragma: no cover
            ellipse_params = None
    if ellipse_params is not None:
        # Maske (nur einmal) erzeugen; 'inside' => Ellipsenbereich schwarz
        mask_full = np.zeros((h, w), np.uint8)
        center, axes, ang = ellipse_params
        cv2.ellipse(mask_full, center, axes, ang, 0, 360, 255, -1)
        # Leichter Blur für weichere Grenze (ähnlich vorherigem GaussianBlur in apply_ellipse_mask_inplace)
        mask_full = cv2.GaussianBlur(mask_full, (0, 0), 2)

    # Precompute ROI crop source window (sofern möglich) – ROI bleibt ja statisch
    half = roi_size // 2
    # Clamp ROI center
    rx1 = max(0, min(w - roi_size, cx - half))
    ry1 = max(0, min(h - roi_size, cy - half))
    rx2 = rx1 + roi_size
    ry2 = ry1 + roi_size

    frames_written = 0
    t0 = time.perf_counter()
    is_ravi = path.lower().endswith('.ravi')
    while True:
        ok, raw = cap.read()
        if not ok or raw is None:
            break
        # Frame Dekodierung abhängig vom Format
        if is_ravi:
            u16 = decode_ravi_frame(raw, w, h, little_endian=little)
            bgr = u16_to_bgr8(u16)
        else:
            # Normales Video: raw ist schon BGR 8-bit
            bgr = raw
        # Sicherheit: Größen-Anpassung falls Decoder abweicht
        if bgr.shape[1] != w or bgr.shape[0] != h:
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)

        if mask_full is not None:
            # Schnelles Füllen der Ellipse mit Schwarz (inside)
            # mask_full 255 => inside ellipse
            if bgr.ndim == 2:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
            m = (mask_full > 0)
            bgr[m] = (0, 0, 0)
        vw_full.write(bgr)

        # ROI extrahieren (schneller, da wir bounding window kennen)
        roi_src = bgr[ry1:ry2, rx1:rx2]
        if roi_src.shape[0] != roi_size or roi_src.shape[1] != roi_size:
            roi_src = cv2.resize(roi_src, (roi_size, roi_size), interpolation=cv2.INTER_AREA)
        roi_frame = roi_src.copy()
        if roi_mask and mask_full is not None:
            # Entsprechenden Mask-Ausschnitt auf ROI anwenden
            mask_roi = mask_full[ry1:ry2, rx1:rx2]
            if mask_roi.shape[0] != roi_size or mask_roi.shape[1] != roi_size:
                mask_roi = cv2.resize(mask_roi, (roi_size, roi_size), interpolation=cv2.INTER_AREA)
            mr = (mask_roi > 0)
            roi_frame[mr] = (0, 0, 0)
        vw_roi.write(roi_frame)
        frames_written += 1

    cap.release()
    vw_full.release()
    vw_roi.release()
    dt = time.perf_counter() - t0
    duration = frames_written / fps if fps > 0 else 0.0

    return jsonify({
        "ok": True,
        "frames": frames_written,
        "fps_src": fps,
        "duration": duration,
        "time_sec": dt,
        "ms_per_frame": (dt / frames_written * 1000.0) if frames_written else None,
        "full_url": f"/outputs/{os.path.basename(out_full)}",
        "roi_url": f"/outputs/{os.path.basename(out_roi)}",
        "ellipse_used": bool(ellipse_params),
        "fast_path": True,
        "ravi": is_ravi,
    })

