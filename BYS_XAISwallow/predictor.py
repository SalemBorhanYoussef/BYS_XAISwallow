# predictor.py – VideoPredictor mit ROI-Strategien & modularen Utils
import time, threading, queue, logging
from typing import Optional, List, Tuple, Dict, Any

import cv2
import numpy as np
import torch

from slowfast_core import (
    make_slowfast_tensors_gpu,
    make_slowfast_tensors_cpu,  # optionaler Fallback
    Config,
)
from yolo_pose_utils import run_yolo_pose
from vision_utils import (
    ellipse_from_face_points,
    apply_ellipse_mask_inplace,
    estimate_neck_xy,
    crop_square,
)
from ravi_utils import decode_ravi_frame, u16_to_bgr8  # für .ravi Decoding wie im Preprocessing

logger = logging.getLogger("predictor")


class VideoPredictor(threading.Thread):
    def __init__(
        self,
        vpath: str,
        cfg: Config,
        model_sf: torch.nn.Module,
        device: torch.device,
        socketio,
        yolo_weights: str = "yolov8n-pose.pt",
        yolo_conf: float = 0.25,
        roi_size: int = 160,
        roi_mode: str = "live_pose",          # live_pose | first_frame_pose | manual
        fixed_neck_xy: Optional[List[int]] = None,
        yolo_overlay: bool = True,            # Skelett/Boxen zeichnen (links, Display)
        face_mask_mode: str = "live",         # live | static | off  (nur Display links)
        preset_face_ellipse: Optional[Any] = None,  # optional: vorgegebene Ellipse
    ):
        super().__init__(daemon=True)
        self.vpath = vpath
        self.cfg = cfg
        self.model = model_sf
        self.device = device
        self.socketio = socketio

        # Steuerung/Status
        self.stop_flag = threading.Event()
        self.paused = False
        self.eof = False
        self.cmd_q: "queue.Queue[tuple[str, float]]" = queue.Queue()

        # Live-Videobild (MJPEG)
        self._jpeg_lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None

        # SlowFast / Preproc
        self.window_size = int(cfg.window_size)
        self.t_fast = int(cfg.t_fast)
        self.t_slow = max(1, self.t_fast // int(cfg.alpha))
        self.out_hw = (int(cfg.resize_h), int(cfg.resize_w))
        self.normalize = bool(cfg.normalize)
        self.mean = cfg.mean
        self.std = cfg.std
        self.use_amp = cfg.use_amp and (device.type == "cuda") and torch.cuda.is_available()
        self.stride = int(cfg.test_stride)

        # ROI / YOLO
        self.roi_size = int(roi_size)
        self.roi_mode = str(roi_mode)
        self.fixed_neck_xy = tuple(fixed_neck_xy) if fixed_neck_xy else None
        self.yolo_overlay = bool(yolo_overlay)      # nur Anzeige links
        self.face_mask_mode = str(face_mask_mode)   # nur Anzeige links

        # YOLO wird lazy geladen
        self._yolo = None
        self._yolo_weights = yolo_weights
        self._yolo_conf = float(yolo_conf)
        self._yolo_device = "cuda" if torch.cuda.is_available() else "cpu"

        # Buffer für Modellfenster (ROI als RGB)
        self.roi_rgb_buf: List[np.ndarray] = []

        # Video-Metadaten
        self.frames_seen = 0
        self.fps_file: float = float(cfg.fps)
        self.n_frames: int = 0

        # "Static"-Artefakte (aus 1. Frame)
        self.static_neck_xy: Optional[Tuple[int, int]] = None
        # (center, axes, angle) – nur Display links
        self.static_face_ellipse = None
        self._manual_center_override: Optional[Tuple[int,int]] = None  # live updates for manual mode
        if self.roi_mode == "manual" and self.fixed_neck_xy is not None:
            self._manual_center_override = (int(self.fixed_neck_xy[0]), int(self.fixed_neck_xy[1]))

        # Optional: preset Face-Ellipse aus App-Optionen übernehmen
        if preset_face_ellipse is not None:
            self.static_face_ellipse = self._normalize_ellipse(preset_face_ellipse)

    # ========== Public Helpers ==========
    def get_latest_jpeg(self) -> Optional[bytes]:
        with self._jpeg_lock:
            return self._latest_jpeg

    def get_meta(self) -> Tuple[float, int]:
        return self.fps_file, self.n_frames

    # ========== Controls ==========
    def set_paused(self, val: bool):
        self.paused = bool(val)

    # ----- Live Edit APIs (called from Flask routes) -----
    def set_manual_center(self, x: int, y: int):
        if self.roi_mode != "manual":
            raise RuntimeError("ROI mode is not 'manual'")
        self._manual_center_override = (int(x), int(y))

    def set_static_face_ellipse(self, ellipse_params):
        # Accept None or (center(x,y), axes(a,b), angle)
        self.static_face_ellipse = self._normalize_ellipse(ellipse_params)

    def _normalize_ellipse(self, val):
        """Normalize ellipse input to ((cx,cy), (ax1,ax2), angle) or None.
        Accepts dict {center:[x,y], axes:[a,b], angle} or tuple/list [[x,y],[a,b],angle].
        """
        if val is None:
            return None
        try:
            # dict form
            if isinstance(val, dict):
                center = val.get("center")
                axes = val.get("axes")
                angle = float(val.get("angle", 0.0))
                if not (center and axes and len(center) == 2 and len(axes) == 2):
                    return None
                return ((int(center[0]), int(center[1])), (int(axes[0]), int(axes[1])), angle)
            # tuple/list form: (center, axes, angle)
            if isinstance(val, (list, tuple)) and len(val) == 3:
                c, a, ang = val
                cx, cy = int(c[0]), int(c[1])
                ax1, ax2 = int(a[0]), int(a[1])
                return ((cx, cy), (ax1, ax2), float(ang))
        except Exception:
            return None
        return None
    
    def get_edit_state(self) -> Dict[str, Any]:
        return {
            "roi_mode": self.roi_mode,
            "manual_center": list(self._manual_center_override) if self._manual_center_override else None,
            "static_face_ellipse": (
                {
                    "center": list(self.static_face_ellipse[0]),
                    "axes": list(self.static_face_ellipse[1]),
                    "angle": float(self.static_face_ellipse[2]),
                }
                if self.static_face_ellipse else None
            ),
            "face_mask_mode": self.face_mask_mode,
        }

    def enqueue_seek(self, t_sec: float):
        self.cmd_q.put(("seek", float(t_sec)))

    def enqueue_replay(self):
        self.cmd_q.put(("replay", 0.0))

    def stop(self, wait: bool = True, timeout: float = 2.0):
        self.stop_flag.set()
        if wait and self.is_alive():
            try:
                self.join(timeout=timeout)
            except RuntimeError:
                pass
        with self._jpeg_lock:
            self._latest_jpeg = None

    # ========== Internal ==========
    def _emit_meta(self):
        duration = self.n_frames / max(1.0, self.fps_file)
        self.socketio.emit(
            "meta",
            {"fps": float(self.fps_file), "frames": int(self.n_frames), "duration": float(duration)},
        )

    # Einheitliches Einlesen + Decoding (unterstützt .ravi → uint16 → BGR8)
    def _read_frame(self, cap, *, w: int, h: int, is_ravi: bool, little_endian: bool = True):
        """Liest ein Frame und liefert (ok, bgr8) zurück.

        Für .mp4 (oder andere) wird das Roh-BGR unverändert genutzt.
        Für .ravi wird das Roh-Frame in uint16 dekodiert und anschließend per u16_to_bgr8
        in ein invertiertes 8-bit BGR Bild konvertiert (wie im Export/Preprocessing Code),
        damit Modell & Anzeige konsistente Darstellung erhalten.
        """
        ok, raw = cap.read()
        if not ok or raw is None:
            return False, None
        if not is_ravi:
            # Robustheit: flache (1,N) Frames rekonstruieren
            if raw.ndim == 2 and raw.shape[0] == 1 and w > 0 and h > 0:
                flat = raw.reshape(-1)
                expected3 = w * h * 3
                expected1 = w * h
                if flat.size == expected3:
                    bgr = flat.reshape(h, w, 3)
                elif flat.size == expected1:
                    gray = flat.reshape(h, w)
                    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                else:
                    # heuristik: versuche 3 Kanäle mit gegebener Breite
                    if flat.size % (w * 3) == 0:
                        h_guess = flat.size // (w * 3)
                        try:
                            bgr = flat.reshape(h_guess, w, 3)
                        except Exception:
                            bgr = raw
                    else:
                        bgr = raw
            else:
                bgr = raw
        else:
            try:
                u16 = decode_ravi_frame(raw, w, h, little_endian=little_endian)
                bgr = u16_to_bgr8(u16)
            except Exception:
                # Fallback: versuche Rohframe als BGR zu verwenden
                bgr = raw
        # Sicherheits-Resize (Decoder-Abweichungen abfangen)
        if bgr is not None and (bgr.shape[1] != w or bgr.shape[0] != h):
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        return True, bgr

    def _need_yolo_this_frame(self) -> bool:
        """
        YOLO nur ausführen, wenn nötig – spart Rechenzeit.
        - live_pose: ja
        - Overlay links: ja
        - face_mask_mode == live: ja
        - first_frame_pose/static/off: nur im ersten Frame (via _process_first_frame_statics)
        """
        if self.roi_mode == "live_pose":
            return True
        if self.yolo_overlay:
            return True
        if self.face_mask_mode == "live":
            return True
        return False

    def _ensure_yolo(self):
        if self._yolo is None:
            from ultralytics import YOLO
            self._yolo = YOLO(self._yolo_weights)

    def _predict_clip(self, clip_rgb: List[np.ndarray]) -> float:
        """
        Nimmt eine Liste von ROI-RGB-Frames (uint8, H×W×3) und gibt die Wahrscheinlichkeit zurück.
        Nutzt make_slowfast_tensors_gpu (läuft auch auf CPU, wenn device=cpu).
        """
        try:
            slow, fast = make_slowfast_tensors_gpu(
                clip_rgb,
                out_hw=self.out_hw,
                t_fast=self.t_fast,
                t_slow=self.t_slow,
                device=self.device,
                normalize=self.normalize,
                mean=self.mean,
                std=self.std,
            )
        except Exception:
            # Optionaler Fallback, falls die o.g. Funktion mal nicht greift
            try:
                slow, fast = make_slowfast_tensors_cpu(
                    clip_rgb, self.t_fast, self.t_slow, tfm=None
                )
            except Exception:
                # Ultra-Minimal-CPU-Fallback
                import torch.nn.functional as F
                x = torch.from_numpy(np.stack(clip_rgb, axis=0)).float() / 255.0
                x = x.permute(0, 3, 1, 2).contiguous()  # [T,3,H,W]
                x = F.interpolate(x, size=self.out_hw, mode="bilinear", align_corners=False)
                T = x.shape[0]
                idx_fast = torch.linspace(0, max(0, T - 1), steps=self.t_fast).round().long()
                fast = x.index_select(0, idx_fast).permute(1, 0, 2, 3).contiguous()
                idx_slow = torch.linspace(0, fast.shape[1] - 1, steps=self.t_slow).round().long()
                slow = fast.index_select(1, idx_slow).contiguous()

        slow = slow.unsqueeze(0).to(self.device, non_blocking=True)
        fast = fast.unsqueeze(0).to(self.device, non_blocking=True)

        if self.use_amp:
            with torch.cuda.amp.autocast():
                logits = self.model([slow, fast])
        else:
            logits = self.model([slow, fast])

        return float(torch.sigmoid(logits).item())

    def _set_jpeg(self, bgr: np.ndarray):
        ok, jpg = cv2.imencode(".jpg", bgr)
        if ok:
            with self._jpeg_lock:
                self._latest_jpeg = jpg.tobytes()

    def _process_first_frame_statics(self, frame_bgr: np.ndarray):
        """
        Ermittelt (falls konfiguriert) statische ROI-Position sowie statische Gesichtsellipse
        anhand des ersten Frames.
        """
        if self.roi_mode == "manual" and self.fixed_neck_xy is not None:
            self.static_neck_xy = (int(self.fixed_neck_xy[0]), int(self.fixed_neck_xy[1]))
            return

        # Für first_frame_pose (ROI) und/oder face_mask_mode=static (Anzeige) brauchen wir Pose im 1. Frame
        if self.roi_mode == "first_frame_pose" or self.face_mask_mode == "static":
            self._ensure_yolo()
            kp, box, _ = run_yolo_pose(self._yolo, frame_bgr, conf=self._yolo_conf, device=self._yolo_device)

            if self.roi_mode == "first_frame_pose":
                self.static_neck_xy = (
                    estimate_neck_xy(kp, box)
                    or (frame_bgr.shape[1] // 2, int(frame_bgr.shape[0] * 0.55))
                )
            if self.face_mask_mode == "static":
                self.static_face_ellipse = ellipse_from_face_points(kp)

    # --------- WICHTIG: Clean-ROI fürs Modell (aus Original!) ----------
    def _frame_pipeline(self, frame_bgr: np.ndarray, seek_preview: bool = False):
        """
        Verarbeitet 1 Frame:
          - Links (Display): optional YOLO-Overlay + Face-Mask (reine Anzeige)
          - Rechts (Display): **clean ROI** (direkt aus Original cropped)
          - Modell: bekommt **clean ROI** (RGB), keine Maske/keine Annotation
        """
        H, W = frame_bgr.shape[:2]

        # 1) YOLO ausführen, wenn aktuell nötig (nur Anzeige/ROI-Ortung)
        kp = box = None
        annotated_bgr = frame_bgr
        if self._need_yolo_this_frame():
            self._ensure_yolo()
            kp, box, annotated_bgr = run_yolo_pose(
                self._yolo, frame_bgr, conf=self._yolo_conf, device=self._yolo_device
            )

        # 2) Gesichtsmaske (nur Anzeige links)
        face_ellipse = None
        if self.face_mask_mode == "live":
            face_ellipse = ellipse_from_face_points(kp)
        elif self.face_mask_mode == "static":
            face_ellipse = self.static_face_ellipse

        # 3) Linke Anzeige vorbereiten (Overlay & Maske erlaubt)
        left = annotated_bgr.copy() if self.yolo_overlay else frame_bgr.copy()
        if face_ellipse is not None:
            apply_ellipse_mask_inplace(left, face_ellipse, fill=(0, 0, 0))  # nur Anzeige!

        # 4) ROI-Zentrum bestimmen
        if self.roi_mode == "live_pose":
            neck_xy = estimate_neck_xy(kp, box) or (W // 2, int(H * 0.55))
        else:
            if self.roi_mode == "manual" and self._manual_center_override is not None:
                neck_xy = self._manual_center_override
            else:
                neck_xy = self.static_neck_xy or (W // 2, int(H * 0.55))

        # 5) **Clean ROI** aus dem ORIGINAL-FRAME ausschneiden (kein Overlay/keine Maske!)
        roi_bgr, (x1, y1, x2, y2) = crop_square(frame_bgr, neck_xy, self.roi_size)

        # 6) Fürs Modell in RGB in den Fenster-Puffer legen
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        self.roi_rgb_buf.append(roi_rgb)

        # 7) Composite für die Anzeige bauen (links skaliertes Anzeige-Bild, rechts clean ROI)
        left_s = cv2.resize(left, (640, int(640 * H / W)))
        right_s = cv2.resize(roi_bgr, (640, left_s.shape[0]))

        # ROI-Box nur zur Orientierung auf der linken Anzeige einzeichnen
        scale = left_s.shape[1] / float(W)
        cv2.rectangle(
            left_s,
            (int(x1 * scale), int(y1 * scale)),
            (int(x2 * scale), int(y2 * scale)),
            (0, 200, 255),
            2,
        )
        comp = np.hstack([left_s, right_s])

        # 8) Prediction fensterweise (Sliding Window)
        prob = None
        enough = len(self.roi_rgb_buf) >= self.window_size
        at_stride = (self.frames_seen - self.window_size) % self.stride == 0 if enough else False
        if enough and at_stride:
            clip = self.roi_rgb_buf[: self.window_size]
            prob = self._predict_clip(clip)
            # Fenster "schieben"
            self.roi_rgb_buf = self.roi_rgb_buf[self.stride :]

        if prob is not None:
            center_idx = self.frames_seen - self.window_size // 2
            t_sec = center_idx / max(1.0, self.fps_file)
            self.socketio.emit(
                "pred",
                {"t": float(t_sec), "p": float(prob), "kind": ("seek" if seek_preview else "live")},
            )
            cv2.putText(
                comp,
                f"p={prob:.3f}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # 9) JPEG aktualisieren
        self._set_jpeg(comp)

    # ========== Thread Loop ==========
    def run(self):
        cap = cv2.VideoCapture(self.vpath)
        if not cap.isOpened():
            logger.error(f"cannot open video: {self.vpath}")
            return

        # Metadaten
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps_meta = cap.get(cv2.CAP_PROP_FPS)
        if fps_meta and fps_meta > 0:
            self.fps_file = float(fps_meta)
        self._emit_meta()

        # Basis-Meta für Decoding
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        is_ravi = self.vpath.lower().endswith('.ravi')
        # Bei Bedarf könnte little_endian über Config/ENV gesteuert werden; aktuell True wie im UI
        little_endian = True

        frame_period = 1.0 / max(1.0, self.fps_file)
        next_tick = time.perf_counter()

        # Ersten Frame lesen & decodieren → statische Artefakte bestimmen (falls konfiguriert)
        ok, frame = self._read_frame(cap, w=w, h=h, is_ravi=is_ravi, little_endian=little_endian)
        if ok and frame is not None:
            self._process_first_frame_statics(frame)

        while not self.stop_flag.is_set():
            # --- eingehende Kommandos (seek/play/pause/replay) ---
            try:
                while True:
                    cmd, val = self.cmd_q.get_nowait()
                    if cmd == "seek":
                        target_idx = int(round(val * self.fps_file))
                        target_idx = max(0, min(target_idx, max(0, self.n_frames - 1)))
                        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                        self.frames_seen = target_idx
                        self.roi_rgb_buf.clear()
                        self.eof = False
                        ok, frame = self._read_frame(cap, w=w, h=h, is_ravi=is_ravi, little_endian=little_endian)
                        if ok and frame is not None:
                            # falls Face-Mask static aber noch nicht gesetzt: aus Seek-Frame ableiten
                            if self.face_mask_mode == "static" and self.static_face_ellipse is None:
                                self._process_first_frame_statics(frame)
                            self._frame_pipeline(frame, seek_preview=True)
                        self.paused = True  # nach Seek pausieren
                    elif cmd == "replay":
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.frames_seen = 0
                        self.roi_rgb_buf.clear()
                        self.eof = False
                        ok, frame = self._read_frame(cap, w=w, h=h, is_ravi=is_ravi, little_endian=little_endian)
                        # statics neu bestimmen
                        self.static_face_ellipse = None
                        self.static_neck_xy = None
                        if ok and frame is not None:
                            self._process_first_frame_statics(frame)
                        self.paused = False
                    elif cmd == "pause":
                        self.paused = True
                    elif cmd == "play":
                        if self.eof:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            self.frames_seen = 0
                            self.roi_rgb_buf.clear()
                            self.eof = False
                            ok, frame = self._read_frame(cap, w=w, h=h, is_ravi=is_ravi, little_endian=little_endian)
                            self.static_face_ellipse = None
                            self.static_neck_xy = None
                            if ok and frame is not None:
                                self._process_first_frame_statics(frame)
                        self.paused = False
            except queue.Empty:
                pass

            # --- Playback-Loop ---
            if self.paused:
                time.sleep(0.02)
                continue

            if not ok or frame is None:
                self.eof = True
                self.paused = True
                time.sleep(0.02)
                continue

            self.frames_seen += 1
            self._frame_pipeline(frame, seek_preview=False)

            # Echtzeit-Taktung
            now = time.perf_counter()
            if next_tick > now:
                time.sleep(next_tick - now)
            next_tick += frame_period

            ok, frame = self._read_frame(cap, w=w, h=h, is_ravi=is_ravi, little_endian=little_endian)

        cap.release()