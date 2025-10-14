# app.py – schlanker Server: lädt Modell, Optionen, startet Predictor
import os
import time
import logging
from dotenv import load_dotenv
from dataclasses import replace
from typing import Optional
import torch
from flask import Flask, render_template, request, Response, send_from_directory, jsonify
from flask_socketio import SocketIO

from slowfast_core import Config, load_model
from predictor import VideoPredictor

from dataproc import bp as preprocess_bp
from training_bp import bp_train

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("swallow-app")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev")
app.config["UPLOAD_FOLDER"] = os.getenv("UPLOAD_DIR", os.path.join(os.path.dirname(__file__), "uploads"))
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
app.register_blueprint(preprocess_bp)
app.register_blueprint(bp_train)
OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "outputs"))
os.makedirs(OUTPUT_DIR, exist_ok=True)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

forced_device = os.getenv("DEVICE", "auto").lower()
if forced_device in ("cuda", "gpu"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
elif forced_device == "cpu":
    device = torch.device("cpu")
else:  # auto
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device={device}")
model: Optional[torch.nn.Module] = None
cfg: Optional[Config] = None
predictor: Optional[VideoPredictor] = None

# App-Optionen (lassen sich via /load_model überschreiben)
APP_OPTS = {
    "yolo_weights": "yolov8n-pose.pt",
    "yolo_conf": 0.25,
    "roi_size": 160,
    "roi_mode": "live_pose",          # live_pose | first_frame_pose | manual
    "fixed_neck_xy": None,            # z.B. [420, 300] (bei roi_mode="manual")
    "yolo_overlay": True,             # Skelett/Boxen rendern
    "face_mask_mode": "live",         # live | static | off
    "preset_face_ellipse": None,       # {(center:[x,y],axes:[a,b],angle:deg)} wenn static
}

def _stop_predictor():
    global predictor
    if predictor is not None:
        try:
            predictor.stop()
        except Exception:
            pass
        predictor = None

@app.route("/")
def index():
    return render_template("index.html", has_model=(model is not None))

@app.route("/load_model", methods=["POST"])
def load_model_route():
    global model, cfg, device, APP_OPTS
    data = request.get_json(force=True, silent=True) or {}
    base = Config()
    # Config-Felder
    for k, v in data.items():
        if hasattr(base, k):
            base = replace(base, **{k: v})
    # App-Optionen
    for k in list(APP_OPTS.keys()):
        if k in data:
            APP_OPTS[k] = data[k]

    # Normalize preset_face_ellipse structure (dict -> tuple) lazily later in predictor

    if not getattr(base, "ckpt", None):
        return jsonify({"ok": False, "error": "ckpt path required"}), 400

    cfg = base
    logger.info(f"[/load_model] ckpt={cfg.ckpt} model={cfg.model} "
                f"roi_mode={APP_OPTS['roi_mode']} roi_size={APP_OPTS['roi_size']} "
                f"overlay={APP_OPTS['yolo_overlay']} face_mask={APP_OPTS['face_mask_mode']}")
    try:
        model = load_model(cfg, device)
        return jsonify({"ok": True, "device": device.type})
    except Exception as ex:
        logger.exception("load_model failed")
        return jsonify({"ok": False, "error": str(ex)}), 500

@app.route("/upload", methods=["POST"])
def upload():
    global predictor, model, cfg, APP_OPTS
    if model is None or cfg is None:
        return jsonify({"ok": False, "error": "Model not loaded"}), 400

    file = request.files.get("video")
    if not file or not file.filename:
        return jsonify({"ok": False, "error": "No file provided"}), 400

    save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(save_path)

    _stop_predictor()
    predictor = VideoPredictor(
        save_path, cfg, model, device, socketio=socketio,
        **APP_OPTS
    )
    predictor.start()
    return jsonify({"ok": True, "path": save_path})

@app.route("/use_path", methods=["POST"])
def use_path():
    global predictor, model, cfg, APP_OPTS
    data = request.get_json(force=True) or {}
    vpath = data.get("path")
    if not vpath or not os.path.exists(vpath):
        return jsonify({"ok": False, "error": "Path missing or not found"}), 400
    if model is None or cfg is None:
        return jsonify({"ok": False, "error": "Model not loaded"}), 400

    _stop_predictor()
    predictor = VideoPredictor(
        vpath, cfg, model, device, socketio=socketio,
        **APP_OPTS
    )
    predictor.start()
    return jsonify({"ok": True})

@app.route("/stop", methods=["POST"])
def stop_stream():
    _stop_predictor()
    return jsonify({"ok": True})

@app.route("/video_feed")
def video_feed():
    def gen():
        boundary = b"--frame\r\n"
        while True:
            if predictor is None:
                time.sleep(0.05)
                continue
            jpg = predictor.get_latest_jpeg()
            if jpg is None:
                time.sleep(0.01)
                continue
            yield boundary
            yield b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.001)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/uploads/<path:fn>")
def serve_upload(fn):
    return send_from_directory(app.config["UPLOAD_FOLDER"], fn)

@app.route("/video_meta")
def video_meta():
    if predictor is None:
        return jsonify({"fps": 0.0, "frames": 0, "duration": 0.0})
    fps, frames = predictor.get_meta()
    return jsonify({"fps": fps, "frames": frames, "duration": frames/max(1.0, fps)})

@app.route("/edit/roi", methods=["POST"])
def set_manual_roi():
    """Set a manual neck/ROI center while in manual roi_mode.
    Body JSON: {"x":int, "y":int}
    """
    global predictor
    if predictor is None:
        return jsonify({"ok": False, "error": "no predictor"}), 400
    data = request.get_json(force=True, silent=True) or {}
    x, y = data.get("x"), data.get("y")
    if x is None or y is None:
        return jsonify({"ok": False, "error": "x,y required"}), 400
    try:
        predictor.set_manual_center(int(x), int(y))
        return jsonify({"ok": True})
    except Exception as ex:
        return jsonify({"ok": False, "error": str(ex)}), 500

@app.route("/edit/ellipse", methods=["POST"])
def set_face_ellipse():
    """Set or clear a face ellipse. Body: {center:[x,y], axes:[a,b], angle:float} or {clear:true}"""
    global predictor
    if predictor is None:
        return jsonify({"ok": False, "error": "no predictor"}), 400
    data = request.get_json(force=True, silent=True) or {}
    if data.get("clear"):
        predictor.set_static_face_ellipse(None)
        return jsonify({"ok": True, "cleared": True})
    try:
        center = data.get("center")
        axes = data.get("axes")
        angle = float(data.get("angle", 0.0))
        if not (center and axes and len(center)==2 and len(axes)==2):
            return jsonify({"ok": False, "error": "center[2], axes[2] required"}), 400
        predictor.set_static_face_ellipse(((int(center[0]), int(center[1])), (int(axes[0]), int(axes[1])), angle))
        return jsonify({"ok": True})
    except Exception as ex:
        return jsonify({"ok": False, "error": str(ex)}), 500

@app.route("/edit/state", methods=["GET"])
def get_edit_state():
    """Return current editable state (roi_mode, manual center, static ellipse, mask mode).
    Useful for front-end initialization after page load/refresh.
    """
    if predictor is None:
        return jsonify({"ok": False, "error": "no predictor"}), 400
    try:
        st = predictor.get_edit_state()
        return jsonify({"ok": True, **st})
    except Exception as ex:  # pragma: no cover
        return jsonify({"ok": False, "error": str(ex)}), 500

@socketio.on("control")
def socket_control(msg):
    if predictor is None:
        return
    action = (msg or {}).get("action")
    if action == "pause":
        predictor.set_paused(True)
    elif action == "play":
        predictor.set_paused(False)
    elif action == "seek":
        t = float((msg or {}).get("t", 0.0))
        predictor.enqueue_seek(t)
    elif action == "replay":
        predictor.enqueue_replay()
        
        
@app.route("/outputs/<path:fn>")
def serve_outputs(fn):
    return send_from_directory(OUTPUT_DIR, fn)

if __name__ == "__main__":
    logger.info("[BOOT] http://localhost:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, allow_unsafe_werkzeug=True)