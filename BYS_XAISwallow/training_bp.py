import os
import json
from flask import Blueprint, request, jsonify, render_template

# Robust import: works when run as package (python -m BYS_XAISwallow.app)
# and when run as a script from inside the BYS_XAISwallow folder.
try:
    from .training_service import training_manager  # type: ignore
except Exception:  # pragma: no cover - fallback for script execution
    from training_service import training_manager  # type: ignore


bp_train = Blueprint("train", __name__, url_prefix="/train")


@bp_train.route("/", methods=["GET"])
def page_train():
    # Render training UI
    # Provide defaults for paths on server side
    defaults = {
        "script": "train_multi_new.py",
        "python": "python",
        "workdir": os.path.dirname(os.path.dirname(__file__)),
    }
    return render_template("train.html", defaults=defaults)


@bp_train.route("/api/start", methods=["POST"])
def api_start():
    data = request.get_json(force=True, silent=True) or {}
    # Build command: python train_multi_new.py --config-json '...'
    python_exe = data.get("python", "python")
    script_path = data.get("script", "train_multi_new.py")
    workdir = data.get("workdir") or os.path.dirname(os.path.dirname(__file__))
    cfg = data.get("config", {})

    if not os.path.isabs(script_path):
        script_path = os.path.join(workdir, script_path)

    if not os.path.exists(script_path):
        return jsonify({"ok": False, "error": f"Script not found: {script_path}"}), 400

    cfg_json = json.dumps(cfg)
    cmd = [python_exe, script_path, "--config-json", cfg_json]
    res = training_manager.start(cmd, cwd=workdir)
    return jsonify(res)


@bp_train.route("/api/stop", methods=["POST"])
def api_stop():
    res = training_manager.stop()
    return jsonify(res)


@bp_train.route("/api/status", methods=["GET"])
def api_status():
    return jsonify({"ok": True, **training_manager.status()})


@bp_train.route("/api/logs", methods=["GET"])
def api_logs():
    try:
        since = int(request.args.get("since", "0"))
    except ValueError:
        since = 0
    try:
        max_lines = int(request.args.get("max", "500"))
    except ValueError:
        max_lines = 500
    return jsonify({"ok": True, **training_manager.get_logs(since=since, max_lines=max_lines)})
