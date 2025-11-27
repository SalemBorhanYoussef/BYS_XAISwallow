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
        "script": os.path.join("code", "train_slowfast.py"),
        "python": "python",
        # Use package directory so relative script path resolves to BYS_XAISwallow/code/...
        "workdir": os.path.dirname(__file__),
    }
    return render_template("train.html", defaults=defaults)


@bp_train.route("/api/start", methods=["POST"])
def api_start():
    data = request.get_json(force=True, silent=True) or {}
    # Build command: python code/train_slowfast.py (SlowFast is new default)
    python_exe = data.get("python", "python")
    script_path = data.get("script", os.path.join("code", "train_slowfast.py"))
    # Default workdir to package directory to find BYS_XAISwallow/code
    workdir = data.get("workdir") or os.path.dirname(__file__)
    # train_slowfast.py currently reads its config from defaults in the file.
    # If a future JSON config is needed, extend the script to accept it.
    cfg = data.get("config", {})
    if not os.path.isabs(script_path):
        script_path = os.path.join(workdir, script_path)

    if not os.path.exists(script_path):
        return jsonify({"ok": False, "error": f"Script not found: {script_path}"}), 400

    # Map UI config fields to CLI flags for train_slowfast.py
    mode = (cfg.get("mode") or "train")
    cmd = [python_exe, script_path, "--mode", mode]
    # required dataset_dir
    if cfg.get("dataset_dir"):
        cmd += ["--dataset_dir", cfg["dataset_dir"]]
    # optional overrides
    for key in [
        "dataset_val_dir","dataset_test_dir","out_dir","epochs","batch_size",
        "lr","resize_h","resize_w","window_size","t_fast","alpha","device",
        "epoch","epoch_max","model_path","video_path","label_path"
    ]:
        if cfg.get(key) is not None:
            cmd += [f"--{key}", str(cfg[key])]
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
