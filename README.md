# BYS_XAISwallow – Swallow Detection & Preprocessing

Real‑time Schluckakt (Swallow) Erkennung auf Video-/RAVI-Dateien unter Nutzung eines SlowFast-Modells (Window-basierte Klassifikation) und YOLO Pose für Kopf-/Nacken-ROI sowie optionale Gesichtsanonymisierung mittels Ellipsenmaske.

## Features
- Live Inferenz (MJPEG Stream + SocketIO Events für Meta & Predictions)
- ROI-Strategien: live YOLO Pose, first frame Pose, manuell fixierter Neck-Punkt
- Dynamische Face-Ellipse (live / statisch / off) zur Pseudonymisierung
- Preprocessing Pipeline: .ravi → erster Frame → Auto-Neck & Ellipse → Export (Full + ROI)
- Adaptive YOLO-Ausführung (nur wenn nötig)
- Modularer VideoPredictor (Thread) mit Seek & Replay

## Projektstruktur (Auszug)
```
BYS_XAISwallow/
    app.py                # Flask + SocketIO App Einstieg
    predictor.py          # VideoPredictor Thread / ROI Pipeline
    slowfast_core.py      # Minimal SlowFast Lade- & Preprocessing-Utilities
    vision_utils.py       # Ellipse, Masking, Neck-Schätzer, Crops
    yolo_pose_utils.py    # YOLO Pose Wrapper
    dataproc.py           # Preprocessing Blueprint (.ravi Export)
    training_bp.py        # Training-Blueprint (UI & REST APIs)
    training_service.py   # Subprozess-Manager für Training + Logs
    static/, templates/   # Frontend (UI, Charts, Controls)
    models/               # (leer, Gewichte via Download platzieren)
    uploads/, outputs/    # Laufzeitverzeichnisse (ignored)
train_multi_new.py        # Trainings-Runner (SlowRAFT++/SlowFast) mit Config-JSON
slowraft_improved.py      # SlowRAFT++ Modell-Bausteine
scripts/
    cleanup_large_files.py  # Hilfsskript (große Dateien identifizieren/auslagern)
```

## Installation
Voraussetzung: Python 3.10–3.12, optional CUDA für Torch.

Windows / PowerShell
- Python 3.10–3.12

```powershell
git clone <THIS_REPO>
cd <THIS_REPO>
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # aktivieren
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Environment konfigurieren (optional .env):
```powershell
copy .env.example .env   # passe SECRET_KEY & Pfade ggf. an
```

Model/Gewichte (Beispiel):
```
models/
    slowfastHub_ws32_e20.pth   # <- eigener Checkpoint (nicht im Repo)
    yolov8n-pose.pt            # YOLO Pose (Ultralytics) – falls nicht automatisch geladen
```

## Start
```powershell
# Variante A: Flask
python -m flask --app BYS_XAISwallow.app run

# Variante B: Direktstart
python .\BYS_XAISwallow\app.py
```
Öffne: http://localhost:5000

1. Modell laden (Pfad zum .pth + Parameter)
2. Video via Pfad oder Upload auswählen
3. Live‑Stream + Prediction Kurve beobachten / Seek nutzen

Preprocessing UI: http://localhost:5000/preprocess/
Training UI: http://localhost:5000/train/


## Konfiguration (ENV Variablen)
| Variable      | Beschreibung | Default |
|---------------|--------------|---------|
| SECRET_KEY    | Flask Session Secret | dev / .env |
| MODEL_DIR     | Ablage Modelle       | ./BYS_XAISwallow/models |
| UPLOAD_DIR    | Uploads              | ./BYS_XAISwallow/uploads |
| OUTPUT_DIR    | Exporte              | ./BYS_XAISwallow/outputs |
| DEVICE        | auto|cpu|cuda        | auto |
| LOG_LEVEL     | INFO/DEBUG/...       | INFO |

## API Endpunkte (Kurz)
- `POST /load_model` – JSON: { ckpt, model, tfast, alpha, ... , yolo_conf, roi_mode }
- `POST /upload` – multipart video
- `POST /use_path` – { path }
- `POST /stop`
- `GET  /video_feed` – MJPEG Stream
- `GET  /video_meta` – { fps, frames, duration }
- SocketIO: `control` (pause/play/seek/replay) + Events: `meta`, `pred`
- Preprocess Blueprint: `/preprocess/...`
 - Training Blueprint: `/train` (Seite), `/train/api/start|stop|status|logs`

## Entwicklung
Empfohlene Tools (optional hinzuzufügen):
```
ruff  # Lint + Format
black # Format (falls gewünscht)
pytest
mypy  # Typprüfung
```

Tests (Beispiel) kannst du unter `tests/` anlegen. Siehe `scripts/cleanup_large_files.py` für Repostrukturpflege.

## Datenschutz / Anonymisierung
Die Ellipse-Maske kann dynamisch aktiviert werden (Modus `live` oder `static`). Für strikte Anforderungen kann beim Start `FACE_MASK_MODE=live` erzwungen oder UI deaktiviert werden.

## Nächste Schritte / Roadmap
- CLI für Batch-Inferenz
- GitHub Actions CI (Lint + Unit Tests)
- Git LFS für große Gewichte & Beispielvideos
- Pre-Commit Hooks (ruff, black)

## Lizenz
Siehe `LICENSE`.

---
Bitte ergänze projektspezifische wissenschaftliche Hintergründe, Modellquellen und Zitationshinweise.

## Troubleshooting
- App startet nicht (Exit Code 1):
    - Starte aus dem Projektstamm: `cd <THIS_REPO>`; verwende `python .\BYS_XAISwallow\app.py`.
    - Prüfe, ob Torch/Ultralytics installiert sind: `pip show torch ultralytics`.
    - Fehlende Modelle: Lege deine `.pth` in `BYS_XAISwallow/models/` und passe den Pfad in der UI an.
    - Port blockiert: Beende alte Instanzen oder starte mit `FLASK_RUN_PORT=5001`.
    - Traceback posten – ich helfe bei der Behebung.
