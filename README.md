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



Model/Gewichte (Beispiel):
```
models/
    slowfast_ws32_e5.pth       # Standard-Checkpoint (1-Logit SlowFast, nicht im Repo)
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


## Training & Evaluation (Scripts)
Die wichtigsten Trainings-/Evaluationsskripte liegen unter `BYS_XAISwallow/code/`.

Hinweis zu Parametern: Die Skripte unterstützen in der Regel `-h/--help` und akzeptieren Standardargumente wie `--mode`, `--dataset_dir`, `--epochs`, `--batch_size`, `--window_size`, `--train_stride`, `--resize_h/--resize_w`, etc. Prüfe die Hilfe deiner Branch/Version für die exakten Namen.

- SlowFast (binary): `code/train_slowfast.py`
    - Zweck: Binäre Fensterklassifikation (1 Logit) mit PyTorchVideo SlowFast‑R50.
    - Train
        ```powershell
        python .\BYS_XAISwallow\code\train_slowfast.py --mode train --dataset_dir .\TSwallowDataset \
            --out_dir .\runs\slowfast --epochs 20
        ```
    - Test (alle Checkpoints in Ordner)
        ```powershell
        python .\BYS_XAISwallow\code\train_slowfast.py --mode test_all --ckpt_dir .\runs\slowfast\checkpoints
        ```
    - Test (einzelnes Video)
        ```powershell
        python .\BYS_XAISwallow\code\train_slowfast.py --mode test_single \
            --ckpt .\BYS_XAISwallow\models\slowfast_ws32_e5.pth \
            --video <pfad\zum\roi_video.mp4> --label <pfad\zum\elan.txt>
        ```

- SlowFast (multiclass): `code/train_slowfast_multiclass.py`
    - Zweck: Mehrklassen‑Training (z. B. `no_event`, `liquid`, `semi-solid`, `solid`).
    - Beispiel (Training)
        ```powershell
        python .\BYS_XAISwallow\code\train_slowfast_multiclass.py --mode train --dataset_dir .\TSwallowDataset \
            --out_dir .\runs\slowfast_mc --epochs 20
        ```
    - Hinweise: Klassenliste ist konfigurierbar; Fensterlabel via Mehrheitsregel mit Event‑Priorität.

- Optical Flow (baseline): `code/train_optflow.py`
    - Zweck: Leichtgewichtiges 3D‑CNN auf vorberechnetem Optical Flow (Mag+Angle).
    - Beispiel (Training mit vorberechnetem Flow)
        ```powershell
        python .\BYS_XAISwallow\code\train_optflow.py --mode train --dataset_dir .\TSwallowDataset \
            --out_dir .\runs\optflow --use_precomputed_flow 1
        ```
    - Hinweis: Siehe Abschnitt „Optical Flow Precompute“ unten zur Erzeugung der Flow‑Dateien.

- Dual Slow+Flow (leichtgewichtig): `code/train_dual_slowflow.py`
    - Zweck: Zwei Pfade – RGB‑Slow (wenige Frames) + Optical‑Flow‑Pfad; frühe/late Fusion.
    - Beispiel (Training)
        ```powershell
        python .\BYS_XAISwallow\code\train_dual_slowflow.py --mode train --dataset_dir .\TSwallowDataset \
            --out_dir .\runs\dual_slowflow
        ```

Allgemeine Datensatzannahme: Für jedes ROI‑Video (`*.mp4/*.avi/...`) liegt im gleichen Ordner eine `.txt`‑Annotation (ELAN‑Export) mit Zeitintervallen und Labels.


## Optical Flow Precompute (DIS)
Skript: `code/precompute_optflow_dis.py`

Features: DIS‑Flow, Ausgabe als HDF5 (`.h5`, quantisiert `int16` + `scale`) oder `.npz` (`float16`), Downscale, Fortschritt, Größenabschätzung.

- Einzeldatei (Video oder Frame‑Ordner) → HDF5
    ```powershell
    python .\BYS_XAISwallow\code\precompute_optflow_dis.py --input <video_oder_frameordner> \
        --output .\runs\optflow --backend h5 --dtype int16 --downscale 0.5
    ```
    Falls deine Branch keine CLI‑Flags unterstützt, öffne das Skript und passe `DEFAULT_CONFIG` direkt an.

- Batch über Dataset‑Ordner (Python‑API)
    ```python
    from BYS_XAISwallow.code.precompute_optflow_dis import batch_precompute_from_dataset
    batch_precompute_from_dataset("./TSwallowDataset", output_root="./runs/optflow")
    ```

HDF5‑Dateien enthalten `flow[T, H, W, 2]` (u,v) und ein Attribut `scale` für Dequantisierung.


## Grad‑CAM Analyse (SlowFast)
Skript: `code/analyze_gradcam_slowfast.py`

Zweck: Visualisiert Grad‑CAM für Slow‑ und Fast‑Pfad auf einem explizit gewählten ROI‑Video (mit zugehöriger `.txt`‑Annotation). Optional werden, falls vorhanden, vorberechnete Optical‑Flow‑Dateien (`.h5/.npz`) mit den CAMs kombiniert und Metriken/Heatmaps exportiert.

- Quickstart
    1) Im Skript `AnalyzeConfig` anpassen (Pfad zum ROI‑Video und Annotation).
    2) Starten:
         ```powershell
         python .\BYS_XAISwallow\code\analyze_gradcam_slowfast.py
         ```
    3) Alternativ programmgesteuert:
         ```python
         from BYS_XAISwallow.code.analyze_gradcam_slowfast import AnalyzeConfig, run_analysis
         run_analysis(AnalyzeConfig(video_path="./path/to/roi.mp4", annotation_path="./path/to/elan.txt"))
         ```

Ausgaben: CAM‑Overlays (PNG/MP4), per‑Frame Heatmaps, optionale Flow‑Heatmaps/-CSV (masked/raw), Zeitreihenplots.


## UI‑Integration (Kurz)
- Training‑UI nutzt standardmäßig SlowFast (Script: `code/train_slowfast.py`).
- Live‑Inferenz lädt ohne Angabe automatisch `models/slowfast_ws32_e5.pth` (falls vorhanden).
- Live‑Pipeline: statisches ROI aus dem ersten Frame; optionale Ellipsenmaske und Anzeige nur der ROI‑Ansicht.


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
