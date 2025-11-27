# ============================================================
# train_slowfast_multiclass.py
# SlowFast (PyTorchVideo) – Multi-Class Classification
# Klassen: no_event, liquid, semi-solid, solid  (konfigurierbar)
#
# Anforderungen:
#   pip install torch torchvision pytorchvideo tensorboard opencv-python scikit-learn pandas
# ============================================================

import os
import re
import time
import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from glob import glob
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from collections import deque, Counter

# ============================================================
# Logging
# ============================================================
logger = logging.getLogger("train_slowfast_mc")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

# ============================================================
# Config
# ============================================================
@dataclass
class Config:
    # --- Daten
    dataset_dir: str = "D:/Master_Arbeit/TSwallowDataset"
    dataset_val_dir: Optional[str] = None
    dataset_test_dir: Optional[str] = None
    val_split_frac: float = 0.15

    # --- Video / Sampling
    fps: int = 32
    resize_h: int = 224
    resize_w: int = 224
    window_size: int = 32
    train_stride: int = 4
    val_stride: int = 8
    test_stride: int = 8

    # --- SlowFast Zeitsampling
    t_fast: int = 32
    alpha: int = 4  # t_slow = t_fast // alpha

    # --- Optimierung
    batch_size: int = 8
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 0.0

    # scheduler: "none" | "cosine" | "plateau_val"
    scheduler: str = "plateau_val"
    plateau_metric: str = "val_f1_macro"  # "val_loss" (min) | "val_f1_macro" (max)
    plateau_patience: int = 2
    plateau_factor: float = 0.5
    plateau_min_lr: float = 1e-7

    early_stopping_patience: int = 7

    # --- System
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 8
    pin_memory: bool = torch.cuda.is_available()
    use_amp: bool = False

    # --- Checkpoints
    out_dir: str = "./models/"
    ckpt: Optional[str] = None

    # --- Klassen (in Reihenfolge der Logits)
    # Wichtig: 'no_event' muss enthalten sein (Default-ID = 0)
    class_list: Tuple[str, ...] = ("no_event", "liquid", "semi-solid", "solid")

    # --- Vorverarbeitung
    normalize: bool = True
    mean: Tuple[float, float, float] = (0.45, 0.45, 0.45)
    std: Tuple[float, float, float] = (0.225, 0.225, 0.225)

    # --- Tricks
    epoch_jitter: bool = True

    # --- TensorBoard
    log_dir_base: str = "runs/slowfast_mc"
    log_every_n_steps: int = 20
    log_weight_hist_every_n_epochs: int = 5
    print_every_10: int = 10
    print_every_100: int = 100

    # --- Freeze / Finetune Policy
    freeze_policy: str = "none"  # "linear" | "partial" | "none"
    pretrained_backbone: bool = True
    bn_eval_freeze_affine: bool = False

    backbone_lr: float = 5e-6
    head_lr: float = 1e-4

    # --- Auswahl für bestes Modell
    best_metric: str = "val_f1_macro"

    # --- Modelltyp (für Logs, Checkpoints etc.)
    model_type: str = "slowfast_mc"

    # --- Event Labeling / Sampling Feintuning
    min_fraction_event: float = 0.20  # Schwelle fuer Fenster-Event-Erkennung (z.B. 0.20 für strengere Event-Fenster)
    pos_step: int = 1                 # Schrittweite fuer positive (Event) Fenster
    neg_step: int = 8                 # Schrittweite fuer negative (no_event) Fenster

    # --- Training Dynamics
    head_warmup_epochs: int = 2       # Anzahl Epochen nur Head (0 = aus)
    grad_clip_max_norm: float = 5.0   # Gradient Clipping (<=0 deaktiviert)
    focal_start_epoch: Optional[int] = 2  # Frueherer Einsatz der FocalLoss

    # --- Manuelle Klassen-Gewichte (Optional): Variante A -> alle 1.0 (Gewichtung nur via Oversampling)
    manual_class_weights: Optional[Tuple[float, ...]] = (1.0, 1.0, 1.0, 1.0)

    # --- Inferenz-Entscheidung
    event_thresh: float = 0.5           # Schwellwert für Event-Priorisierung
    no_event_dampen: float = 0.3        # no_event-Logit-Abwertung (0.2 konservativ, 0.4 aggressiver)

    # --- Datenaugmentation (nur Training)
    augment_train: bool = True
    aug_hflip_p: float = 0.5
    aug_brightness: float = 0.1
    aug_contrast: float = 0.1
    aug_crop_scale: Tuple[float, float] = (0.9, 1.0)  # RandomResizedCrop scale (<=1.0)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# ============================================================
# Utility
# ============================================================
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def list_video_label_pairs(ds_dir: str) -> List[Tuple[str, str]]:
    """Finde (video, label)-Paare rekursiv. Erwartet .txt Labels in demselben Ordner."""
    pairs: List[Tuple[str, str]] = []
    if not os.path.isdir(ds_dir):
        logger.warning("Verzeichnis existiert nicht: %s", ds_dir)
        return pairs

    exts = ("mp4", "avi", "mov", "mkv")
    matches = []
    for ext in exts:
        pattern = os.path.join(ds_dir, "**", f"*roi*.{ext}")
        matches.extend(sorted(glob(pattern, recursive=True)))

    logger.info("%d ROI-Videos gefunden.", len(matches))
    for v in sorted(set(matches)):
        dirpath = os.path.dirname(v)
        fname = os.path.splitext(os.path.basename(v))[0]
        candidates = [
            os.path.join(dirpath, "annotation_" + fname.replace("_roi", "") + ".txt"),
            os.path.join(dirpath, fname.replace("_roi", "") + ".txt"),
            os.path.join(dirpath, fname + ".txt"),
            os.path.join(dirpath, fname.replace("roi_", "annotation_") + ".txt"),
        ]
        m = re.search(r"(\d+)", fname)
        if m:
            num = m.group(1)
            for t in sorted(glob(os.path.join(dirpath, f"*{num}*.txt"))):
                candidates.append(t)
        found = None
        for c in candidates:
            if os.path.exists(c):
                found = c
                break
        if not found:
            ann_txts = sorted(glob(os.path.join(dirpath, "*annotation*.txt")))
            if ann_txts:
                found = ann_txts[0]
        if found:
            pairs.append((os.path.abspath(v), os.path.abspath(found)))
        else:
            logger.warning("Keine Label-Datei gefunden für: %s", v)
    logger.info("Insgesamt %d (video, txt)-Paare.", len(pairs))
    return pairs


def build_tfm_cpu(normalize: bool, size_hw: Tuple[int, int], mean, std, augment: bool = False, cfg: Optional[Config] = None):
    from torchvision import transforms as T
    h, w = size_hw
    tfms = [T.ToPILImage()]
    if augment and cfg is not None:
        # Hinweis: RandomResizedCrop scale muss <= 1.0 sein
        tfms.append(T.RandomResizedCrop((h, w), scale=cfg.aug_crop_scale))
        if cfg.aug_hflip_p and cfg.aug_hflip_p > 0:
            tfms.append(T.RandomHorizontalFlip(p=float(cfg.aug_hflip_p)))
        if (cfg.aug_brightness and cfg.aug_brightness > 0) or (cfg.aug_contrast and cfg.aug_contrast > 0):
            tfms.append(T.ColorJitter(brightness=float(cfg.aug_brightness), contrast=float(cfg.aug_contrast)))
    else:
        tfms.append(T.Resize((h, w), antialias=True))
    tfms.append(T.ToTensor())
    if normalize:
        tfms.append(T.Normalize(mean=mean, std=std))
    return T.Compose(tfms)


def sample_indices(num_src: int, num_out: int) -> List[int]:
    if num_src <= 0:
        return [0] * num_out
    if num_out <= 1:
        return [0]
    if num_src == num_out:
        return list(range(num_src))
    idx = np.linspace(0, max(0, num_src - 1), num_out)
    return np.round(idx).astype(int).tolist()


def make_slowfast_tensors_cpu(frames_rgb, t_fast: int, t_slow: int, tfm):
    """
    Returns tensors with shape:
    - slow: C x T_slow x H x W
    - fast: C x T_fast x H x W
    """
    idx_fast = sample_indices(len(frames_rgb), t_fast)
    fast_clip = [tfm(frames_rgb[i]) for i in idx_fast]
    fast = torch.stack(fast_clip, dim=1)
    idx_slow = sample_indices(len(fast_clip), t_slow)
    slow_clip = [fast_clip[i] for i in idx_slow]
    slow = torch.stack(slow_clip, dim=1)
    return slow, fast


# ============================================================
# Annotation → Frame-Klassen
# ============================================================
def build_frame_class_ids(
    txt_path: str,
    total_frames: int,
    fps: int,
    class_to_id: Dict[str, int],
    default_class_id: int = 0,
) -> np.ndarray:
    """
    Baut für jedes Frame eine Klassen-ID (int). Standard: no_event (0).
    Erwartet Tab-getrennte Zeilen mit:
      ...  start(sec)  ...  end(sec)  ...  label
    (Start in Spalte 4, Ende in Spalte 6, Label in Spalte 9 – wie in deinem Code)
    """
    arr = np.full(total_frames, default_class_id, dtype=np.int64)
    if not os.path.exists(txt_path):
        logger.warning("Label file not found: %s", txt_path)
        return arr

    unknown = Counter()
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            # Start/Ende
            try:
                s = float(parts[3].replace(",", "."))
                e = float(parts[5].replace(",", "."))
            except ValueError:
                continue
            raw_label = parts[8].strip().lower()

            # robuste Label-Normalisierung: vereinheitliche Trennzeichen und gängige Aliase
            norm = raw_label.replace("_", "-").replace(" ", "-")
            alias = {
                "semisolid": "semi-solid",
                "semi-solid": "semi-solid",
                "semi--solid": "semi-solid",
                "solid-food": "solid",
                "liquids": "liquid",
                "noevent": "no_event",
            }
            norm = alias.get(norm, norm)
            if norm not in class_to_id:
                unknown[norm] += 1
                continue

            si = max(0, int(round(s * fps)))
            ei = min(total_frames - 1, int(round(e * fps)))
            if ei >= si:
                arr[si : ei + 1] = class_to_id[norm]
    if unknown:
        logger.warning("Unbekannte Labels in %s: %s", os.path.basename(txt_path), dict(unknown))
    return arr


def window_majority_class(
    frame_ids: np.ndarray,
    event_prioritize: bool = True,
    min_fraction: float = 0.2,
    no_event_id: int = 0,
    center_override: bool = True,
) -> int:
    """
    Wählt eine Fenster-Klasse:
    1) Schwellenbasiert: sobald eine Klasse >= min_fraction der Frames belegt, nimm diese
       (priorisiere Events != no_event_id bei Kollisionen).
    2) Fallback: klassische Mehrheit; bei Gleichstand optional Events bevorzugen.
    """
    if frame_ids.size == 0:
        return int(no_event_id)
    if center_override:
        c = frame_ids[len(frame_ids) // 2]
        if c != no_event_id:
            return int(c)
    cnt = Counter(frame_ids.tolist())
    total = max(1, frame_ids.size)
    # 1) Kandidaten nach Schwelle
    candidates = [(cls, n) for cls, n in cnt.items() if (n / total) >= float(min_fraction)]
    if candidates:
        # Events bevorzugen (cls != no_event_id), dann nach Häufigkeit
        candidates.sort(key=lambda x: ((x[0] == no_event_id) if event_prioritize else False, -x[1]))
        return int(candidates[0][0])
    # 2) Mehrheit mit optionaler Event-Priorität
    best_cls, best_n = no_event_id, -1
    for cls_id, n in cnt.items():
        prefer_event = (best_cls == no_event_id and cls_id != no_event_id) if event_prioritize else False
        if (n > best_n) or (n == best_n and prefer_event):
            best_cls, best_n = cls_id, n
    return int(best_cls)


# ============================================================
# Dataset
# ============================================================
class SlowFastDatasetMC(Dataset):
    """Fenster-basierte Multi-Class Labels (Mehrheitsklasse je Fenster)."""

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        window_size: int,
        stride: int,
        fps: int,
        t_fast: int,
        alpha: int,
        resize_hw: Tuple[int, int],
        normalize: bool,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        class_list: Tuple[str, ...],
        epoch_jitter: bool = True,
        augment: bool = False,
        cfg: Optional[Config] = None,
    ):
        self.window = int(window_size)
        self.stride = max(1, int(stride))
        self.fps = int(fps)
        self.t_fast = int(t_fast)
        self.t_slow = max(1, self.t_fast // int(alpha))
        self.resize_hw = (int(resize_hw[0]), int(resize_hw[1]))
        self.normalize = bool(normalize)
        self.mean, self.std = mean, std
        self.enable_epoch_jitter = bool(epoch_jitter)
        self.index: List[Tuple[str, int, np.ndarray, int]] = []  # (vpath, start, frame_cls_ids, total_frames)
        self.epoch: int = 0
        self.class_list = tuple(s.lower() for s in class_list)
        self.class_to_id = {c: i for i, c in enumerate(self.class_list)}
        self.num_classes = len(self.class_list)
        # Konfigurierbare Label- und Sampling-Parameter
        self._min_frac = (cfg.min_fraction_event if cfg else 0.15)
        self._pos_step = (cfg.pos_step if cfg else 1)
        self._neg_step = (cfg.neg_step if cfg else max(1, self.stride * 2))

        # Build frame-wise class ids per video, index windows
        for vpath, tpath in pairs:
            cap = cv2.VideoCapture(vpath)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            cap.release()
            if n <= 0:
                logger.warning("Empty video: %s", vpath)
                continue
            frame_ids = build_frame_class_ids(
                tpath, n, self.fps, class_to_id=self.class_to_id, default_class_id=0
            )
            max_start = max(1, n - self.window + 1)
            # adaptiver Stride je nach Fensterklasse (approx über zentriertes Fenster)
            s = 0
            while s < max_start:
                center = min(max(0, s + (self.window // 2)), n - 1)
                start = max(0, center - (self.window // 2))
                end = min(n, start + self.window)
                label_center = window_majority_class(
                    frame_ids[start:end], min_fraction=self._min_frac, no_event_id=0
                )
                # Oversample sparse-event windows: if any event present, step densely
                has_any_event = bool(np.any(frame_ids[start:end] != 0))
                if label_center == 0:
                    step = self._pos_step if has_any_event else self._neg_step
                else:
                    step = self._pos_step
                self.index.append((vpath, s, frame_ids, n))
                s += step

        self.tfm_cpu = build_tfm_cpu(self.normalize, self.resize_hw, self.mean, self.std, augment=augment, cfg=cfg)

        # Precompute window labels (epoch-static baseline)
        self.window_labels = np.array(
            [
                window_majority_class(
                    fid[s : s + self.window], min_fraction=self._min_frac, no_event_id=0
                )
                for (_, s, fid, _) in self.index
            ],
            dtype=np.int64,
        )

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _epoch_jitter(self, idx: int) -> int:
        if not self.enable_epoch_jitter or self.stride <= 1:
            return 0
        # deterministisch per idx/epoch
        return ((idx * 1315423911) ^ (self.epoch * 2654435761)) % self.stride

    def epoch_window_labels(self) -> np.ndarray:
        """Optional: pro Epoche leicht verschobene Fensterlabels (Jitter)."""
        labs = np.zeros(len(self.index), dtype=np.int64)
        for i, (vpath, start, fid, n) in enumerate(self.index):
            j = self._epoch_jitter(i)
            s = min(start + j, max(0, n - self.window))
            labs[i] = window_majority_class(
                fid[s : s + self.window], min_fraction=self._min_frac, no_event_id=0
            )
        return labs

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        vpath, start, fid, n = self.index[idx]
        j = self._epoch_jitter(idx)
        s = min(start + j, max(0, n - self.window))

        center = min(max(0, s + (self.window // 2)), n - 1)
        half = self.window // 2
        read_start = center - half
        if read_start < 0:
            read_start = 0
        if read_start + self.window > n:
            read_start = max(0, n - self.window)

        cap = cv2.VideoCapture(vpath)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, read_start)
        except Exception:
            pass

        frames = []
        last = None
        for _ in range(self.window):
            ok, frame = cap.read()
            if not ok or frame is None:
                if last is not None:
                    frames.append(last)
                else:
                    frames.append(np.zeros((self.resize_hw[0], self.resize_hw[1], 3), np.uint8))
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last = frame
            frames.append(frame)
        cap.release()

        slow, fast = make_slowfast_tensors_cpu(frames, self.t_fast, self.t_slow, self.tfm_cpu)
        label = window_majority_class(
            fid[read_start : read_start + self.window], min_fraction=self._min_frac, no_event_id=0
        )
        return [slow, fast], torch.tensor(label, dtype=torch.long)  # CE erwartet long


# ============================================================
# Modell
# ============================================================
def build_model_slowfast_mc(device: torch.device, num_classes: int, pretrained: bool = True) -> nn.Module:
    try:
        from pytorchvideo.models.hub import slowfast_r50
    except Exception as e:
        raise RuntimeError("Bitte 'pytorchvideo' installieren: pip install pytorchvideo") from e
    m = slowfast_r50(pretrained=pretrained)
    # Ersetze Head auf num_classes
    m.blocks[-1].proj = nn.Linear(m.blocks[-1].proj.in_features, num_classes)
    return m.to(device)


def apply_freeze_policy_slowfast(model: nn.Module, policy: str, bn_freeze_affine: bool = True):
    assert hasattr(model, "blocks"), "Unerwartete SlowFast-Struktur"
    backbone_modules = [model.blocks[i] for i in range(0, 5)]  # 0..4
    pool_module = model.blocks[5]
    head_module = model.blocks[6]

    def _set_requires_grad(module: nn.Module, flag: bool):
        for p in module.parameters():
            p.requires_grad = flag

    def _bn_eval_and_freeze_affine(module: nn.Module, freeze_affine: bool = True):
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()
            if freeze_affine and getattr(module, "affine", False):
                if module.weight is not None:
                    module.weight.requires_grad = False
                if module.bias is not None:
                    module.bias.requires_grad = False
        for child in module.children():
            _bn_eval_and_freeze_affine(child, freeze_affine)

    if policy == "linear":
        for m in backbone_modules + [pool_module]:
            _set_requires_grad(m, False)
        _set_requires_grad(head_module, True)
    elif policy == "partial":
        for i in range(0, 4):
            _set_requires_grad(model.blocks[i], False)
        _set_requires_grad(pool_module, False)
        _set_requires_grad(model.blocks[4], True)
        _set_requires_grad(head_module, True)
    elif policy == "none":
        _set_requires_grad(model, True)
    else:
        raise ValueError(f"Unbekannte freeze_policy: {policy}")

    if bn_freeze_affine:
        _bn_eval_and_freeze_affine(model, freeze_affine=True)


# ============================================================
# DataLoader-Helfer
# ============================================================
def _build_weighted_loader(ds: SlowFastDatasetMC, cfg: Config, cur_labels: np.ndarray, pow_gamma: float = 2.0) -> DataLoader:
    """
    Minimaler & sicherer Oversampling-Ansatz mit WeightedRandomSampler:
    - Zähle Häufigkeit je Klasse in cur_labels
    - Gewicht pro Sample = max_count / count[label]
    - replacement=True, num_samples=len(dataset)
    Hinweis: pow_gamma ist hier nicht mehr relevant und wird ignoriert.
    """
    from collections import Counter
    labels_list = cur_labels.tolist() if isinstance(cur_labels, np.ndarray) else list(cur_labels)
    cls_counts = Counter(labels_list)
    if len(cls_counts) == 0:
        # Fallback: uniform
        weights = [1.0 for _ in labels_list]
    else:
        max_count = max(cls_counts.values())
        weights = [float(max_count) / float(cls_counts[int(lbl)]) for lbl in labels_list]
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(labels_list), replacement=True)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.num_workers > 0),
    )


def _split_train_val_pairs(train_pairs: List[Tuple[str,str]], frac: float, seed: int):
    if not 0.0 < frac < 1.0:
        return train_pairs, []
    rng = np.random.RandomState(seed)
    idx = np.arange(len(train_pairs))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(train_pairs) * frac)))
    val_idx = set(idx[:n_val].tolist())
    train_sub = [p for i, p in enumerate(train_pairs) if i not in val_idx]
    val_sub   = [p for i, p in enumerate(train_pairs) if i in val_idx]
    return train_sub, val_sub


def make_dataloaders(cfg: Config, device: torch.device):
    all_pairs = list_video_label_pairs(cfg.dataset_dir)

    # ---- Behalte deine gewünschte Filterlogik bei (falls nötig anpassen) ----
    all_pairs = [x for x in all_pairs if "tasks" in x[0]]

    if not all_pairs:
        raise RuntimeError("Keine (video, txt)-Paare im Trainingsordner gefunden.")

    rng = np.random.RandomState(cfg.seed)
    idx = np.arange(len(all_pairs))
    rng.shuffle(idx)

    if cfg.dataset_val_dir and cfg.dataset_test_dir:
        val_pairs = list_video_label_pairs(cfg.dataset_val_dir)
        val_pairs = [x for x in val_pairs if ("tasks" in x[0]) and ("dis" not in x[0]) and ("raft" not in x[0])]
        test_pairs = list_video_label_pairs(cfg.dataset_test_dir)
        test_pairs = [x for x in test_pairs if ("tasks" in x[0]) and ("dis" not in x[0]) and ("raft" not in x[0])]
        train_pairs = [x for x in all_pairs if ("tasks" in x[0]) and ("dis" not in x[0]) and ("raft" not in x[0])]
    elif cfg.dataset_test_dir:
        test_pairs = list_video_label_pairs(cfg.dataset_test_dir)
        train_pairs, val_pairs = _split_train_val_pairs(all_pairs, cfg.val_split_frac, cfg.seed)
    else:
        # 70/15/15 Split
        n = len(all_pairs)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]
        train_pairs = [all_pairs[i] for i in train_idx]
        val_pairs = [all_pairs[i] for i in val_idx]
        test_pairs = [all_pairs[i] for i in test_idx]
        logger.info("⚙️ Keine separaten Val/Test-Verzeichnisse angegeben → 70/15/15 Split angewendet.")

    logger.info("Found train videos=%d | val videos=%d | test videos=%d",
                len(train_pairs), len(val_pairs), len(test_pairs))

    # --- Datasets ---
    train_ds = SlowFastDatasetMC(
        train_pairs, cfg.window_size, cfg.train_stride, cfg.fps,
        cfg.t_fast, cfg.alpha,
        (cfg.resize_h, cfg.resize_w), cfg.normalize, cfg.mean, cfg.std,
        class_list=cfg.class_list, epoch_jitter=cfg.epoch_jitter, augment=cfg.augment_train, cfg=cfg,
    )
    val_ds = SlowFastDatasetMC(
        val_pairs, cfg.window_size, cfg.val_stride, cfg.fps,
        cfg.t_fast, cfg.alpha,
        (cfg.resize_h, cfg.resize_w), cfg.normalize, cfg.mean, cfg.std,
        class_list=cfg.class_list, epoch_jitter=False, augment=False, cfg=cfg,
    )
    test_ds = SlowFastDatasetMC(
        test_pairs, cfg.window_size, cfg.test_stride, cfg.fps,
        cfg.t_fast, cfg.alpha,
        (cfg.resize_h, cfg.resize_w), cfg.normalize, cfg.mean, cfg.std,
        class_list=cfg.class_list, epoch_jitter=False, augment=False, cfg=cfg,
    )

    # --- Dataloader ---
    train_ds.set_epoch(0)
    cur_labels = train_ds.epoch_window_labels() if cfg.epoch_jitter else train_ds.window_labels

    # Klassenverteilung & Anteil Events (nach adaptivem Stride)
    binc = np.bincount(cur_labels, minlength=len(cfg.class_list))
    total = int(binc.sum()) if binc.size > 0 else 0
    if total > 0:
        msg = ", ".join([
            f"{cfg.class_list[i]}: {int(binc[i])} ({(binc[i]/total):.1%})" for i in range(len(cfg.class_list))
        ])
        logger.info("Train windows per class (after adaptive stride): %s", msg)

    train_loader = _build_weighted_loader(train_ds, cfg, cur_labels, pow_gamma=2.0)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.num_workers > 0),
    )

    # Log (grobe) Klassenverteilung pro Klasse (absolut)
    num_classes = len(cfg.class_list)
    binc = np.bincount(cur_labels, minlength=num_classes)
    for i, c in enumerate(cfg.class_list):
        logger.info("Train windows class '%s' : %d", c, int(binc[i]))

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds


# ============================================================
# Evaluation
# ============================================================
def event_priority_pred(probs, event_thresh: float = 0.5, no_event_id: int = 0):
    """Event-first decision: predict no_event unless best event prob clears threshold."""
    import numpy as np
    probs = np.asarray(probs)
    C = probs.shape[1]
    ev_mask = np.ones(C, dtype=bool)
    ev_mask[no_event_id] = False
    ev_max = probs[:, ev_mask].max(axis=1)
    ev_arg = probs[:, ev_mask].argmax(axis=1)
    event_class_ids = np.where(ev_mask)[0]
    ev_pred_ids = event_class_ids[ev_arg]
    pred = np.where(ev_max >= event_thresh, ev_pred_ids, no_event_id)
    return pred

@torch.no_grad()
def evaluate_multiclass(
    loader: DataLoader,
    model: nn.Module,
    criterion,
    device: torch.device,
    num_classes: int,
    smooth_window: int = 1,
    use_amp: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
    event_thresh: float = 0.5,
    no_event_dampen: float = 0.3,
):
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
    )

    if device.type == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    model.eval()
    total_loss, total_n = 0.0, 0
    all_logits = []
    all_targets = []

    for (slow, fast), y in loader:
        slow, fast = slow.to(device, non_blocking=True), fast.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)  # long
        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda", enabled=True, dtype=(amp_dtype or torch.float16)):
                logits = model([slow, fast])  # (B, C)
                loss = criterion(logits, y)
        else:
            logits = model([slow, fast])
            loss = criterion(logits, y)

        total_loss += float(loss.item()) * y.size(0)
        total_n += y.size(0)
        # Inference-time bias correction: dampen no_event logit slightly
        logits_eval = logits.detach().clone()
        try:
            logits_eval[:, 0] -= float(no_event_dampen)
        except Exception:
            pass
        all_logits.append(logits_eval.to(torch.float32).cpu().numpy())
        all_targets.append(y.detach().cpu().numpy())

    if not all_logits:
        return {
            "loss": 0.0, "acc": 0.0,
            "f1_macro": 0.0, "f1_micro": 0.0, "f1_weighted": 0.0,
            "per_class": None, "cm": np.zeros((num_classes, num_classes), dtype=int)
        }

    logits = np.concatenate(all_logits, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    preds = event_priority_pred(probs, event_thresh=event_thresh, no_event_id=0)

    acc = accuracy_score(targets, preds)
    f1_macro = f1_score(targets, preds, average="macro", zero_division=0)
    f1_micro = f1_score(targets, preds, average="micro", zero_division=0)
    f1_weighted = f1_score(targets, preds, average="weighted", zero_division=0)
    prec, rec, f1, _ = precision_recall_fscore_support(targets, preds, labels=list(range(num_classes)), zero_division=0)
    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))

    per_class = {
        "precision": prec.tolist(),
        "recall": rec.tolist(),
        "f1": f1.tolist(),
    }

    return {
        "loss": total_loss / max(1, total_n),
        "acc": float(acc),
        "f1_macro": float(f1_macro),
        "f1_micro": float(f1_micro),
        "f1_weighted": float(f1_weighted),
        "per_class": per_class,
        "cm": cm,
        "y_true": targets,
        "y_pred": preds,
    }


def _add_confusion_matrix_image(writer: SummaryWriter, tag: str, cm: np.ndarray, class_names: List[str], global_step: int = 0):
    import matplotlib.pyplot as plt
    import io
    fig = plt.figure(figsize=(0.9 + 0.5*len(class_names), 0.9 + 0.5*len(class_names)))
    ax = plt.gca()
    ax.imshow(cm, interpolation='nearest')
    ax.set_title("Confusion Matrix")
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=140)
    plt.close(fig)
    buf.seek(0)
    import PIL.Image as Image
    img = Image.open(buf)
    img = np.array(img).transpose(2,0,1)  # CHW
    writer.add_image(tag, img, global_step)


# ============================================================
# Training
# ============================================================
def build_param_groups(model: nn.Module, head_modules: List[nn.Module], backbone_lr: float, head_lr: float, weight_decay: float):
    head_param_ids = set()
    head_params = []
    for hm in head_modules:
        for p in hm.parameters():
            if p.requires_grad:
                head_params.append(p)
                head_param_ids.add(id(p))
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_param_ids]
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay, "name": "backbone"})
    if head_params:
        groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay, "name": "head"})
    return groups


def _make_run_dirs_and_writer(cfg: Config):
    ts = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(cfg.log_dir_base, f"{cfg.model_type}_{ts}")
    os.makedirs(log_dir, exist_ok=True)
    meta = {"timestamp": ts, "model_type": cfg.model_type, "cfg": asdict(cfg)}
    with open(os.path.join(log_dir, "RUN_META.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text("config/json", f"```\n{cfg.to_json()}\n```", 0)
    return writer, log_dir


def compute_class_weights_from_labels(labels: np.ndarray, num_classes: int, pow_gamma: float = 1.5) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    inv = (1.0 / counts) ** pow_gamma
    weights = inv / inv.sum() * num_classes  # skaliere grob
    return torch.tensor(weights, dtype=torch.float32)


def get_effective_class_weights(cfg: Config, num_classes: int, device: torch.device, cur_labels: Optional[np.ndarray] = None) -> torch.Tensor:
    """Return class weights, preferring manual weights from config if valid, otherwise computed from labels.
    If labels are missing and no manual weights, defaults to uniform weights.
    """
    if cfg.manual_class_weights is not None:
        try:
            w = list(cfg.manual_class_weights)
            if len(w) == num_classes:
                return torch.tensor(w, dtype=torch.float32, device=device)
            else:
                logger.warning("manual_class_weights length (%d) != num_classes (%d) -> fallback to computed/auto.", len(w), num_classes)
        except Exception:
            logger.warning("Invalid manual_class_weights in config, fallback to computed/auto.")
    if cur_labels is not None:
        return compute_class_weights_from_labels(cur_labels, num_classes, pow_gamma=1.5).to(device)
    return torch.ones(num_classes, dtype=torch.float32, device=device)


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha  # shape [C]
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: (B,C), target: (B,)
        logp = torch.log_softmax(logits, dim=1)        # (B,C)
        p = torch.softmax(logits, dim=1)               # (B,C)
        idx = torch.arange(target.size(0), device=logits.device)
        pt = p[idx, target]                            # (B,)
        logpt = logp[idx, target]
        at = self.alpha[target]                        # (B,)
        loss = - at * ((1 - pt).clamp(min=1e-6) ** self.gamma) * logpt
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def train(cfg: Config):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    if device.type == "cuda":
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
    os.makedirs(cfg.out_dir, exist_ok=True)

    writer, log_dir = _make_run_dirs_and_writer(cfg)

    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = make_dataloaders(cfg, device)

    num_classes = len(cfg.class_list)
    model = build_model_slowfast_mc(device, num_classes=num_classes, pretrained=cfg.pretrained_backbone)

    if cfg.ckpt and os.path.isfile(cfg.ckpt):
        state = torch.load(cfg.ckpt, map_location=device)
        model.load_state_dict(state, strict=False)
        logger.info("Loaded weights from %s", cfg.ckpt)

    apply_freeze_policy_slowfast(model, cfg.freeze_policy, bn_freeze_affine=cfg.bn_eval_freeze_affine)
    head_modules = [model.blocks[-1].proj]

    # Loss mit Klassen-Gewichten (aus Trainingsfenstern der aktuellen Epoche)
    cur_labels = train_ds.epoch_window_labels() if cfg.epoch_jitter else train_ds.window_labels
    class_weights = get_effective_class_weights(cfg, num_classes, device, cur_labels)
    use_focal = False  # initial ruhigere Optimierung mit CE
    focal_start_epoch = cfg.focal_start_epoch if (cfg.focal_start_epoch is not None and cfg.focal_start_epoch >= 0) else None
    if use_focal:
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer mit LR-Splitting
    param_groups = build_param_groups(
        model,
        head_modules=head_modules,
        backbone_lr=cfg.backbone_lr,
        head_lr=cfg.head_lr,
        weight_decay=cfg.weight_decay,
    )
    if not param_groups:
        raise RuntimeError("Keine trainierbaren Parameter – prüfe freeze_policy.")
    optimizer = optim.Adam(param_groups)

    # Scheduler
    if cfg.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.1)
    elif cfg.scheduler == "plateau_val":
        mode = "min" if cfg.plateau_metric == "val_loss" else "max"
        try:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=mode, factor=cfg.plateau_factor, patience=cfg.plateau_patience,
                min_lr=cfg.plateau_min_lr, eps=1e-8, verbose=True
            )
        except TypeError:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=mode, factor=cfg.plateau_factor, patience=cfg.plateau_patience,
                min_lr=cfg.plateau_min_lr, eps=1e-8
            )
    else:
        scheduler = None

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp and (device.type == "cuda"))
    except Exception:
        class _DummyScaler:
            def is_enabled(self): return False
            def scale(self, x): return x
            def step(self, opt): opt.step()
            def update(self): pass
        scaler = _DummyScaler()

    # Head-Bias init auf Klassen-Priors
    def init_head_bias_uniform(model: nn.Module, num_classes: int):
        with torch.no_grad():
            head = model.blocks[-1].proj
            if head.bias is None:
                head.bias = nn.Parameter(torch.zeros(num_classes, device=next(model.parameters()).device))
            else:
                head.bias.zero_()

    init_head_bias_uniform(model, num_classes)

    # Optional Head-Warmup: falls gesetzt, Backbone temporär einfrieren
    if cfg.head_warmup_epochs > 0:
        apply_freeze_policy_slowfast(model, "linear", bn_freeze_affine=cfg.bn_eval_freeze_affine)

    global_step = 0
    best_metric_value = -1.0
    best_path = None
    es_counter = 0

    def _global_grad_norm(parameters, norm_type: float = 2.0) -> float:
        params = [p for p in parameters if p.grad is not None]
        if not params:
            return 0.0
        device_l = params[0].grad.device
        total = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device_l) for p in params]), norm_type)
        return total.item()

    for epoch in range(1, cfg.epochs + 1):
        # AMP Robustness decision
        just_unfroze = (cfg.head_warmup_epochs > 0 and epoch <= (cfg.head_warmup_epochs + 1))
        use_amp_this_epoch = cfg.use_amp and not just_unfroze and (device.type == "cuda")
        amp_dtype = torch.bfloat16 if use_amp_this_epoch and torch.cuda.is_bf16_supported() else torch.float16

        # Temporary LR scaling in the first epoch after unfreeze (and warmup epoch boundary)
        for g in optimizer.param_groups:
            base = cfg.head_lr if g.get("name") == "head" else cfg.backbone_lr
            scale = 0.5 if just_unfroze else 1.0
            g["lr"] = base * scale
        # epoch jitter + sampler + klassen-gewichte refresh
        if cfg.epoch_jitter:
            train_ds.set_epoch(epoch)
            cur_labels = train_ds.epoch_window_labels()
        else:
            cur_labels = train_ds.window_labels

        # Switch auf FocalLoss ab gewünschter Epoche
        if focal_start_epoch is not None and epoch >= focal_start_epoch:
            use_focal = True

        # Update WeightedRandomSampler
        train_loader = _build_weighted_loader(train_ds, cfg, cur_labels, pow_gamma=2.0)

        # Optional Warmup Ende: nach head_warmup_epochs wieder gewählte Policy aktivieren
        if cfg.head_warmup_epochs > 0 and epoch == (cfg.head_warmup_epochs + 1):
            apply_freeze_policy_slowfast(model, cfg.freeze_policy, bn_freeze_affine=cfg.bn_eval_freeze_affine)

        # Update Loss-Gewichte
        class_weights = get_effective_class_weights(cfg, num_classes, device, cur_labels)
        if use_focal:
            criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        for i, w in enumerate(class_weights.detach().cpu().numpy()):
            writer.add_scalar(f"class_weights/{cfg.class_list[i]}", float(w), epoch)

        model.train()
        run_loss, run_n = 0.0, 0
        t0 = time.time()

        for bidx, ((slow, fast), y) in enumerate(train_loader, 1):
            slow, fast = slow.to(device, non_blocking=True), fast.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)  # long

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled() and use_amp_this_epoch:
                with torch.amp.autocast("cuda", enabled=True, dtype=amp_dtype):
                    logits = model([slow, fast])  # (B, C)
                    loss = criterion(logits, y)
            else:
                logits = model([slow, fast])
                loss = criterion(logits, y)

            # NaN/Inf Guards
            if (not torch.isfinite(loss)) or (not torch.isfinite(logits).all()):
                print(f"[WARN] non-finite {'loss' if not torch.isfinite(loss) else 'logits'} @ epoch {epoch} step {bidx} -> skip batch")
                optimizer.zero_grad(set_to_none=True)
                continue

            if scaler.is_enabled() and use_amp_this_epoch:
                scaler.scale(loss).backward()
                grad_norm = _global_grad_norm(model.parameters())
                # Gradient Clipping before optimizer step
                clip_val = 1.0 if just_unfroze else cfg.grad_clip_max_norm
                if clip_val > 0:
                    try:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                    except Exception:
                        pass
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = _global_grad_norm(model.parameters())
                # Gradient Clipping before optimizer step
                clip_val = 1.0 if just_unfroze else cfg.grad_clip_max_norm
                if clip_val > 0:
                    try:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                    except Exception:
                        pass
                optimizer.step()

            run_loss += loss.item() * y.size(0)
            run_n += y.size(0)

            if global_step % cfg.log_every_n_steps == 0:
                for gi, g in enumerate(optimizer.param_groups):
                    lr = g.get("lr", cfg.lr)
                    writer.add_scalar(f"train/lr_group{gi}_{g.get('name','g')}", lr, global_step)
                writer.add_scalar("train/loss_step", loss.item(), global_step)
                writer.add_scalar("train/grad_norm_step", grad_norm, global_step)

                elapsed = time.time() - t0
                total_batches = len(train_loader)
                avg_batch_time = elapsed / max(1, bidx)
                est_total_sec = avg_batch_time * total_batches
                est_remaining_sec = max(0.0, est_total_sec - elapsed)
                writer.add_scalar("time/epoch_elapsed_sec", elapsed, global_step)
                writer.add_scalar("time/epoch_eta_sec", est_remaining_sec, global_step)
                writer.add_scalar("time/epoch_total_est_sec", est_total_sec, global_step)
                writer.flush()

            if cfg.print_every_10 > 0 and (bidx % cfg.print_every_10 == 0):
                avg10 = run_loss / max(1, run_n)
                print(f"[Epoch {epoch:02d}] step {bidx:05d} | avg_loss_so_far {avg10:.4f}", flush=True)
                writer.add_scalar("train/loss_every10", avg10, global_step)
                writer.flush()

            if cfg.print_every_100 > 0 and (bidx % cfg.print_every_100 == 0):
                avg100 = run_loss / max(1, run_n)
                print(f"[Epoch {epoch:02d}] step {bidx:05d} | avg_loss_so_far(100) {avg100:.4f}", flush=True)
                writer.add_scalar("train/loss_every100", avg100, global_step)
                writer.flush()

            global_step += 1

        tr_loss = run_loss / max(1, run_n)
        writer.add_scalar("train/loss_epoch", tr_loss, epoch)
        writer.add_scalar("train/steps_per_epoch", bidx, epoch)
        writer.add_scalar("time/epoch_seconds", time.time() - t0, epoch)

        # ---- Validation ----
        val_metrics = evaluate_multiclass(
            val_loader, model, criterion, device,
            num_classes=num_classes, use_amp=use_amp_this_epoch, amp_dtype=amp_dtype,
            event_thresh=cfg.event_thresh, no_event_dampen=cfg.no_event_dampen
        )

        writer.add_scalar("val/loss", val_metrics['loss'], epoch)
        writer.add_scalar("val/acc", val_metrics['acc'], epoch)
        writer.add_scalar("val/f1_macro", val_metrics['f1_macro'], epoch)
        writer.add_scalar("val/f1_micro", val_metrics['f1_micro'], epoch)
        writer.add_scalar("val/f1_weighted", val_metrics['f1_weighted'], epoch)

        # Per-Class
        if val_metrics["per_class"]:
            for i, cname in enumerate(cfg.class_list):
                writer.add_scalar(f"val/per_class/{cname}_precision", val_metrics["per_class"]["precision"][i], epoch)
                writer.add_scalar(f"val/per_class/{cname}_recall",    val_metrics["per_class"]["recall"][i], epoch)
                writer.add_scalar(f"val/per_class/{cname}_f1",        val_metrics["per_class"]["f1"][i], epoch)

        # Confusion Matrix
        _add_confusion_matrix_image(writer, "val/confusion_matrix", val_metrics["cm"], list(cfg.class_list), epoch)

        # ---- Best-Selection
        current_metric = (val_metrics["f1_macro"] if cfg.best_metric == "val_f1_macro" else -val_metrics["loss"])
        improved = current_metric > best_metric_value
        if improved:
            best_metric_value = current_metric
            best_path = os.path.join(cfg.out_dir, f"best_{cfg.model_type}.pth")
            torch.save(model.state_dict(), best_path)
            meta = {
                "epoch": epoch,
                "best_metric": cfg.best_metric,
                "best_metric_value": float(best_metric_value),
                "model_type": cfg.model_type,
                "class_list": list(cfg.class_list),
                "cfg": asdict(cfg),
                "tb_log_dir": log_dir,
            }
            os.makedirs(cfg.out_dir, exist_ok=True)
            with open(os.path.join(cfg.out_dir, f"best_{cfg.model_type}.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            writer.add_text("checkpoints/best",
                            f"epoch={epoch}, {cfg.best_metric}={best_metric_value:.4f}, path={best_path}",
                            epoch)
            logger.info("✅ New BEST (%s=%.3f) -> %s", cfg.best_metric, best_metric_value, best_path)
            es_counter = 0
        else:
            es_counter += 1

        # ---- Scheduler Step
        if cfg.scheduler == "cosine":
            if isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR):
                scheduler.step()
        elif cfg.scheduler == "plateau_val":
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                monitored = val_metrics['loss'] if cfg.plateau_metric == "val_loss" else val_metrics['f1_macro']
                scheduler.step(monitored)

        # optionale Modellhistogramme
        if cfg.log_weight_hist_every_n_epochs > 0 and (epoch % cfg.log_weight_hist_every_n_epochs == 0):
            for name, p in model.named_parameters():
                if p is not None and p.requires_grad and p.data is not None:
                    writer.add_histogram(f"model/weights/{name}", p.detach().cpu().numpy(), epoch)
                if p is not None and p.grad is not None:
                    writer.add_histogram(f"model/grads/{name}", p.grad.detach().cpu().numpy(), epoch)

        # Save pro Epoche (alle)
        ep_path = os.path.join(cfg.out_dir, f"{cfg.model_type}_ws{cfg.window_size}_e{epoch}.pth")
        torch.save(model.state_dict(), ep_path)
        writer.add_text("checkpoints/epoch", f"epoch={epoch}, path={ep_path}", epoch)

        # Early Stopping?
        if cfg.early_stopping_patience > 0 and es_counter >= cfg.early_stopping_patience:
            logger.info("⏹️ Early stopping after no improvement for %d epochs.", es_counter)
            break

        writer.flush()

    # ---- Finale Test-Evaluation ----
    if best_path and os.path.isfile(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device), strict=False)
        logger.info("Loaded BEST checkpoint for final test: %s", best_path)

    final_amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    test_metrics = evaluate_multiclass(
        test_loader, model, criterion, device,
        num_classes=num_classes, use_amp=(cfg.use_amp and device.type == "cuda"), amp_dtype=final_amp_dtype,
        event_thresh=cfg.event_thresh, no_event_dampen=cfg.no_event_dampen
    )

    writer.add_scalar("test/loss", test_metrics['loss'])
    writer.add_scalar("test/acc", test_metrics['acc'])
    writer.add_scalar("test/f1_macro", test_metrics['f1_macro'])
    writer.add_scalar("test/f1_micro", test_metrics['f1_micro'])
    writer.add_scalar("test/f1_weighted", test_metrics['f1_weighted'])
    _add_confusion_matrix_image(writer, "test/confusion_matrix", test_metrics["cm"], list(cfg.class_list), 0)

    logger.info("[TEST] Loss %.4f | Acc %.3f | F1(macro) %.3f | F1(micro) %.3f | F1(weighted) %.3f",
                test_metrics['loss'], test_metrics['acc'],
                test_metrics['f1_macro'], test_metrics['f1_micro'], test_metrics['f1_weighted'])

    # CSV-Export der Test-Metriken
    out_csv = os.path.join(cfg.out_dir, f"{cfg.model_type}_test_metrics.csv")
    df = pd.DataFrame({
        "metric": ["loss", "acc", "f1_macro", "f1_micro", "f1_weighted"],
        "value": [test_metrics['loss'], test_metrics['acc'],
                  test_metrics['f1_macro'], test_metrics['f1_micro'], test_metrics['f1_weighted']]
    })
    df.to_csv(out_csv, index=False)
    logger.info("✅ Test-Metriken gespeichert unter: %s", out_csv)

    writer.close()
    return model


# ============================================================
# Optional: Ein einzelnes Video scoren & CSV exportieren
# ============================================================
@torch.no_grad()
def test_single_video_multiclass(cfg: Config, model_path: str, video_path: str, label_path: str, batch_size: int = 8):
    device = torch.device(cfg.device)
    num_classes = len(cfg.class_list)
    class_to_id = {c: i for i, c in enumerate(cfg.class_list)}

    model = build_model_slowfast_mc(device, num_classes=num_classes, pretrained=False)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Fehler beim Öffnen von {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_ids = build_frame_class_ids(label_path, total_frames, cfg.fps, class_to_id=class_to_id, default_class_id=0)

    window = cfg.window_size
    stride = cfg.test_stride
    tfm = build_tfm_cpu(cfg.normalize, (cfg.resize_h, cfg.resize_w), cfg.mean, cfg.std)

    frame_buffer = deque(maxlen=window)
    preds_cls, centers = [], []

    read_idx = 0
    while read_idx < window:
        ret, fr = cap.read()
        if not ret:
            break
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        frame_buffer.append(fr)
        read_idx += 1

    batch_x, batch_centers = [], []
    t_centers = []
    with torch.no_grad():
        frame_idx = read_idx - 1
        while True:
            if len(frame_buffer) < window:
                break
            frames = list(frame_buffer)
            slow, fast = make_slowfast_tensors_cpu(frames, cfg.t_fast, cfg.t_fast // cfg.alpha, tfm)
            batch_x.append([slow, fast])
            center_idx = frame_idx - (window - 1) + window // 2
            batch_centers.append(center_idx)

            if len(batch_x) >= batch_size:
                slow_b = torch.stack([b[0] for b in batch_x]).to(device)
                fast_b = torch.stack([b[1] for b in batch_x]).to(device)
                logits = model([slow_b, fast_b])
                logits_eval = logits.detach().clone()
                try:
                    logits_eval[:, 0] -= float(cfg.no_event_dampen)
                except Exception:
                    pass
                probs = torch.softmax(logits_eval, dim=1).to(torch.float32).cpu().numpy()
                out = event_priority_pred(probs, event_thresh=float(cfg.event_thresh), no_event_id=0)
                preds_cls.extend(out.tolist())
                centers.extend(batch_centers)
                batch_x.clear()
                batch_centers.clear()

            ret = True
            for _ in range(stride):
                ret, fr = cap.read()
                frame_idx += 1
                if not ret:
                    break
                fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                frame_buffer.append(fr)
            if not ret:
                break

        if batch_x:
            slow_b = torch.stack([b[0] for b in batch_x]).to(device)
            fast_b = torch.stack([b[1] for b in batch_x]).to(device)
            logits = model([slow_b, fast_b])
            logits_eval = logits.detach().clone()
            try:
                logits_eval[:, 0] -= float(cfg.no_event_dampen)
            except Exception:
                pass
            probs = torch.softmax(logits_eval, dim=1).to(torch.float32).cpu().numpy()
            out = event_priority_pred(probs, event_thresh=float(cfg.event_thresh), no_event_id=0)
            preds_cls.extend(out.tolist())
            centers.extend(batch_centers)

    cap.release()

    centers = np.array(centers)
    t_centers = centers / cfg.fps if centers.size > 0 else np.array([], dtype=np.float32)
    gt_labels = []
    for c in centers.astype(int):
        start = int(max(0, c - (window // 2)))
        end = int(min(total_frames, start + window))
        gt_labels.append(
            window_majority_class(frame_ids[start:end], min_fraction=cfg.min_fraction_event, no_event_id=0) if end > start else 0
        )
    gt_labels = np.array(gt_labels, dtype=np.int64)

    # Metriken berechnen
    metrics = {}
    if len(preds_cls) > 0 and gt_labels.size > 0:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
        y_pred = np.array(preds_cls, dtype=int)
        y_true = gt_labels.astype(int)
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(num_classes)), zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        metrics = {
            "accuracy": float(acc),
            "per_class_precision": prec.tolist(),
            "per_class_recall": rec.tolist(),
            "per_class_f1": f1.tolist(),
            "confusion_matrix": cm.tolist(),
            "class_names": list(cfg.class_list),
            "n_windows": int(len(y_true)),
        }

        # Ausgabe
        print("\n=== Single Video Metrics ===")
        print(f"Video: {os.path.basename(video_path)} | Windows: {len(y_true)} | Accuracy: {acc:.4f}")
        print("Per-Class (prec | rec | f1):")
        for i, cname in enumerate(cfg.class_list):
            print(f"  {cname:12s}  {prec[i]:.3f}  {rec[i]:.3f}  {f1[i]:.3f}")
        print("Confusion Matrix (rows=true, cols=pred):")
        # format confusion matrix compact
        header = "       " + " ".join([f"{cname[:6]:>6s}" for cname in cfg.class_list])
        print(header)
        for i, row in enumerate(cm):
            row_str = " ".join([f"{int(v):6d}" for v in row])
            print(f"{cfg.class_list[i][:6]:>6s} {row_str}")
    else:
        print("⚠️ Keine auswertbaren Fenster für Metriken.")

    # Export CSV mit Predictions & optional Ground Truth
    try:
        csv_name = os.path.splitext(os.path.basename(video_path))[0] + "_preds_mc.csv"
        csv_path = os.path.join(os.path.dirname(video_path), csv_name)
        df = pd.DataFrame({
            "center_frame": centers.astype(int) if centers.size > 0 else np.array([], dtype=int),
            "time_s": t_centers,
            "pred_class_id": np.array(preds_cls, dtype=int),
            "pred_class_name": [cfg.class_list[i] for i in preds_cls] if len(preds_cls) > 0 else [],
            "gt_class_id": gt_labels,
            "gt_class_name": [cfg.class_list[i] for i in gt_labels] if gt_labels.size > 0 else [],
        })
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"✅ Predictions exported to CSV: {csv_path}")
    except Exception as e:
        print("⚠️ Could not write CSV:", e)

    return preds_cls, t_centers, gt_labels, metrics

