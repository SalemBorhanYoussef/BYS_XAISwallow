# ============================================================
# train_optflow.py
# Lightweight Single-Path Optical Flow Baseline (Mag + Angle)
#
# Anforderungen:
#   pip install torch torchvision tensorboard opencv-python scikit-learn
# ============================================================

import os
import re
import math
import time
import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
try:
    from precompute_optflow_dis import load_h5_flow
    PRECOMP_AVAILABLE = True
except Exception:
    PRECOMP_AVAILABLE = False
from glob import glob
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from collections import deque

# Optional X3D wrapper (builds x3d_s adapted for optical-flow inputs)
try:
    from optflow_x3d import build_x3d_optflow  # type: ignore
except Exception:
    build_x3d_optflow = None

# ============================================================
# Logging
# ============================================================
logger = logging.getLogger("train_optflow")
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
    window_size: int = 32              # Anzahl Frames pro Fenster
    train_stride: int = 4
    val_stride: int = 8
    test_stride: int = 8

    # --- Optimierung
    batch_size: int = 8
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 0.0

    # scheduler: "none" | "cosine" | "plateau_val"
    scheduler: str = "cosine"
    plateau_metric: str = "val_loss"   # "val_loss" (min) | "val_f1_best" (max)
    plateau_patience: int = 3
    plateau_factor: float = 0.5
    plateau_min_lr: float = 1e-7

    early_stopping_patience: int = 7

    # --- System
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0
    pin_memory: bool = torch.cuda.is_available()
    use_amp: bool = True

    # --- Checkpoints
    out_dir: str = "./models/"
    ckpt: Optional[str] = None

    # --- Labels (Eval-Binarisierung)
    positive_label: str = "none"
    pos_threshold_eval: float = 0.5

    # --- Vorverarbeitung (auf Flow-Kanälen)
    # Für 2 Kanäle (Mag, Ang). Wenn normalize=False, werden Kanäle skaliert:
    # Mag -> z-Norm per Fenster, Angle -> /pi auf [0,2] und dann z-Norm.
    normalize: bool = False
    mean: Tuple[float, float] = (0.0, 0.0)
    std: Tuple[float, float] = (1.0, 1.0)

    # --- Tricks
    epoch_jitter: bool = True

    # --- TensorBoard
    log_dir_base: str = "runs/optflow"
    log_every_n_steps: int = 20
    log_weight_hist_every_n_epochs: int = 5
    print_every_10: int = 10
    print_every_100: int = 100

    # --- Freeze / Finetune Policy (nicht relevant, aber Felder gelassen)
    freeze_policy: str = "none"
    pretrained_backbone: bool = False
    bn_eval_freeze_affine: bool = False

    backbone_lr: float = 1e-4
    head_lr: float = 1e-4

    # --- Auswahl für bestes Modell
    best_metric: str = "val_f1_best"

    # --- Diskrete Threshold-Kandidaten
    thresh_candidates: Tuple[float, ...] = (0.2, 0.4, 0.5, 0.6, 0.8)

    # --- Modelltyp (für Logs, Checkpoints etc.)
    model_type: str = "optflow"

    flow_type: str = "dis"  # oder "raft"
    # --- Precomputed flow support (in-place lookup only)
    use_precomputed_flow: bool = True

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

# ============================================================
# Utility-Funktionen
# ============================================================
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

def list_video_label_pairs(ds_dir: str) -> List[Tuple[str, str]]:
    """Finde (video, label)-Paare rekursiv."""
    pairs: List[Tuple[str, str]] = []

    if not os.path.isdir(ds_dir):
        print(f"[WARN] Das Verzeichnis '{ds_dir}' existiert nicht.")
        return pairs

    exts = ("mp4", "avi", "mov", "mkv")
    matches = []
    for ext in exts:
        pattern = os.path.join(ds_dir, "**", f"*roi*.{ext}")
        matches.extend(sorted(glob(pattern, recursive=True)))

    print(f"[INFO] {len(matches)} ROI-Videos gefunden.\n")

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
            if m:
                num = m.group(1)
                ann_matches = sorted(glob(os.path.join(dirpath, f"annotation*{num}*.txt")))
                if ann_matches:
                    found = ann_matches[0]
            if not found:
                ann_txts = sorted(glob(os.path.join(dirpath, "*annotation*.txt")))
                if ann_txts:
                    found = ann_txts[0]

        if found:
            pairs.append((os.path.abspath(v), os.path.abspath(found)))
        else:
            print(f"[WARN] Keine Label-Datei gefunden für: {v}")

    print(f"--- Insgesamt {len(pairs)} Paare erkannt. ---\n")
    return pairs

def elan_flags_from_txt(txt_path: str, total_frames: int, fps: int, positive_label: str) -> np.ndarray:
    flags = np.zeros(total_frames, dtype=np.int64)
    if not os.path.exists(txt_path):
        logger.warning("Label file not found: %s", txt_path)
        return flags
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            try:
                s = float(parts[3].replace(",", "."))
                e = float(parts[5].replace(",", "."))
            except ValueError:
                continue
            label = parts[8].strip().lower()
            if label == positive_label.lower():
                si = max(0, int(round(s * fps)))
                ei = min(total_frames - 1, int(round(e * fps)))
                if ei >= si:
                    flags[si: ei + 1] = 1
    return flags

def normalize_flow_tensor(x: torch.Tensor, use_stats: bool, mean: Tuple[float, float], std: Tuple[float, float]) -> torch.Tensor:
    """
    x: [2, T, H, W]  (Mag, Angle)
    Wenn use_stats=True: (x - mean)/std kanalweise.
    Wenn False: einfache robuste Normierung je Fenster.
       - Mag: z-Norm (über alle Pixel/Time, robust)
       - Angle: in [0, 2*pi) -> /pi (0..2) -> z-Norm
    """
    if use_stats:
        m = torch.tensor(mean, dtype=x.dtype, device=x.device).view(2, 1, 1, 1)
        s = torch.tensor(std, dtype=x.dtype, device=x.device).view(2, 1, 1, 1)
        s = torch.clamp(s, min=1e-6)
        return (x - m) / s

    # Robust-Normalisierung pro Fenster
    mag = x[0]  # [T,H,W]
    ang = x[1]  # [T,H,W], in rad

    # Angle in [0, 2] skalieren
    ang_scaled = ang / math.pi  # 0..~2

    def z_norm(t):
        mu = torch.mean(t)
        sd = torch.std(t)
        return (t - mu) / (sd + 1e-6)

    mag_n = z_norm(mag)
    ang_n = z_norm(ang_scaled)
    return torch.stack([mag_n, ang_n], dim=0)


# ============================================================
# Dataset – nutzt vorberechnete Flow-Videos (DIS oder RAFT)
# ============================================================
class OpticalFlowDataset(Dataset):
    """
    Liest vorberechnete Optical-Flow-Videos (Mag+Angle als Farbkodierung)
    und erzeugt daraus Zeitfenster [2, T, H, W].
    """
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        window_size: int,
        stride: int,
        fps: int,
        positive_label: str,
        resize_hw: Tuple[int, int],
        normalize: bool,
        mean: Tuple[float, float],
        std: Tuple[float, float],
        epoch_jitter: bool = True,
        flow_type: str = "dis",
    ):
        # For reliability we only support DIS here (CPU) — compute flow on the fly
        if flow_type.lower() != "dis":
            raise ValueError("Only flow_type='dis' is supported for on-the-fly computation")
        self.flow_type = "dis"
        self.window = int(window_size)
        self.stride = max(1, int(stride))
        self.fps = int(fps)
        self.resize_hw = (int(resize_hw[0]), int(resize_hw[1]))
        self.normalize = bool(normalize)
        self.mean, self.std = mean, std
        self.enable_epoch_jitter = bool(epoch_jitter)
        self.index: List[Tuple[str, int, np.ndarray, int, str]] = []
        self.epoch: int = 0

        for flow_path, label_path in pairs:
            if not os.path.exists(flow_path):
                logger.warning("Kein Flow-Video gefunden: %s", flow_path)
                continue

            if not os.path.exists(label_path):
                logger.warning("Keine Label-Datei gefunden: %s", label_path)
                continue

            # Determine number of frames (for raw videos) or number of flow pairs (for precomputed flow files)
            ext = os.path.splitext(flow_path)[1].lower()
            try:
                if ext == '.h5':
                    if PRECOMP_AVAILABLE:
                        f, ds, _scale = load_h5_flow(flow_path)
                        pairs_count = ds.shape[0]
                        f.close()
                    else:
                        # try with h5py directly
                        import h5py as _h5
                        with _h5.File(flow_path, 'r') as _f:
                            pairs_count = _f['flow'].shape[0]
                    n = pairs_count + 1
                elif ext == '.npz':
                    with np.load(flow_path) as dat:
                        pairs_count = dat['flow'].shape[0]
                    n = pairs_count + 1
                elif ext == '.npy':
                    arr = np.load(flow_path, mmap_mode='r')
                    pairs_count = arr.shape[0]
                    n = pairs_count + 1
                else:
                    cap = cv2.VideoCapture(flow_path)
                    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                    cap.release()
            except Exception as e:
                logger.warning("Fehler beim Lesen von %s: %s", flow_path, str(e))
                continue

            if n <= 1:
                continue
            flags = elan_flags_from_txt(label_path, n, self.fps, positive_label)
            max_start = max(1, n - self.window + 1)
            for s in range(0, max_start, self.stride):
                self.index.append((flow_path, s, flags, n, label_path))

        # Create DIS optical flow calculator once per dataset instance (CPU)
        try:
            self.dis_calc = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        except Exception:
            # Fallback if DIS not available
            self.dis_calc = None


        self.soft_labels = np.array(
            [float(flags[s:s+self.window].mean()) for (_, s, flags, _, _) in self.index],
            dtype=np.float32
        )

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _epoch_jitter(self, idx: int) -> int:
        if not self.enable_epoch_jitter or self.stride <= 1:
            return 0
        return ((idx * 1315423911) ^ (self.epoch * 2654435761)) % self.stride

    def epoch_soft_labels(self) -> np.ndarray:
        labs = np.zeros(len(self.index), dtype=np.float32)
        for i, (_, start, flags, n, _) in enumerate(self.index):
            j = self._epoch_jitter(i)
            s = min(start + j, max(0, n - self.window))
            labs[i] = float(flags[s:s+self.window].mean())
        return labs

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        vpath, start, flags, n, _ = self.index[idx]
        j = self._epoch_jitter(idx)
        s = min(start + j, max(0, n - self.window))

        ext = os.path.splitext(vpath)[1].lower()
        mags, angs = [], []
        if ext in ('.h5', '.npz', '.npy'):
            # Load precomputed flow slices (pairs -> T flows)
            try:
                if ext == '.h5':
                    if PRECOMP_AVAILABLE:
                        f, ds, scale = load_h5_flow(vpath)
                        flows = ds[s:s + self.window]
                        flows = flows.astype(np.float32) / float(scale)
                        f.close()
                    else:
                        import h5py as _h5
                        with _h5.File(vpath, 'r') as _f:
                            ds = _f['flow']
                            scale = ds.attrs.get('scale', 1.0)
                            flows = ds[s:s + self.window].astype(np.float32) / float(scale)
                elif ext == '.npz':
                    dat = np.load(vpath)
                    flows = dat['flow'][s:s + self.window].astype(np.float32)
                else:
                    arr = np.load(vpath, mmap_mode='r')
                    flows = arr[s:s + self.window].astype(np.float32)
            except Exception:
                # fallback to zero flows
                flows = np.zeros((self.window, self.resize_hw[0], self.resize_hw[1], 2), dtype=np.float32)

            # Ensure we have exactly `window` flow frames: pad by repeating last flow if needed
            if getattr(flows, 'shape', None) is None:
                flows = np.array(flows)
            if flows.shape[0] < self.window:
                if flows.shape[0] == 0:
                    flows = np.zeros((self.window, self.resize_hw[0], self.resize_hw[1], 2), dtype=np.float32)
                else:
                    last = flows[-1]
                    pad_cnt = self.window - flows.shape[0]
                    pad = np.repeat(last[None, ...], pad_cnt, axis=0)
                    flows = np.concatenate([flows, pad], axis=0)

            for i in range(self.window):
                flow = flows[i]
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
                mags.append(mag.astype(np.float32))
                angs.append(ang.astype(np.float32))
        else:
            cap = cv2.VideoCapture(vpath)
            cap.set(cv2.CAP_PROP_POS_FRAMES, s)

            # We read window+1 frames to compute window flow frames (pairs)
            frames = []
            read_count = 0
            need = self.window + 1
            while read_count < need:
                ok, fr = cap.read()
                if not ok or fr is None:
                    break
                fr = cv2.resize(fr, (self.resize_hw[1], self.resize_hw[0]))
                frames.append(fr)
                read_count += 1
            cap.release()

            # If too short, pad with last frame
            if len(frames) == 0:
                pad = np.zeros((self.resize_hw[0], self.resize_hw[1], 3), dtype=np.uint8)
                frames = [pad] * need
            elif len(frames) < need:
                last = frames[-1]
                while len(frames) < need:
                    frames.append(last)

            # Convert to grayscale and compute DIS flow between consecutive frames
            for i in range(self.window):
                f1 = frames[i]
                f2 = frames[i + 1]
                g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
                g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
                if self.dis_calc is not None:
                    flow = self.dis_calc.calc(g1, g2, None)
                else:
                    # fallback: zero flow
                    flow = np.zeros((self.resize_hw[0], self.resize_hw[1], 2), dtype=np.float32)
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
                mags.append(mag.astype(np.float32))
                angs.append(ang.astype(np.float32))

        mag = np.stack(mags, axis=0)  # [T,H,W]
        ang = np.stack(angs, axis=0)

        x_np = np.stack([mag, ang], axis=0).astype(np.float32)  # [2,T,H,W]
        x = torch.from_numpy(x_np)
        x = normalize_flow_tensor(x, use_stats=self.normalize, mean=self.mean, std=self.std)

        y = float(flags[s:s + self.window].mean())
        return x, torch.tensor(y, dtype=torch.float32)


# ============================================================
# Modell – kompaktes 3D-CNN
# ============================================================
class OpticalFlowNet(nn.Module):
    """
    Improved lightweight 3D residual network for optical-flow inputs.
    Input: [B, 2, T, H, W]  (2 = (Mag, Angle))
    Output: 1 Logit

    Design goals:
    - Larger temporal receptive field than the simple baseline
    - Residual blocks for stable training
    - Moderate parameter count to keep training fast
    """
    def __init__(self, in_ch: int = 2, base_ch: int = 32, dropout: float = 0.2):
        super().__init__()

        def conv3x3(in_c, out_c, k=(3,3,3), stride=(1,1,1)):
            return nn.Conv3d(in_c, out_c, kernel_size=k, stride=stride, padding=tuple((kk//2) for kk in k), bias=False)

        class ResidualBlock(nn.Module):
            def __init__(self, in_c, out_c, stride=(1,1,1)):
                super().__init__()
                self.conv1 = conv3x3(in_c, out_c, k=(3,3,3), stride=stride)
                self.bn1 = nn.BatchNorm3d(out_c)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = conv3x3(out_c, out_c, k=(3,3,3), stride=(1,1,1))
                self.bn2 = nn.BatchNorm3d(out_c)
                self.downsample = None
                if stride != (1,1,1) or in_c != out_c:
                    self.downsample = nn.Sequential(
                        nn.Conv3d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm3d(out_c)
                    )

            def forward(self, x):
                identity = x
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = self.bn2(out)
                if self.downsample is not None:
                    identity = self.downsample(identity)
                out += identity
                return self.relu(out)

        # Stem: early spatial downsampling but keep temporal resolution
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, base_ch, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3), bias=False),
            nn.BatchNorm3d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        )

        # Residual stages: (temporal, spatial) strides chosen to keep temporal granularity
        self.layer1 = nn.Sequential(
            ResidualBlock(base_ch, base_ch * 2, stride=(1,1,1)),
            ResidualBlock(base_ch * 2, base_ch * 2, stride=(1,1,1)),
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(base_ch * 2, base_ch * 4, stride=(1,2,2)),
            ResidualBlock(base_ch * 4, base_ch * 4, stride=(1,1,1)),
        )
        # A light temporal-downsampling stage to increase receptive field
        self.layer3 = nn.Sequential(
            ResidualBlock(base_ch * 4, base_ch * 8, stride=(2,2,2)),
            ResidualBlock(base_ch * 8, base_ch * 8, stride=(1,1,1)),
        )

        self.head_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(base_ch * 8, 1)

        # Log parameter count for quick visibility
        try:
            total_params = sum(p.numel() for p in self.parameters())
            logger.info("OpticalFlowNet (improved) parameter count: %d", int(total_params))
        except Exception:
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2, T, H, W]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.head_pool(x)   # [B, C, 1,1,1]
        x = x.flatten(1)
        x = self.dropout(x)
        out = self.head(x)
        return out

# ============================================================
# DataLoader-Helfer
# ============================================================
def _build_weighted_loader(ds: Dataset, cfg: Config, cur_soft: np.ndarray) -> DataLoader:
    pos_mask = (cur_soft >= cfg.pos_threshold_eval)
    pos_count = int(pos_mask.sum())
    neg_count = len(cur_soft) - pos_count
    pos_count = max(1, pos_count)
    neg_count = max(1, neg_count)
    pos_w = neg_count / pos_count
    weights_np = np.where(pos_mask, pos_w, 1.0).astype(np.float32)

    sampler = WeightedRandomSampler(weights=weights_np.tolist(), num_samples=len(ds), replacement=True)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
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
    all_pairs = [x for x in all_pairs if "lying" in x[0]]
    # If configured, try to map raw video paths to precomputed flow files
    def find_flow_for_video(video_path: str, flow_root: Optional[str]) -> Optional[str]:
        """Find a precomputed flow file for a given video.

        Priority order:
        1. Same directory as the video (exact stem match, then fuzzy match)
        2. If provided, search `flow_root` recursively (exact then fuzzy)

        Returns path string or None if not found.
        """
        stem = Path(video_path).stem
        video_dir = Path(video_path).parent

        # 1) Look in the same directory as the ROI video first
        for ext in ('.h5', '.npz', '.npy'):
            p = video_dir / (stem + ext)
            if p.exists():
                return str(p)
        # fuzzy in same dir
        for ext in ('.h5', '.npz', '.npy'):
            matches = list(video_dir.glob(f"*{stem}*{ext}"))
            if matches:
                return str(matches[0])

        # 2) If a flow_root is supplied, search it (recursive)
        if flow_root:
            proot = Path(flow_root)
            if proot.exists():
                for ext in ('.h5', '.npz', '.npy'):
                    matches = list(proot.rglob(stem + ext))
                    if matches:
                        return str(matches[0])
                # fuzzy match in provided root
                for ext in ('.h5', '.npz', '.npy'):
                    matches = list(proot.rglob(f"*{stem}*{ext}"))
                    if matches:
                        return str(matches[0])

        return None
    # ggf. Video-Subset wählen: (no hard-coded filters)
    if not all_pairs:
        raise RuntimeError("Keine (video, txt)-Paare im Trainingsordner gefunden.")

    rng = np.random.RandomState(cfg.seed)
    idx = np.arange(len(all_pairs))
    rng.shuffle(idx)

    if cfg.dataset_val_dir or cfg.dataset_test_dir:
        if cfg.dataset_val_dir:
            val_pairs = list_video_label_pairs(cfg.dataset_val_dir)
            val_pairs = [x for x in val_pairs if "lying" in x[0]]
            train_pairs = all_pairs
        else:
            train_pairs, val_pairs = _split_train_val_pairs(all_pairs, cfg.val_split_frac, cfg.seed)
        if cfg.dataset_test_dir:
            test_pairs = list_video_label_pairs(cfg.dataset_test_dir)
            test_pairs = [x for x in test_pairs if "lying" in x[0]]
        else:
            # 85/15 Split zu train/val; Test = leer
            test_pairs = []
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

    logger.info(
        "Found train videos=%d | val videos=%d | test videos=%d",
        len(train_pairs), len(val_pairs), len(test_pairs)
    )

    # Map video->precomputed flow files if requested.
    # NOTE: per user request we only look for precomputed flows in the same
    if cfg.use_precomputed_flow:
        def map_pairs(pairs):
            mapped = []
            for v, label in pairs:
                # find_flow_for_video will search the video's directory first; pass None
                # to avoid searching any global root.
                fpath = find_flow_for_video(v, None)
                if fpath:
                    mapped.append((fpath, label))
                else:
                    logger.warning("Kein vorcomputiertes Flow gefunden (in-place) für %s", v)
            return mapped

        train_pairs = map_pairs(train_pairs)
        val_pairs = map_pairs(val_pairs)
        test_pairs = map_pairs(test_pairs)

    # --- Datasets ---
    ds_kwargs = dict(
        window_size=cfg.window_size,
        fps=cfg.fps,
        positive_label=cfg.positive_label,
        resize_hw=(cfg.resize_h, cfg.resize_w),
        normalize=cfg.normalize,
        mean=cfg.mean,
        std=cfg.std,
        flow_type=cfg.flow_type,
    )

    train_ds = OpticalFlowDataset(
        train_pairs, stride=cfg.train_stride, epoch_jitter=cfg.epoch_jitter, **ds_kwargs
    )
    val_ds = OpticalFlowDataset(
        val_pairs, stride=cfg.val_stride, epoch_jitter=False, **ds_kwargs
    )
    test_ds = OpticalFlowDataset(
        test_pairs, stride=cfg.test_stride, epoch_jitter=False, **ds_kwargs
    )

    # --- Dataloader ---
    train_ds.set_epoch(0)
    cur_soft = train_ds.epoch_soft_labels() if cfg.epoch_jitter else train_ds.soft_labels

    train_loader = _build_weighted_loader(train_ds, cfg, cur_soft)
    val_loader = DataLoader(
        val_ds, batch_size=8, shuffle=False,
        num_workers=4, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_ds, batch_size=8, shuffle=False,
        num_workers=4, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )

    logger.info(
        "Windows: Train=%d | Val=%d | Test=%d | pos_mean(train)=%.4f | win_pos_rate(train)=%.4f",
        len(train_ds), len(val_ds), len(test_ds),
        float(cur_soft.mean()), float((cur_soft >= cfg.pos_threshold_eval).mean())
    )

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds

# ============================================================
# Eval + Loss + Plots
# ============================================================
def _best_threshold_from_list(gt_bin: np.ndarray, probs: np.ndarray, candidates: Tuple[float, ...]):
    from sklearn.metrics import precision_recall_fscore_support
    best_t, best_f1, best_rec = None, -1.0, 0.0
    for t in candidates:
        pr_bin = (probs >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(gt_bin, pr_bin, average="binary", zero_division=0)
        better = (f1 > best_f1) or (np.isclose(f1, best_f1) and rec > best_rec)
        if better:
            best_t, best_f1, best_rec = float(t), float(f1), float(rec)
    return (best_t if best_t is not None else float(candidates[0])), float(max(best_f1, 0.0))

@torch.no_grad()
def evaluate_windows_with_loss(
    loader: DataLoader,
    model: nn.Module,
    criterion,
    device: torch.device,
    pos_threshold: float,
    thresh_candidates: Tuple[float, ...],
    log_pr_roc: bool = True,
    use_amp: bool = False,
    smooth_window: int = 5,
):
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, roc_auc_score,
        confusion_matrix, precision_recall_curve, roc_curve, auc as sk_auc,
        average_precision_score
    )

    # Reduce fragmentation and cached memory before evaluation reporting
    if device.type == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
            
    model.eval()
    total_loss, total_n = 0.0, 0
    all_probs, all_soft = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).unsqueeze(1)
        # Use AMP autocast during evaluation to reduce activation memory when available
        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast(enabled=True):
                logits = model(x)
                loss = criterion(logits, y)
        else:
            logits = model(x)
            loss = criterion(logits, y)
        probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
        soft = y.squeeze(1).detach().cpu().numpy()

        total_loss += float(loss.item()) * y.size(0)
        total_n += y.size(0)
        all_probs.append(probs)
        all_soft.append(soft)

    if not all_probs:
        return {
            "loss": 0.0, "acc": 0.0, "prec": 0.0, "rec": 0.0, "f1": 0.0, "auc": float("nan"),
            "ap": float("nan"), "best_thresh": pos_threshold, "f1_best": float("nan"),
            "curves": None, "cm": np.array([[0,0],[0,0]], dtype=int),
            "y_true": np.array([], dtype=np.int32), "y_prob": np.array([], dtype=np.float32)
        }

    probs = np.concatenate(all_probs)
    soft = np.concatenate(all_soft)

    # Temporal smoothing (uniform moving average) of window-level predictions
    k = int(max(0, smooth_window)) if smooth_window is not None else 0
    if k <= 1:
        probs_sm = probs.astype(np.float32)
    else:
        kernel = np.ones(k, dtype=np.float32) / float(k)
        try:
            probs_sm = np.convolve(probs.astype(np.float32), kernel, mode="same")
        except Exception:
            probs_sm = probs.astype(np.float32)
    gt_bin = (soft >= pos_threshold).astype(int)
    # Use smoothed probabilities for thresholding/curves/metrics
    pr_bin_fix = (probs_sm >= pos_threshold).astype(int)

    acc = accuracy_score(gt_bin, pr_bin_fix)
    prec, rec, f1, _ = precision_recall_fscore_support(gt_bin, pr_bin_fix, average="binary", zero_division=0)

    precision, recall, pr_thresholds = precision_recall_curve(gt_bin, probs_sm)
    ap = average_precision_score(gt_bin, probs_sm)

    best_thresh, f1_best = _best_threshold_from_list(gt_bin, probs_sm, thresh_candidates)

    try:
        fpr, tpr, _ = roc_curve(gt_bin, probs_sm)
        roc_auc = sk_auc(fpr, tpr) if len(np.unique(gt_bin)) > 1 else float("nan")
        auc = roc_auc_score((soft > 0.0).astype(int), probs_sm)
    except Exception:
        fpr, tpr, roc_auc, auc = np.array([0,1]), np.array([0,1]), float("nan"), float("nan")

    cm = confusion_matrix(gt_bin, pr_bin_fix, labels=[0,1])

    curves = None
    if log_pr_roc:
        curves = {"precision": precision, "recall": recall, "fpr": fpr, "tpr": tpr, "roc_auc": float(roc_auc)}

    return {
        "loss": total_loss / max(1, total_n),
        "acc": float(acc), "prec": float(prec), "rec": float(rec), "f1": float(f1),
        "auc": float(auc), "ap": float(ap),
        "best_thresh": float(best_thresh), "f1_best": float(f1_best),
        "curves": curves,
        "cm": cm,
        "y_true": gt_bin.astype(np.int32),
        # y_prob: smoothed probabilities used for metrics; keep raw copy too
        "y_prob": probs_sm.astype(np.float32),
        "y_prob_raw": probs.astype(np.float32),
        "smooth_window": k,
    }

def _add_confusion_matrix_image(writer: SummaryWriter, tag: str, cm: np.ndarray, class_names=("neg","pos"), global_step: int = 0):
    import matplotlib.pyplot as plt
    import io
    fig = plt.figure(figsize=(3,3))
    ax = plt.gca()
    ax.imshow(cm, interpolation='nearest')
    ax.set_title("Confusion Matrix")
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120)
    plt.close(fig)
    buf.seek(0)
    import PIL.Image as Image
    img = Image.open(buf)
    img = np.array(img).transpose(2,0,1)  # CHW
    writer.add_image(tag, img, global_step)

def evaluate_confusion_matrix(metrics_dict, threshold_used):
    if "cm" in metrics_dict:
        return metrics_dict["cm"]
    return np.array([[0,0],[0,0]], dtype=int)

# ============================================================
# GPU Memory Cleanup Helper
# ============================================================
def free_cuda_memory(sync: bool = True) -> None:
    """Aggressively release cached CUDA memory.

    Called after validation/test evaluation phases to keep peak memory small.
    This does NOT free model parameters (still on device); it clears the
    allocator cache and triggers Python garbage collection.
    """
    try:
        import gc
        if torch.cuda.is_available():
            if sync:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            # Empty PyTorch cache
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            # Optionally reset peak stats (for monitoring)
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        # Run Python GC twice (generational)
        gc.collect()
        gc.collect()
    except Exception:
        # Silent failure – we never want training to crash due to cleanup.
        pass

# ============================================================
# Training Helpers
# ============================================================
def _global_grad_norm(parameters, norm_type: float = 2.0) -> float:
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0
    device = params[0].grad.device
    total = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in params]), norm_type)
    return total.item()

def init_head_bias_like(model: nn.Module, prior: float):
    """Setzt Bias im letzten Linear auf logit(prior/(1-prior)), wenn vorhanden."""
    p = float(max(1e-4, min(1 - 1e-4, prior)))
    bias = math.log(p / (1 - p))
    target = None
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.out_features == 1:
            target = m
    if target is not None and target.bias is not None:
        with torch.no_grad():
            target.bias.fill_(bias)

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
    # Eigener Unterordner mit Timestamp
    ts = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(cfg.log_dir_base, f"{cfg.model_type}_{ts}")
    os.makedirs(log_dir, exist_ok=True)
    # Meta-Datei
    meta = {
        "timestamp": ts,
        "model_type": cfg.model_type,
        "cfg": asdict(cfg),
    }
    with open(os.path.join(log_dir, "RUN_META.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text("config/json", f"```\n{cfg.to_json()}\n```", 0)
    return writer, log_dir

# ============================================================
# Train
# ============================================================
def train(cfg: Config):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)

    writer, log_dir = _make_run_dirs_and_writer(cfg)

    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = make_dataloaders(cfg, device)

    # ---- Modell ----
    if "x3d" in cfg.model_type.lower():
        if build_x3d_optflow is None:
            raise RuntimeError("X3D model requested but `optflow_x3d` or its dependencies are not available. Install pytorchvideo and ensure optflow_x3d.py is present.")
        model = build_x3d_optflow(device=device, pretrained=cfg.pretrained_backbone, in_ch=2)
    else:
        model = OpticalFlowNet(in_ch=2).to(device)
    if cfg.ckpt and os.path.isfile(cfg.ckpt):
        state = torch.load(cfg.ckpt, map_location=device)
        model.load_state_dict(state, strict=False)
        logger.info("Loaded weights from %s", cfg.ckpt)

    # Identify head modules for parameter groups (head vs backbone)
    head_modules = []
    # OpticalFlowNet has attribute .head; X3D wrapper replaced final Linear somewhere in the backbone
    if hasattr(model, "head") and isinstance(getattr(model, "head"), nn.Module):
        head_modules = [model.head]
    else:
        # find last nn.Linear with out_features==1
        for name, m in reversed(list(model.named_modules())):
            if isinstance(m, nn.Linear) and getattr(m, "out_features", None) == 1:
                head_modules = [m]
                break
    if not head_modules:
        # fallback: try to find any Linear
        for name, m in reversed(list(model.named_modules())):
            if isinstance(m, nn.Linear):
                head_modules = [m]
                break

    # Head-Bias (für Imbalance)
    init_soft = train_ds.epoch_soft_labels() if cfg.epoch_jitter else train_ds.soft_labels
    init_pos_rate = float((init_soft >= cfg.pos_threshold_eval).mean())
    init_head_bias_like(model, max(init_pos_rate, 1e-6))
    writer.add_scalar("train/init_pos_rate", init_pos_rate, 0)

    def make_criterion(pos_rate: float):
        pos_rate = max(min(float(pos_rate), 1.0 - 1e-6), 1e-6)
        pw = (1.0 - pos_rate) / pos_rate
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))

    criterion = make_criterion(init_pos_rate)

    # Param-Gruppen (Head vs Backbone) mit evtl. unterschiedlichen LRs
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
        plateau_kwargs = dict(
            mode=mode,
            factor=cfg.plateau_factor,
            patience=cfg.plateau_patience,
            min_lr=cfg.plateau_min_lr,
            eps=1e-8,
        )
        try:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, **plateau_kwargs)
        except TypeError:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **plateau_kwargs)
    # scheduler_mode is unused — no need to track it
    else:
        scheduler = None

    # Use the cuda.amp GradScaler in the canonical form; enable only when use_amp and CUDA available
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp and (device.type == "cuda"))

    global_step = 0
    best_metric_value = -1.0
    best_path = None
    best_val_thresh = cfg.pos_threshold_eval
    es_counter = 0

    # ---- Training ----
    for epoch in range(1, cfg.epochs + 1):
        # epoch jitter + sampler/pos_weight refresh
        if cfg.epoch_jitter:
            train_ds.set_epoch(epoch)
            cur_soft = train_ds.epoch_soft_labels()
        else:
            cur_soft = train_ds.soft_labels
        train_loader = _build_weighted_loader(train_ds, cfg, cur_soft)
        pos_rate_epoch = float((cur_soft >= cfg.pos_threshold_eval).mean())
        criterion = make_criterion(pos_rate_epoch)
        writer.add_scalar("train/pos_rate_epoch", pos_rate_epoch, epoch)

        model.train()
        run_loss, run_n = 0.0, 0
        t0 = time.time()
        for bidx, (x, y) in enumerate(train_loader, 1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                with torch.amp.autocast("cuda", enabled=True):
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                grad_norm = _global_grad_norm(model.parameters())
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                grad_norm = _global_grad_norm(model.parameters())
                optimizer.step()

            run_loss += loss.item() * y.size(0)
            run_n += y.size(0)

            if global_step % cfg.log_every_n_steps == 0:
                for gi, g in enumerate(optimizer.param_groups):
                    lr = g.get("lr", cfg.lr)
                    writer.add_scalar(f"train/lr_group{gi}_{g.get('name','g')}", lr, global_step)
                writer.add_scalar("train/loss_step", loss.item(), global_step)
                writer.add_scalar("train/grad_norm_step", grad_norm, global_step)
                # Timing & ETA for current epoch
                elapsed = time.time() - t0
                try:
                    total_batches = len(train_loader)
                except Exception:
                    total_batches = bidx
                avg_batch_time = elapsed / max(1, bidx)
                est_total_sec = avg_batch_time * total_batches
                est_remaining_sec = max(0.0, est_total_sec - elapsed)
                writer.add_scalar("time/epoch_elapsed_sec", elapsed, global_step)
                writer.add_scalar("time/epoch_eta_sec", est_remaining_sec, global_step)
                writer.add_scalar("time/epoch_total_est_sec", est_total_sec, global_step)
                writer.flush()

            if cfg.print_every_10 > 0 and (bidx % cfg.print_every_10 == 0):
                avg10 = run_loss / max(1, run_n)
                # compute elapsed and ETA for nicer console output
                elapsed = time.time() - t0
                try:
                    total_batches = len(train_loader)
                except Exception:
                    total_batches = bidx
                avg_batch_time = elapsed / max(1, bidx)
                est_total_sec = avg_batch_time * total_batches
                est_remaining_sec = max(0.0, est_total_sec - elapsed)
                eta_h = int(est_remaining_sec // 3600)
                eta_m = int((est_remaining_sec % 3600) // 60)
                eta_s = int(est_remaining_sec % 60)
                print(f"[Epoch {epoch:02d}] step {bidx:05d} | avg_loss_so_far {avg10:.4f} | elapsed {int(elapsed)}s | ETA {eta_h}h{eta_m}m{eta_s}s", flush=True)
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
        val_metrics = evaluate_windows_with_loss(
            val_loader, model, criterion, device,
            cfg.pos_threshold_eval, cfg.thresh_candidates, log_pr_roc=True,
            use_amp=cfg.use_amp,
        )

        # TB: Val
        writer.add_scalar("val/loss", val_metrics['loss'], epoch)
        writer.add_scalar("val/acc", val_metrics['acc'], epoch)
        writer.add_scalar("val/precision@fix", val_metrics['prec'], epoch)
        writer.add_scalar("val/recall@fix", val_metrics['rec'], epoch)
        writer.add_scalar("val/f1@fix", val_metrics['f1'], epoch)
        writer.add_scalar("val/auc", val_metrics['auc'], epoch)
        writer.add_scalar("val/AP", val_metrics['ap'], epoch)
        writer.add_scalar("val/f1_best_from_list", val_metrics['f1_best'], epoch)
        writer.add_scalar("val/best_thresh_from_list", val_metrics['best_thresh'], epoch)

        if val_metrics["y_true"].size > 0:
            writer.add_pr_curve(
                tag="val/pr_curve",
                labels=torch.from_numpy(val_metrics["y_true"]).int(),
                predictions=torch.from_numpy(val_metrics["y_prob"]).float(),
                global_step=epoch,
            )
        curves = val_metrics["curves"]
        if curves is not None:
            writer.add_scalar("val/roc_auc_curve", curves["roc_auc"], epoch)

        _add_confusion_matrix_image(
            writer, "val/confusion_matrix@fix",
            evaluate_confusion_matrix(val_metrics, cfg.pos_threshold_eval),
            class_names=("neg","pos"), global_step=epoch
        )

        # ---- Best-Selection (nach cfg.best_metric) ----
        current_metric = (val_metrics["f1_best"] if cfg.best_metric == "val_f1_best" else val_metrics["ap"])
        improved = (current_metric > best_metric_value)
        if improved:
            best_metric_value = current_metric
            best_val_thresh = float(val_metrics["best_thresh"])
            best_path = os.path.join(cfg.out_dir, f"best_{cfg.model_type}.pth")
            torch.save(model.state_dict(), best_path)
            meta = {
                "epoch": epoch,
                "best_metric": cfg.best_metric,
                "best_metric_value": best_metric_value,
                "best_threshold_from_list": best_val_thresh,
                "thresh_candidates": cfg.thresh_candidates,
                "model_type": cfg.model_type,
                "cfg": asdict(cfg),
                "tb_log_dir": log_dir,
            }
            os.makedirs(cfg.out_dir, exist_ok=True)
            with open(os.path.join(cfg.out_dir, f"best_{cfg.model_type}.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            writer.add_text("checkpoints/best",
                            f"epoch={epoch}, {cfg.best_metric}={best_metric_value:.4f}, "
                            f"thresh*={best_val_thresh:.3f}, path={best_path}",
                            epoch)
            logger.info("✅ New BEST (%s=%.3f, thresh_from_list=%.3f) -> %s",
                        cfg.best_metric, best_metric_value, best_val_thresh, best_path)
            es_counter = 0
        else:
            es_counter += 1

        # ---- Scheduler Step ----
        if cfg.scheduler == "cosine":
            if isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR):
                scheduler.step()
        elif cfg.scheduler == "plateau_val":
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                monitored = val_metrics['loss'] if cfg.plateau_metric == "val_loss" else val_metrics['f1_best']
                scheduler.step(monitored)

        # optionale Modellhistogramme
        if cfg.log_weight_hist_every_n_epochs > 0 and (epoch % cfg.log_weight_hist_every_n_epochs == 0):
            for name, p in model.named_parameters():
                if p is not None and p.requires_grad and p.data is not None:
                    writer.add_histogram(f"model/weights/{name}", p.detach().cpu().numpy(), epoch)
                if p is not None and p.grad is not None:
                    writer.add_histogram(f"model/grads/{name}", p.grad.detach().cpu().numpy(), epoch)

        # Save pro Epoche (alle)
        ep_path = os.path.join(cfg.out_dir, f"{cfg.model_type}_ws{cfg.window_size}_e{epoch}_l.pth")
        torch.save(model.state_dict(), ep_path)
        writer.add_text("checkpoints/epoch", f"epoch={epoch}, path={ep_path}", epoch)

        # Early Stopping?
        if cfg.early_stopping_patience > 0 and es_counter >= cfg.early_stopping_patience:
            logger.info("⏹️ Early stopping after no improvement for %d epochs.", es_counter)
            break

        writer.flush()

        # --- Proaktives Freigeben von GPU-Speicher nach Validierung ---
        if device.type == "cuda":
            free_cuda_memory(sync=True)

    # ---- Finale Test-Evaluation ----
    if best_path and os.path.isfile(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device), strict=False)
        logger.info("Loaded BEST checkpoint for final test: %s", best_path)

    # Test mit fixem Val-Threshold (cfg.pos_threshold_eval)
    test_metrics_fix = evaluate_windows_with_loss(
        test_loader, model, criterion, device,
        cfg.pos_threshold_eval, cfg.thresh_candidates, log_pr_roc=True,
        use_amp=cfg.use_amp,
    )
    # Test mit Val-optimaler Schwelle (aus Kandidatenliste)
    test_metrics_valbest = evaluate_windows_with_loss(
        test_loader, model, criterion, device,
        best_val_thresh, cfg.thresh_candidates, log_pr_roc=False,
        use_amp=cfg.use_amp,
    )

    # TB: Test
    writer.add_scalar("test/loss@fix", test_metrics_fix['loss'])
    writer.add_scalar("test/f1@fix", test_metrics_fix['f1'])
    writer.add_scalar("test/AP", test_metrics_fix['ap'])
    writer.add_scalar("test/auc", test_metrics_fix['auc'])
    writer.add_scalar("test/f1@val_best_thresh", test_metrics_valbest['f1'])
    writer.add_scalar("test/best_thresh_from_list_val", best_val_thresh)

    if test_metrics_fix["y_true"].size > 0:
        writer.add_pr_curve(
            tag="test/pr_curve@fix",
            labels=torch.from_numpy(test_metrics_fix["y_true"]).int(),
            predictions=torch.from_numpy(test_metrics_fix["y_prob"]).float(),
            global_step=0,
        )
    if test_metrics_fix["curves"] is not None:
        writer.add_scalar("test/roc_auc_curve", test_metrics_fix["curves"]["roc_auc"])
        _add_confusion_matrix_image(
            writer, "test/confusion_matrix@fix",
            evaluate_confusion_matrix(test_metrics_fix, cfg.pos_threshold_eval),
            class_names=("neg","pos"), global_step=0
        )

    # Final GPU memory cleanup after all evaluations
    if device.type == "cuda":
        free_cuda_memory(sync=True)

    logger.info("[TEST] @fix=%.2f | Loss %.4f F1 %.3f AP %.3f AUC %.3f",
                cfg.pos_threshold_eval, test_metrics_fix['loss'], test_metrics_fix['f1'],
                test_metrics_fix['ap'], test_metrics_fix['auc'])
    logger.info("[TEST] @val* thresh_from_list=%.3f | F1 %.3f",
                best_val_thresh, test_metrics_valbest['f1'])

    writer.close()
    return model


# ============================================================
# Test helpers: test_all_models, test_single_video
# ============================================================
 


def test_all_models(
    cfg: Config,
    epoch: Optional[int] = None,
    epoch_max: Optional[int] = None,
):
    """Evaluate saved Optical-Flow checkpoints on the test set and write a CSV summary.

    Args:
        cfg: Config object.
        epoch: If provided, test checkpoints for this single epoch (unless epoch_max given).
        epoch_max: If provided, test checkpoints for all epochs from 1..epoch_max (inclusive).

    Examples:
        test_all_models(cfg)  # default: all saved models for this model_type/window_size
        test_all_models(cfg, epoch=3)  # only epoch 3
        test_all_models(cfg, epoch_max=13)  # epochs 1..13
    """
    device = torch.device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Build dataloader once
    _, _, test_loader, _, _, _ = make_dataloaders(cfg, device)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Build search pattern depending on inputs
    model_pattern = "*{model_type}_ws{window_size}_e{epoch}.pth"
    collected = []
    if epoch_max is not None:
        for e in range(1, int(epoch_max) + 1):
            try:
                pat = model_pattern.format(model_type=cfg.model_type, window_size=cfg.window_size, epoch=e)
            except Exception as exc:
                raise ValueError(f"Could not format internal model_pattern for epoch={e}: {exc}") from exc
            collected.extend(sorted(glob(os.path.join(cfg.out_dir, pat))))
    elif epoch is not None:
        try:
            pat = model_pattern.format(model_type=cfg.model_type, window_size=cfg.window_size, epoch=epoch)
        except Exception as exc:
            raise ValueError(f"Could not format internal model_pattern: {exc}") from exc
        collected.extend(sorted(glob(os.path.join(cfg.out_dir, pat))))
    else:
        collected.extend(sorted(glob(os.path.join(cfg.out_dir, f"*{cfg.model_type}_ws{cfg.window_size}_*.pth"))))

    # Deduplicate while preserving order
    model_paths = []
    for p in collected:
        if p not in model_paths:
            model_paths.append(p)

    best_path = os.path.join(cfg.out_dir, f"best_{cfg.model_type}.pth")
    if os.path.isfile(best_path) and best_path not in model_paths:
        model_paths.insert(0, best_path)

    if not model_paths:
        print("⚠️ Keine Modelle gefunden unter:", cfg.out_dir)
        return

    print(f"📦 {len(model_paths)} gespeicherte Modelle gefunden.\n")

    results = []

    for model_path in model_paths:
        print(f"🔍 Teste Modell: {os.path.basename(model_path)}")

        # Build model according to cfg.model_type
        if "x3d" in cfg.model_type.lower():
            if build_x3d_optflow is None:
                raise RuntimeError("X3D model requested but `optflow_x3d` or its dependencies are not available.")
            model = build_x3d_optflow(device=device, pretrained=False, in_ch=2)
        else:
            model = OpticalFlowNet(in_ch=2).to(device)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state, strict=False)
        model.eval()

        metrics = evaluate_windows_with_loss(
            loader=test_loader,
            model=model,
            criterion=criterion,
            device=device,
            pos_threshold=cfg.pos_threshold_eval,
            thresh_candidates=cfg.thresh_candidates,
            log_pr_roc=False,
            use_amp=cfg.use_amp,
        )

        results.append({
            "model": os.path.basename(model_path),
            "loss": metrics["loss"],
            "acc": metrics["acc"],
            "prec": metrics["prec"],
            "rec": metrics["rec"],
            "f1@fix": metrics["f1"],
            "f1_best": metrics["f1_best"],
            "best_thresh": metrics["best_thresh"],
            "auc": metrics["auc"],
            "ap": metrics["ap"],
        })

        print(f"→ F1={metrics['f1']:.3f} | F1_best={metrics['f1_best']:.3f} | "
              f"Loss={metrics['loss']:.4f} | AUC={metrics['auc']:.3f}")

    df = pd.DataFrame(results)
    out_csv = os.path.join(cfg.out_dir, "test_metrics_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Ergebnisse gespeichert unter: {out_csv}")

    print("\n📊 Übersicht:")
    try:
        print(df.to_markdown(index=False))
    except Exception:
        print(df)


def test_single_video(cfg: Config, model_path: str, video_path: str, label_path: str, batch_size: int = 16, smooth_window: int = 5):
    """
    Testet ein einzelnes Video. Unterstützt zwei Modi:
      - flow-encoded videos (dateiname enthält '_flow_'): erwartet HSV-coded flow frames (H=angle, V=mag)
      - raw ROI videos: berechnet DIS opt. flow on-the-fly (wie Dataset) und verwendet daraus mag/ang

    Gibt preds, times, flags zurück und zeigt einen Plot.
    """
    device = torch.device(cfg.device)
    # Build model per cfg
    if "x3d" in cfg.model_type.lower():
        if build_x3d_optflow is None:
            raise RuntimeError("X3D model requested but `optflow_x3d` or its dependencies are not available.")
        model = build_x3d_optflow(device=device, pretrained=False, in_ch=2)
    else:
        model = OpticalFlowNet(in_ch=2).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Fehler beim Öffnen von {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {total_frames} Frames aus {video_path}")

    flags = elan_flags_from_txt(label_path, total_frames, cfg.fps, cfg.positive_label)
    gt_time = np.arange(total_frames) / cfg.fps

    is_flow_video = ("_flow_" in os.path.basename(video_path))

    resize = (cfg.resize_w, cfg.resize_h)
    window = cfg.window_size
    stride = cfg.test_stride

    # For raw videos create DIS calc
    dis_calc = None
    if not is_flow_video:
        try:
            dis_calc = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        except Exception:
            dis_calc = None

    frame_buffer = deque(maxlen=(window + (0 if is_flow_video else 1)))
    preds, centers = [], []

    # Preload initial frames
    read_idx = 0
    while read_idx < (window if is_flow_video else (window + 1)):
        ret, fr = cap.read()
        if not ret:
            break
        fr = cv2.resize(fr, (resize[0], resize[1])) if (fr.shape[1], fr.shape[0]) != (resize[0], resize[1]) else fr
        frame_buffer.append(fr)
        read_idx += 1

    batch_x, batch_centers = [], []

    with torch.no_grad():
        frame_idx = 0
        while True:
            if is_flow_video:
                # build mags/angs directly from HSV-coded frames in buffer
                if len(frame_buffer) < window:
                    break
                mags = []
                angs = []
                for fr in list(frame_buffer)[:window]:
                    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
                    h = hsv[..., 0].astype(np.float32) * (np.pi / 180.0 * 2)
                    mag = hsv[..., 2].astype(np.float32) / 255.0
                    mag = cv2.resize(mag, resize)
                    h = cv2.resize(h, resize)
                    mags.append(mag)
                    angs.append(h)
            else:
                # raw ROI: need window+1 frames to compute window flows
                if len(frame_buffer) < window + 1:
                    break
                mags = []
                angs = []
                frames = list(frame_buffer)[- (window + 1):]
                for i in range(window):
                    f1 = frames[i]
                    f2 = frames[i + 1]
                    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
                    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
                    if dis_calc is not None:
                        flow = dis_calc.calc(g1, g2, None)
                    else:
                        flow = np.zeros((resize[1], resize[0], 2), dtype=np.float32)
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
                    mag = cv2.resize(mag, resize)
                    ang = cv2.resize(ang, resize)
                    mags.append(mag.astype(np.float32))
                    angs.append(ang.astype(np.float32))

            x_np = np.stack([np.stack(mags, axis=0), np.stack(angs, axis=0)], axis=0).astype(np.float32)
            x = torch.from_numpy(x_np)
            x = normalize_flow_tensor(x, cfg.normalize, cfg.mean, cfg.std)
            batch_x.append(x)
            batch_centers.append(frame_idx + window // 2)

            if len(batch_x) >= batch_size:
                batch = torch.stack(batch_x).to(device)
                out = torch.sigmoid(model(batch)).squeeze(1).cpu().numpy()
                preds.extend(out.tolist())
                centers.extend(batch_centers)
                batch_x.clear()
                batch_centers.clear()

            # advance by stride: read next 'stride' raw frames and append
            ret = True
            for _ in range(stride):
                ret, fr = cap.read()
                frame_idx += 1
                if not ret:
                    break
                fr = cv2.resize(fr, (resize[0], resize[1])) if (fr.shape[1], fr.shape[0]) != (resize[0], resize[1]) else fr
                frame_buffer.append(fr)
            if not ret:
                break

        # remaining
        if batch_x:
            batch = torch.stack(batch_x).to(device)
            out = torch.sigmoid(model(batch)).squeeze(1).cpu().numpy()
            preds.extend(out.tolist())
            centers.extend(batch_centers)

    cap.release()

    preds = np.array(preds)
    centers = np.array(centers)
    t = centers / cfg.fps

    # Smooth predictions over temporal centers if requested (uniform moving average)
    if smooth_window is None:
        smooth_window = 0
    k = int(smooth_window)
    if k > 1:
        kernel = np.ones(k, dtype=np.float32) / float(k)
        try:
            smoothed = np.convolve(preds.astype(np.float32), kernel, mode="same")
        except Exception:
            smoothed = preds.astype(np.float32)
    else:
        smoothed = preds.astype(np.float32)

    # Compute window-level ground-truth (soft) and print metrics
    try:
        from sklearn.metrics import (
            accuracy_score, precision_recall_fscore_support, roc_auc_score,
            confusion_matrix, precision_recall_curve, average_precision_score
        )
        if centers.size == 0:
            print("⚠️ Keine Fenster/Vorhersagen für dieses Video gefunden.")
        else:
            soft_labels = []
            for c in centers.astype(int):
                start = int(max(0, c - (window // 2)))
                end = int(min(total_frames, start + window))
                seg = flags[start:end]
                if seg.size == 0:
                    soft = 0.0
                else:
                    soft = float(seg.mean())
                soft_labels.append(soft)
            soft_labels = np.array(soft_labels, dtype=np.float32)
            gt_bin = (soft_labels >= cfg.pos_threshold_eval).astype(int)
            probs = smoothed.astype(np.float32)

            pr_bin_fix = (probs >= cfg.pos_threshold_eval).astype(int)
            acc = accuracy_score(gt_bin, pr_bin_fix) if gt_bin.size > 0 else float('nan')
            prec, rec, f1, _ = precision_recall_fscore_support(gt_bin, pr_bin_fix, average='binary', zero_division=0)
            precision, recall, _ = precision_recall_curve(gt_bin, probs) if gt_bin.size > 0 else (np.array([]), np.array([]), np.array([]))
            ap = average_precision_score(gt_bin, probs) if gt_bin.size > 0 else float('nan')
            try:
                auc = roc_auc_score(gt_bin, probs) if len(np.unique(gt_bin)) > 1 else float('nan')
            except Exception:
                auc = float('nan')
            cm = confusion_matrix(gt_bin, pr_bin_fix, labels=[0,1]) if gt_bin.size > 0 else np.array([[0,0],[0,0]])
            best_thresh, f1_best = _best_threshold_from_list(gt_bin, probs, cfg.thresh_candidates) if gt_bin.size > 0 else (cfg.pos_threshold_eval, float('nan'))

            print("\n--- Einzelvideo-Metriken ---")
            print(f"Windows: {len(probs)} | Pos-rate(window): {float((soft_labels>=cfg.pos_threshold_eval).mean()):.4f}")
            print(f"Accuracy @fix={cfg.pos_threshold_eval:.2f}: {acc:.4f}")
            print(f"Precision @fix: {prec:.4f} | Recall @fix: {rec:.4f} | F1 @fix: {f1:.4f}")
            print(f"Average Precision (AP): {ap:.4f} | ROC AUC: {auc if not np.isnan(auc) else 'nan'}")
            print(f"Best thresh from candidates: {best_thresh:.3f} | F1 at best thresh: {f1_best:.4f}")
            print("Confusion matrix (rows=true [neg,pos], cols=pred [neg,pos]):")
            print(cm)
            print("--- End metrics ---\n")
    except Exception as e:
        print("⚠️ Metrics could not be computed (sklearn missing or error):", e)

    # plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 5))
        # raw (thin, transparent)
        plt.plot(t, preds, label="raw predictions", color="blue", linewidth=1, alpha=0.4)
        # smoothed (bold)
        if k > 1:
            plt.plot(t, smoothed, label=f"smoothed (k={k})", color="blue", linewidth=2)
        else:
            plt.plot(t, preds, label="predictions", color="blue", linewidth=2)
        plt.axhline(cfg.pos_threshold_eval, color="red", linestyle="--", label="Threshold")
        plt.fill_between(gt_time, 0, flags, color="gray", alpha=0.3, label="Ground Truth (Schluck)")
        plt.title(f"Schluckwahrscheinlichkeit & Ground Truth – {os.path.basename(video_path)}")
        plt.xlabel("Zeit (s)")
        plt.ylabel("Wahrscheinlichkeit / Label")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception:
        pass

    return preds, t, flags