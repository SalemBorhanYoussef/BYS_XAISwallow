# ============================================================
# train_dual_slowflow.py
# Einfaches Dual-Pfad Modell: RGB-Slow (4 Frames) + OptFlow (precomputed)
# Ziel: leichte, schnelle Architektur in Anlehnung an SlowFast-Idee,
#       aber deutlich kompakter als das originale ResNet-basierte SlowFast.
#
# Anforderungen:
#   pip install torch torchvision tensorboard opencv-python scikit-learn h5py
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
import torch.nn.functional as F
import torch.optim as optim
from glob import glob
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

try:
	from precompute_optflow_dis import load_h5_flow
	PRECOMP_AVAILABLE = True
except Exception:
	PRECOMP_AVAILABLE = False


# ============================================================
# Logging
# ============================================================
logger = logging.getLogger("train_dual_slowflow")
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

	# --- Checkpoints / Logging
	out_dir: str = "./models/"
	log_dir_base: str = "runs/dual_slowflow"

	# --- Console/step logging
	print_every_10: int = 10
	print_every_100: int = 100
	log_every_n_steps: int = 20

	# --- Labels (Eval-Binarisierung)
	positive_label: str = "none"
	pos_threshold_eval: float = 0.5

	# --- Flow-Vorverarbeitung (Mag, Angle)
	normalize: bool = False
	mean: Tuple[float, float] = (0.0, 0.0)
	std: Tuple[float, float] = (1.0, 1.0)

	# --- RGB-Slow Pfad
	rgb_normalize: bool = True
	rgb_mean: Tuple[float, float, float] = (0.45, 0.45, 0.45)
	rgb_std: Tuple[float, float, float] = (0.225, 0.225, 0.225)
	slow_rgb_frames: int = 4

	# --- Datensuche
	use_precomputed_flow: bool = True

	# --- Auswahl für bestes Modell
	best_metric: str = "val_f1_best"
	thresh_candidates: Tuple[float, ...] = (0.2, 0.4, 0.5, 0.6, 0.8)

	def to_json(self) -> str:
		return json.dumps(asdict(self), indent=2)


# ============================================================
# Utility-Funktionen
# ============================================================
def set_seed(seed: int) -> None:
	torch.manual_seed(seed)
	np.random.seed(seed)


def list_video_label_pairs(ds_dir: str) -> List[Tuple[str, str]]:
	pairs: List[Tuple[str, str]] = []
	if not os.path.isdir(ds_dir):
		logger.warning("Verzeichnis existiert nicht: %s", ds_dir)
		return pairs
	exts = ("mp4", "avi", "mov", "mkv")
	matches = []
	for ext in exts:
		pattern = os.path.join(ds_dir, "**", f"*roi*.{ext}")
		matches.extend(sorted(glob(pattern, recursive=True)))
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
		if not found and m:
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
	return pairs


def elan_flags_from_txt(txt_path: str, total_frames: int, fps: int, positive_label: str) -> np.ndarray:
	flags = np.zeros(total_frames, dtype=np.int64)
	if not os.path.exists(txt_path):
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
	# x: [2, T, H, W] (Mag, Angle)
	if use_stats:
		m = torch.tensor(mean, dtype=x.dtype, device=x.device).view(2, 1, 1, 1)
		s = torch.tensor(std, dtype=x.dtype, device=x.device).view(2, 1, 1, 1)
		s = torch.clamp(s, min=1e-6)
		return (x - m) / s
	mag = x[0]
	ang = x[1] / math.pi  # scale to ~[0,2]
	def z_norm(t):
		mu = torch.mean(t)
		sd = torch.std(t)
		return (t - mu) / (sd + 1e-6)
	mag_n = z_norm(mag)
	ang_n = z_norm(ang)
	return torch.stack([mag_n, ang_n], dim=0)


# ============================================================
# Dataset – Dual Slow (RGB-4) + Precomputed Flow (Mag+Ang)
# ============================================================
class DualSlowFlowDataset(Dataset):
	def __init__(
		self,
		triples: List[Tuple[str, str, str]],  # (video_path, flow_path, label_path)
		window_size: int,
		stride: int,
		fps: int,
		positive_label: str,
		resize_hw: Tuple[int, int],
		flow_normalize: bool,
		flow_mean: Tuple[float, float],
		flow_std: Tuple[float, float],
		rgb_normalize: bool,
		rgb_mean: Tuple[float, float, float],
		rgb_std: Tuple[float, float, float],
		slow_rgb_frames: int = 4,
		epoch_jitter: bool = True,
	):
		self.window = int(window_size)
		self.stride = max(1, int(stride))
		self.fps = int(fps)
		self.resize_hw = (int(resize_hw[0]), int(resize_hw[1]))
		self.flow_normalize = bool(flow_normalize)
		self.flow_mean, self.flow_std = flow_mean, flow_std
		self.rgb_normalize = bool(rgb_normalize)
		self.rgb_mean, self.rgb_std = rgb_mean, rgb_std
		self.slow_rgb_frames = max(1, int(slow_rgb_frames))
		self.enable_epoch_jitter = bool(epoch_jitter)
		self.index: List[Tuple[str, str, int, np.ndarray, int]] = []  # (video, flow, start, flags, n)
		self.epoch: int = 0

		for vpath, fpath, label_path in triples:
			if not (os.path.exists(vpath) and os.path.exists(fpath) and os.path.exists(label_path)):
				continue
			cap = cv2.VideoCapture(vpath)
			n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
			cap.release()
			if n <= 0:
				continue
			flags = elan_flags_from_txt(label_path, n, self.fps, positive_label)
			max_start = max(1, n - self.window + 1)
			for s in range(0, max_start, self.stride):
				self.index.append((vpath, fpath, s, flags, n))

		self.soft_labels = np.array([float(flags[s:s + self.window].mean()) for (_, _, s, flags, _) in self.index], dtype=np.float32)

	def set_epoch(self, epoch: int):
		self.epoch = int(epoch)

	def _epoch_jitter(self, idx: int) -> int:
		if not self.enable_epoch_jitter or self.stride <= 1:
			return 0
		return ((idx * 1315423911) ^ (self.epoch * 2654435761)) % self.stride

	def epoch_soft_labels(self) -> np.ndarray:
		labs = np.zeros(len(self.index), dtype=np.float32)
		for i, (_, _, start, flags, n) in enumerate(self.index):
			j = self._epoch_jitter(i)
			s = min(start + j, max(0, n - self.window))
			labs[i] = float(flags[s:s + self.window].mean())
		return labs

	def __len__(self) -> int:
		return len(self.index)

	def _load_flow_window(self, fpath: str, s: int) -> torch.Tensor:
		ext = os.path.splitext(fpath)[1].lower()
		flows = None
		try:
			if ext == '.h5':
				if PRECOMP_AVAILABLE:
					f, ds, scale = load_h5_flow(fpath)
					flows = ds[s:s + self.window]
					flows = flows.astype(np.float32) / float(scale)
					f.close()
				else:
					import h5py as _h5
					with _h5.File(fpath, 'r') as _f:
						ds = _f['flow']
						scale = ds.attrs.get('scale', 1.0)
						flows = ds[s:s + self.window].astype(np.float32) / float(scale)
			elif ext == '.npz':
				dat = np.load(fpath)
				flows = dat['flow'][s:s + self.window].astype(np.float32)
			elif ext == '.npy':
				arr = np.load(fpath, mmap_mode='r')
				flows = arr[s:s + self.window].astype(np.float32)
		except Exception:
			flows = None
		if flows is None:
			flows = np.zeros((self.window, self.resize_hw[0], self.resize_hw[1], 2), dtype=np.float32)
		if flows.shape[0] < self.window:
			last = flows[-1] if flows.shape[0] > 0 else np.zeros_like(flows[0])
			pad_cnt = self.window - flows.shape[0]
			pad = np.repeat(last[None, ...], pad_cnt, axis=0)
			flows = np.concatenate([flows, pad], axis=0)
		mags, angs = [], []
		for i in range(self.window):
			flow = flows[i]
			mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
			# ensure spatial size matches target resize (W,H order for cv2)
			if mag.shape[0] != self.resize_hw[0] or mag.shape[1] != self.resize_hw[1]:
				mag = cv2.resize(mag, (self.resize_hw[1], self.resize_hw[0]))
				ang = cv2.resize(ang, (self.resize_hw[1], self.resize_hw[0]))
			mags.append(mag.astype(np.float32))
			angs.append(ang.astype(np.float32))
		x_np = np.stack([np.stack(mags, 0), np.stack(angs, 0)], axis=0).astype(np.float32)
		x = torch.from_numpy(x_np)
		x = normalize_flow_tensor(x, use_stats=self.flow_normalize, mean=self.flow_mean, std=self.flow_std)
		return x

	def _load_rgb_slow(self, vpath: str, s: int) -> torch.Tensor:
		cap = cv2.VideoCapture(vpath)
		try:
			cap.set(cv2.CAP_PROP_POS_FRAMES, s)
		except Exception:
			pass
		frames = []
		last = None
		for i in range(self.window):
			ok, fr = cap.read()
			if not ok or fr is None:
				if last is None:
					fr = np.zeros((self.resize_hw[0], self.resize_hw[1], 3), dtype=np.uint8)
				else:
					fr = last
			fr = cv2.resize(fr, (self.resize_hw[1], self.resize_hw[0]))
			fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
			frames.append(fr)
			last = fr
		cap.release()
		if self.slow_rgb_frames <= 1:
			idxs = [self.window // 2]
		else:
			idxs = np.linspace(0, self.window - 1, self.slow_rgb_frames)
			idxs = np.round(idxs).astype(int).tolist()
		clips = []
		for i in idxs:
			img = frames[i].astype(np.float32) / 255.0
			chw = np.transpose(img, (2, 0, 1))
			clips.append(chw)
		rgb_np = np.stack(clips, axis=1)  # [3,F,H,W]
		if self.rgb_normalize:
			m = np.array(self.rgb_mean, dtype=np.float32).reshape(3, 1, 1, 1)
			sdev = np.array(self.rgb_std, dtype=np.float32).reshape(3, 1, 1, 1)
			rgb_np = (rgb_np - m) / (sdev + 1e-6)
		return torch.from_numpy(rgb_np.astype(np.float32))

	def __getitem__(self, idx: int):
		vpath, fpath, start, flags, n = self.index[idx]
		j = self._epoch_jitter(idx)
		s = min(start + j, max(0, n - self.window))
		x_flow = self._load_flow_window(fpath, s)
		x_rgb = self._load_rgb_slow(vpath, s)
		y = float(flags[s:s + self.window].mean())
		return (x_rgb, x_flow), torch.tensor(y, dtype=torch.float32)


# ============================================================
# Modell – Mini SlowFast ResNet-Lite (ohne Attention, Residual + Fusion)
# ============================================================


# --- Residual Basic Block (3D) ---
class BasicBlock3D(nn.Module):
	def __init__(self, in_ch, out_ch, stride=1):
		super().__init__()
		self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm3d(out_ch)
		self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm3d(out_ch)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = None
		if stride != 1 or in_ch != out_ch:
			self.downsample = nn.Sequential(
				nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm3d(out_ch)
			)

	def forward(self, x):
		identity = x
		out = self.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		if self.downsample is not None:
			identity = self.downsample(x)
		out = out + identity
		return self.relu(out)


# --- 3D Stem ---
class Stem3D(nn.Module):
	def __init__(self, in_ch, out_ch, k_t=1):
		super().__init__()
		self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=(k_t,7,7), stride=(1,2,2), padding=(0,3,3), bias=False)
		self.bn = nn.BatchNorm3d(out_ch)
		self.relu = nn.ReLU(inplace=True)
		self.pool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = self.relu(x)
		return self.pool(x)


# --- Mini SlowFast Fusion (fast -> slow) ---
class FuseFastToSlow(nn.Module):
	def __init__(self, fast_ch: int, slow_ch: int, t_stride: int):
		super().__init__()
		# temporal conv to downsample fast T to slow T, keep spatial dims
		self.conv = nn.Conv3d(fast_ch, slow_ch, kernel_size=(7,1,1), stride=(t_stride,1,1), padding=(3,0,0), bias=False)
		self.bn = nn.BatchNorm3d(slow_ch)
		self.relu = nn.ReLU(inplace=True)
	def forward(self, fast: torch.Tensor) -> torch.Tensor:
		return self.relu(self.bn(self.conv(fast)))


class DualSlowFlowNet(nn.Module):
	"""Mini SlowFast ResNet-Lite (inspired):
	- Slow path for RGB (few frames)
	- Fast path for Optical Flow (many frames)
	- Early fusion (concat) and late fusion (residual add) from fast->slow using temporal convs that downsample T_fast to T_slow.
	Lightweight variant to keep training fast and memory small.
	Returns logits [B,1].
	"""
	def __init__(self, num_classes: int = 1, slow_frames: int = 4, flow_frames: int = 32):
		super().__init__()
		self.slow_frames = slow_frames
		self.flow_frames = flow_frames
		t_ratio = max(1, flow_frames // max(1, slow_frames))

		# Ultra-light stems (previous 32/8 -> now 24/6)
		self.slow_stem = Stem3D(3, 24, k_t=1)
		self.fast_stem = Stem3D(2, 6, k_t=3)

		# Early fusion fast->slow (reduce motion projection channels: 12 instead of 16)
		self.fuse1 = FuseFastToSlow(fast_ch=6, slow_ch=12, t_stride=t_ratio)

		# First residual blocks (concatenated slow: 24 + 12 = 36)
		self.slow_block1 = BasicBlock3D(36, 48, stride=1)   # expand modestly
		self.fast_block1 = BasicBlock3D(6, 12, stride=1)

		# Late fusion: project fast (12ch) to match slow (48ch)
		self.fuse2 = FuseFastToSlow(fast_ch=12, slow_ch=48, t_stride=t_ratio)

		# Refinement blocks (keep widths stable)
		self.slow_block2 = BasicBlock3D(48, 48, stride=1)
		self.fast_block2 = BasicBlock3D(12, 12, stride=1)

		# Global pooling + slim head
		self.pool_slow = nn.AdaptiveAvgPool3d(1)
		self.pool_fast = nn.AdaptiveAvgPool3d(1)
		fusion_dim = 48 + 12  # 60 total channels pooled
		self.fc = nn.Sequential(
			nn.Dropout(0.3),
			nn.Linear(fusion_dim, 32),
			nn.ReLU(inplace=True),
			nn.Linear(32, num_classes),
		)

	def forward(self, rgb: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
		# stems
		slow = self.slow_stem(rgb)   # [B,32, Ts, H/4, W/4]
		fast = self.fast_stem(flow)  # [B, 8, Tf, H/4, W/4]

		# early fusion (concat) – ensure spatial dims match (fast stem pooling may differ)
		fuse1 = self.fuse1(fast)
		if fuse1.shape[-2:] != slow.shape[-2:]:  # spatial H,W adjust
			fuse1 = F.interpolate(fuse1, size=slow.shape[-2:], mode="trilinear", align_corners=False)
		slow = torch.cat([slow, fuse1], dim=1)

		# first residual stage
		slow = self.slow_block1(slow)  # [B,64, Ts, ...]
		fast = self.fast_block1(fast)  # [B,16, Tf, ...]

		# late fusion (add)
		fuse2 = self.fuse2(fast)
		if fuse2.shape[-2:] != slow.shape[-2:]:
			fuse2 = F.interpolate(fuse2, size=slow.shape[-2:], mode="trilinear", align_corners=False)
		slow = slow + fuse2

		# refinement
		slow = self.slow_block2(slow)  # [B,64, Ts, ...]
		fast = self.fast_block2(fast)  # [B,16, Tf, ...]

		# global pooling
		slow_feat = self.pool_slow(slow).flatten(1)
		fast_feat = self.pool_fast(fast).flatten(1)
		fused = torch.cat([slow_feat, fast_feat], dim=1)
		return self.fc(fused)


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


def make_dataloaders(cfg: Config):
	all_pairs = list_video_label_pairs(cfg.dataset_dir)
	all_pairs = [x for x in all_pairs if "lying" in x[0]]
	if not all_pairs:
		raise RuntimeError("Keine (video, txt)-Paare gefunden.")

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
			test_pairs = []
	else:
		n = len(all_pairs)
		n_train = int(0.7 * n)
		n_val = int(0.15 * n)
		train_idx = idx[:n_train]
		val_idx = idx[n_train:n_train + n_val]
		test_idx = idx[n_train + n_val:]
		train_pairs = [all_pairs[i] for i in train_idx]
		val_pairs   = [all_pairs[i] for i in val_idx]
		test_pairs  = [all_pairs[i] for i in test_idx]

	def find_flow_for_video(video_path: str) -> Optional[str]:
		stem = Path(video_path).stem
		vid_dir = Path(video_path).parent
		for ext in ('.h5', '.npz', '.npy'):
			p = vid_dir / (stem + ext)
			if p.exists():
				return str(p)
		for ext in ('.h5', '.npz', '.npy'):
			matches = list(vid_dir.glob(f"*{stem}*{ext}"))
			if matches:
				return str(matches[0])
		return None

	def pairs_to_triples(pairs):
		triples = []
		for v, label in pairs:
			f = find_flow_for_video(v) if cfg.use_precomputed_flow else None
			if not f:
				logger.warning("Kein vorcomputiertes Flow gefunden für %s", v)
				continue
			triples.append((v, f, label))
		return triples

	train_triples = pairs_to_triples(train_pairs)
	val_triples   = pairs_to_triples(val_pairs)
	test_triples  = pairs_to_triples(test_pairs)

	ds_kwargs = dict(
		window_size=cfg.window_size,
		fps=cfg.fps,
		positive_label=cfg.positive_label,
		resize_hw=(cfg.resize_h, cfg.resize_w),
		flow_normalize=cfg.normalize,
		flow_mean=cfg.mean,
		flow_std=cfg.std,
		rgb_normalize=cfg.rgb_normalize,
		rgb_mean=cfg.rgb_mean,
		rgb_std=cfg.rgb_std,
		slow_rgb_frames=cfg.slow_rgb_frames,
	)

	train_ds = DualSlowFlowDataset(train_triples, stride=cfg.train_stride, epoch_jitter=True, **ds_kwargs)
	val_ds   = DualSlowFlowDataset(val_triples,   stride=cfg.val_stride,   epoch_jitter=False, **ds_kwargs)
	test_ds  = DualSlowFlowDataset(test_triples,  stride=cfg.test_stride,  epoch_jitter=False, **ds_kwargs)

	train_ds.set_epoch(0)
	cur_soft = train_ds.epoch_soft_labels()

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

	logger.info("Windows: Train=%d | Val=%d | Test=%d | pos_mean(train)=%.4f | win_pos_rate(train)=%.4f",
				len(train_ds), len(val_ds), len(test_ds), float(cur_soft.mean()),
				float((cur_soft >= cfg.pos_threshold_eval).mean()))

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

	if device.type == "cuda":
		try:
			torch.cuda.empty_cache()
		except Exception:
			pass

	model.eval()
	total_loss, total_n = 0.0, 0
	all_probs, all_soft = [], []

	for (x_rgb, x_flow), y in loader:
		x_rgb = x_rgb.to(device, non_blocking=True)
		x_flow = x_flow.to(device, non_blocking=True)
		y = y.to(device, non_blocking=True).unsqueeze(1)
		if use_amp and device.type == "cuda":
			with torch.cuda.amp.autocast(enabled=True):
				logits = model(x_rgb, x_flow)
				loss = criterion(logits, y)
		else:
			logits = model(x_rgb, x_flow)
			loss = criterion(logits, y)
		probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
		soft = y.squeeze(1).detach().cpu().numpy()
		total_loss += float(loss.item()) * y.size(0)
		total_n += y.size(0)
		all_probs.append(probs)
		all_soft.append(soft)

	if not all_probs:
		return {"loss": 0.0, "acc": 0.0, "prec": 0.0, "rec": 0.0, "f1": 0.0, "auc": float("nan"),
				"ap": float("nan"), "best_thresh": pos_threshold, "f1_best": float("nan"),
				"curves": None, "cm": np.array([[0,0],[0,0]], dtype=int),
				"y_true": np.array([], dtype=np.int32), "y_prob": np.array([], dtype=np.float32)}

	probs = np.concatenate(all_probs)
	soft  = np.concatenate(all_soft)

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
	pr_bin_fix = (probs_sm >= pos_threshold).astype(int)

	acc = accuracy_score(gt_bin, pr_bin_fix)
	prec, rec, f1, _ = precision_recall_fscore_support(gt_bin, pr_bin_fix, average="binary", zero_division=0)
	precision, recall, _ = precision_recall_curve(gt_bin, probs_sm)
	ap = average_precision_score(gt_bin, probs_sm)
	best_thresh, f1_best = _best_threshold_from_list(gt_bin, probs_sm, thresh_candidates)

	try:
		fpr, tpr, _ = roc_curve(gt_bin, probs_sm)
		roc_auc = sk_auc(fpr, tpr) if len(np.unique(gt_bin)) > 1 else float("nan")
		auc = roc_auc_score((soft > 0.0).astype(int), probs_sm)
	except Exception:
		fpr, tpr, roc_auc, auc = np.array([0,1]), np.array([0,1]), float("nan"), float("nan")

	cm = confusion_matrix(gt_bin, pr_bin_fix, labels=[0,1])
	curves = {"precision": precision, "recall": recall, "fpr": fpr, "tpr": tpr, "roc_auc": float(roc_auc)} if log_pr_roc else None

	return {"loss": total_loss / max(1, total_n),
			"acc": float(acc), "prec": float(prec), "rec": float(rec), "f1": float(f1),
			"auc": float(auc), "ap": float(ap),
			"best_thresh": float(best_thresh), "f1_best": float(f1_best),
			"curves": curves, "cm": cm,
			"y_true": gt_bin.astype(np.int32), "y_prob": probs_sm.astype(np.float32)}


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
	img = np.array(img).transpose(2,0,1)
	writer.add_image(tag, img, global_step)


# ============================================================
# Helpers
# ============================================================
def _global_grad_norm(parameters, norm_type: float = 2.0) -> float:
	params = [p for p in parameters if p.grad is not None]
	if not params:
		return 0.0
	device = params[0].grad.device
	total = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in params]), norm_type)
	return total.item()


def _make_run_dirs_and_writer(cfg: Config):
	ts = time.strftime("%Y%m%d-%H%M%S")
	log_dir = os.path.join(cfg.log_dir_base, f"dual_slowflow_{ts}")
	os.makedirs(log_dir, exist_ok=True)
	meta = {"timestamp": ts, "model_type": "dual_slowflow", "cfg": asdict(cfg)}
	with open(os.path.join(log_dir, "RUN_META.json"), "w", encoding="utf-8") as f:
		json.dump(meta, f, indent=2)
	writer = SummaryWriter(log_dir=log_dir)
	writer.add_text("config/json", f"```\n{cfg.to_json()}\n```", 0)
	return writer, log_dir


def free_cuda_memory(sync: bool = True) -> None:
	try:
		import gc
		if torch.cuda.is_available():
			if sync:
				try:
					torch.cuda.synchronize()
				except Exception:
					pass
			try:
				torch.cuda.empty_cache()
			except Exception:
				pass
			try:
				torch.cuda.reset_peak_memory_stats()
			except Exception:
				pass
		gc.collect()
		gc.collect()
	except Exception:
		pass


# ============================================================
# Train
# ============================================================
def train(cfg: Config):
	set_seed(cfg.seed)
	device = torch.device(cfg.device)
	os.makedirs(cfg.out_dir, exist_ok=True)

	writer, log_dir = _make_run_dirs_and_writer(cfg)
	train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = make_dataloaders(cfg)

	model = DualSlowFlowNet(slow_frames=cfg.slow_rgb_frames, flow_frames=cfg.window_size).to(device)

	# Pos-Weight aus aktueller Pos-Rate
	init_soft = train_ds.epoch_soft_labels()
	init_pos_rate = float((init_soft >= cfg.pos_threshold_eval).mean())
	writer.add_scalar("train/init_pos_rate", init_pos_rate, 0)
	def make_criterion(pos_rate: float):
		pos_rate = max(min(float(pos_rate), 1.0 - 1e-6), 1e-6)
		pw = (1.0 - pos_rate) / pos_rate
		return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
	criterion = make_criterion(init_pos_rate)

	optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
	if cfg.scheduler == "cosine":
		scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.1)
	elif cfg.scheduler == "plateau_val":
		mode = "min" if cfg.plateau_metric == "val_loss" else "max"
		scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=cfg.plateau_factor,
														 patience=cfg.plateau_patience, min_lr=cfg.plateau_min_lr)
	else:
		scheduler = None

	# Use new AMP GradScaler API (deprecation-safe)
	scaler = torch.amp.GradScaler(device_type="cuda", enabled=cfg.use_amp and (device.type == "cuda"))

	global_step = 0
	best_metric_value = -1.0
	best_path = None
	best_val_thresh = cfg.pos_threshold_eval
	es_counter = 0

	for epoch in range(1, cfg.epochs + 1):
		# pos_weight refresh per epoch (optional – hier konstant belassen)
		model.train()
		run_loss, run_n = 0.0, 0
		t0 = time.time()
		for bidx, ((x_rgb, x_flow), y) in enumerate(train_loader, 1):
			x_rgb = x_rgb.to(device, non_blocking=True)
			x_flow = x_flow.to(device, non_blocking=True)
			y = y.to(device, non_blocking=True).unsqueeze(1)

			optimizer.zero_grad(set_to_none=True)
			if scaler.is_enabled():
				with torch.amp.autocast("cuda", enabled=True):
					logits = model(x_rgb, x_flow)
					loss = criterion(logits, y)
				scaler.scale(loss).backward()
				grad_norm = _global_grad_norm(model.parameters())
				scaler.step(optimizer)
				scaler.update()
			else:
				logits = model(x_rgb, x_flow)
				loss = criterion(logits, y)
				loss.backward()
				grad_norm = _global_grad_norm(model.parameters())
				optimizer.step()

			run_loss += loss.item() * y.size(0)
			run_n += y.size(0)

			if global_step % cfg.log_every_n_steps == 0:
				for gi, g in enumerate(optimizer.param_groups):
					writer.add_scalar(f"train/lr_group{gi}", g.get("lr", cfg.lr), global_step)
				writer.add_scalar("train/loss_step", loss.item(), global_step)
				writer.add_scalar("train/grad_norm_step", grad_norm, global_step)
				elapsed = time.time() - t0
				try:
					total_batches = len(train_loader)
				except Exception:
					total_batches = bidx
				avg_bt = elapsed / max(1, bidx)
				est_total = avg_bt * total_batches
				writer.add_scalar("time/epoch_elapsed_sec", elapsed, global_step)
				writer.add_scalar("time/epoch_eta_sec", max(0.0, est_total - elapsed), global_step)
				writer.add_scalar("time/epoch_total_est_sec", est_total, global_step)
				writer.flush()

			global_step += 1

			# Console prints every N steps (10/100) similar to baseline trainer
			if cfg.print_every_10 > 0 and (bidx % cfg.print_every_10 == 0):
				avg10 = run_loss / max(1, run_n)
				# elapsed/ETA
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

		tr_loss = run_loss / max(1, run_n)
		writer.add_scalar("train/loss_epoch", tr_loss, epoch)
		writer.add_scalar("train/steps_per_epoch", bidx, epoch)
		writer.add_scalar("time/epoch_seconds", time.time() - t0, epoch)

		# Validation
		val_metrics = evaluate_windows_with_loss(
			val_loader, model, criterion, device,
			cfg.pos_threshold_eval, cfg.thresh_candidates, log_pr_roc=True,
			use_amp=cfg.use_amp,
		)

		writer.add_scalar("val/loss", val_metrics['loss'], epoch)
		writer.add_scalar("val/acc", val_metrics['acc'], epoch)
		writer.add_scalar("val/precision@fix", val_metrics['prec'], epoch)
		writer.add_scalar("val/recall@fix", val_metrics['rec'], epoch)
		writer.add_scalar("val/f1@fix", val_metrics['f1'], epoch)
		writer.add_scalar("val/auc", val_metrics['auc'], epoch)
		writer.add_scalar("val/AP", val_metrics['ap'], epoch)
		writer.add_scalar("val/f1_best_from_list", val_metrics['f1_best'], epoch)
		writer.add_scalar("val/best_thresh_from_list", val_metrics['best_thresh'], epoch)
		if val_metrics.get("y_true") is not None and val_metrics["y_true"].size > 0:
			writer.add_pr_curve(
				tag="val/pr_curve",
				labels=torch.from_numpy(val_metrics["y_true"]).int(),
				predictions=torch.from_numpy(val_metrics["y_prob"]).float(),
				global_step=epoch,
			)
		if val_metrics.get("curves") is not None:
			writer.add_scalar("val/roc_auc_curve", val_metrics["curves"]["roc_auc"], epoch)
		_add_confusion_matrix_image(writer, "val/confusion_matrix@fix", val_metrics.get("cm", np.array([[0,0],[0,0]])),
									class_names=("neg","pos"), global_step=epoch)

		# Best checkpoint & early stopping
		current_metric = (val_metrics["f1_best"] if cfg.best_metric == "val_f1_best" else val_metrics["ap"])
		if current_metric > best_metric_value:
			best_metric_value = current_metric
			best_val_thresh = float(val_metrics["best_thresh"])
			best_path = os.path.join(cfg.out_dir, "best_dual_slowflow.pth")
			torch.save(model.state_dict(), best_path)
			meta = {
				"epoch": epoch,
				"best_metric": cfg.best_metric,
				"best_metric_value": best_metric_value,
				"best_threshold_from_list": best_val_thresh,
				"thresh_candidates": cfg.thresh_candidates,
				"model_type": "dual_slowflow",
				"cfg": asdict(cfg),
				"tb_log_dir": log_dir,
			}
			os.makedirs(cfg.out_dir, exist_ok=True)
			with open(os.path.join(cfg.out_dir, "best_dual_slowflow.json"), "w", encoding="utf-8") as f:
				json.dump(meta, f, indent=2)
			logger.info("✅ New BEST (%s=%.3f, thresh*=%.3f) -> %s", cfg.best_metric, best_metric_value, best_val_thresh, best_path)
			es_counter = 0
		else:
			es_counter += 1

		if cfg.scheduler == "cosine" and isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR):
			scheduler.step()
		elif cfg.scheduler == "plateau_val" and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
			monitored = val_metrics['loss'] if cfg.plateau_metric == "val_loss" else val_metrics['f1_best']
			scheduler.step(monitored)

		# save every epoch
		ep_path = os.path.join(cfg.out_dir, f"dual_slowflow_ws{cfg.window_size}_e{epoch}.pth")
		torch.save(model.state_dict(), ep_path)
		writer.add_text("checkpoints/epoch", f"epoch={epoch}, path={ep_path}", epoch)

		if cfg.early_stopping_patience > 0 and es_counter >= cfg.early_stopping_patience:
			logger.info("⏹️ Early stopping after no improvement for %d epochs.", es_counter)
			break

		writer.flush()
		if device.type == "cuda":
			free_cuda_memory(sync=True)

	# Final Test
	if best_path and os.path.isfile(best_path):
		model.load_state_dict(torch.load(best_path, map_location=device), strict=False)
		logger.info("Loaded BEST checkpoint for final test: %s", best_path)

	test_metrics_fix = evaluate_windows_with_loss(
		test_loader, model, criterion, device,
		cfg.pos_threshold_eval, cfg.thresh_candidates, log_pr_roc=True,
		use_amp=cfg.use_amp,
	)
	test_metrics_valbest = evaluate_windows_with_loss(
		test_loader, model, criterion, device,
		best_val_thresh, cfg.thresh_candidates, log_pr_roc=False,
		use_amp=cfg.use_amp,
	)

	writer.add_scalar("test/loss@fix", test_metrics_fix['loss'])
	writer.add_scalar("test/f1@fix", test_metrics_fix['f1'])
	writer.add_scalar("test/AP", test_metrics_fix['ap'])
	writer.add_scalar("test/auc", test_metrics_fix['auc'])
	writer.add_scalar("test/f1@val_best_thresh", test_metrics_valbest['f1'])
	writer.add_scalar("test/best_thresh_from_list_val", best_val_thresh)

	if test_metrics_fix.get("y_true") is not None and test_metrics_fix["y_true"].size > 0:
		writer.add_pr_curve(
			tag="test/pr_curve@fix",
			labels=torch.from_numpy(test_metrics_fix["y_true"]).int(),
			predictions=torch.from_numpy(test_metrics_fix["y_prob"]).float(),
			global_step=0,
		)
	if test_metrics_fix.get("curves") is not None:
		writer.add_scalar("test/roc_auc_curve", test_metrics_fix["curves"]["roc_auc"])
		_add_confusion_matrix_image(writer, "test/confusion_matrix@fix",
									test_metrics_fix.get("cm", np.array([[0,0],[0,0]])),
									class_names=("neg","pos"), global_step=0)

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
# Test helpers: test_all_models, test_single_video (Dual model)
# ============================================================

@torch.no_grad()
def test_all_models(cfg: Config, epoch: Optional[int] = None, epoch_max: Optional[int] = None):
	"""Load DualSlowFlowNet checkpoints and evaluate on the test set.

	Parameters:
	- cfg: Config object
	- epoch: if given, only include checkpoints for this epoch number (searches _e{epoch} patterns)
	- epoch_max: if given, include checkpoints for epochs 1..epoch_max (searches for each epoch)

	If neither epoch nor epoch_max is provided, all matching checkpoints for the configured
	window size are evaluated. Saves a CSV summary to cfg.out_dir/test_metrics_summary_dual.csv when pandas
	is available.
	"""
	device = torch.device(cfg.device)
	os.makedirs(cfg.out_dir, exist_ok=True)

	# Build dataloader once
	_, _, test_loader, _, _, _ = make_dataloaders(cfg)
	criterion = torch.nn.BCEWithLogitsLoss()

	# gather model paths according to epoch / epoch_max filters
	model_paths = []
	if epoch is not None:
		model_paths = sorted(glob(os.path.join(cfg.out_dir, f"*dual_slowflow_ws{cfg.window_size}_e{int(epoch)}*.pth")))
	elif epoch_max is not None:
		paths = []
		for e in range(1, int(epoch_max) + 1):
			paths.extend(sorted(glob(os.path.join(cfg.out_dir, f"*dual_slowflow_ws{cfg.window_size}_e{e}*.pth"))))
		model_paths = sorted(set(paths))
	else:
		model_paths = sorted(glob(os.path.join(cfg.out_dir, f"*dual_slowflow_ws{cfg.window_size}_*.pth")))

	# include best checkpoint if present
	best_path = os.path.join(cfg.out_dir, "best_dual_slowflow.pth")
	if os.path.isfile(best_path) and best_path not in model_paths:
		model_paths.insert(0, best_path)

	if not model_paths:
		print("⚠️ Keine DualSlowFlow-Modelle gefunden unter:", cfg.out_dir)
		return

	print(f"📦 {len(model_paths)} gespeicherte DualSlowFlow-Modelle gefunden.\n")

	results = []
	for model_path in model_paths:
		print(f"🔍 Teste Modell: {os.path.basename(model_path)}")
		model = DualSlowFlowNet(slow_frames=cfg.slow_rgb_frames, flow_frames=cfg.window_size).to(device)
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

		print(f"→ F1={metrics['f1']:.3f} | F1_best={metrics['f1_best']:.3f} | Loss={metrics['loss']:.4f} | AUC={metrics['auc']:.3f}")

	# optional CSV via pandas
	try:
		import pandas as pd
		df = pd.DataFrame(results)
		out_csv = os.path.join(cfg.out_dir, "test_metrics_summary_dual.csv")
		df.to_csv(out_csv, index=False)
		print(f"\n✅ Ergebnisse gespeichert unter: {out_csv}")
		try:
			print("\n📊 Übersicht:")
			print(df.to_markdown(index=False))
		except Exception:
			print(df)
	except Exception:
		print("(Hinweis) pandas nicht verfügbar – CSV-Export übersprungen.")


@torch.no_grad()
def test_single_video(
	cfg: Config,
	model_path: str,
	video_path: str,
	label_path: str,
	batch_size: int = 8,
	smooth_window: int = 5,
):
	"""Test a single ROI video with the DualSlowFlow model.
	For each window of length cfg.window_size, it computes optical flow on the fly using OpenCV DIS
	and builds the RGB-slow clip from the same window (uniform sampling of cfg.slow_rgb_frames).
	Prints metrics and shows a simple plot if matplotlib is installed.
	Returns (preds, times, flags).
	"""
	from collections import deque

	device = torch.device(cfg.device)
	model = DualSlowFlowNet(slow_frames=cfg.slow_rgb_frames, flow_frames=cfg.window_size).to(device)
	state = torch.load(model_path, map_location=device)
	model.load_state_dict(state, strict=False)
	model.eval()

	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		print(f"❌ Fehler beim Öffnen von {video_path}")
		return [], [], []

	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	print(f"[INFO] {total_frames} Frames aus {video_path}")

	flags = elan_flags_from_txt(label_path, total_frames, cfg.fps, cfg.positive_label)
	gt_time = np.arange(total_frames) / cfg.fps

	resize = (cfg.resize_w, cfg.resize_h)
	window = cfg.window_size
	stride = cfg.test_stride

	# DIS optical flow
	try:
		dis_calc = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
	except Exception:
		dis_calc = None

	frame_buffer = deque(maxlen=window + 1)
	preds, centers = [], []

	# Preload initial frames (window+1 for flow computation across T pairs)
	read_idx = 0
	while read_idx < window + 1:
		ret, fr = cap.read()
		if not ret:
			break
		if (fr.shape[1], fr.shape[0]) != (resize[0], resize[1]):
			fr = cv2.resize(fr, (resize[0], resize[1]))
		frame_buffer.append(fr)
		read_idx += 1

	batch_rgb, batch_flow, batch_centers = [], [], []

	def build_rgb_clip(frames_list: List[np.ndarray]) -> torch.Tensor:
		if cfg.slow_rgb_frames <= 1:
			idxs = [len(frames_list) // 2]
		else:
			idxs = np.linspace(0, len(frames_list) - 1, cfg.slow_rgb_frames)
			idxs = np.round(idxs).astype(int).tolist()
		clips = []
		for i in idxs:
			img = cv2.cvtColor(frames_list[i], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
			chw = np.transpose(img, (2, 0, 1))
			clips.append(chw)
		rgb_np = np.stack(clips, axis=1).astype(np.float32)  # [3,F,H,W]
		if cfg.rgb_normalize:
			m = np.array(cfg.rgb_mean, dtype=np.float32).reshape(3, 1, 1, 1)
			sdev = np.array(cfg.rgb_std, dtype=np.float32).reshape(3, 1, 1, 1)
			rgb_np = (rgb_np - m) / (sdev + 1e-6)
		return torch.from_numpy(rgb_np)

	def build_flow_clip(frames_list: List[np.ndarray]) -> torch.Tensor:
		mags, angs = [] , []
		for i in range(window):
			f1 = frames_list[i]
			f2 = frames_list[i + 1]
			g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
			g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
			if dis_calc is not None:
				flow = dis_calc.calc(g1, g2, None)
			else:
				flow = np.zeros((resize[1], resize[0], 2), dtype=np.float32)
			mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
			mags.append(mag.astype(np.float32))
			angs.append(ang.astype(np.float32))
		x_np = np.stack([np.stack(mags, 0), np.stack(angs, 0)], axis=0).astype(np.float32)
		x = torch.from_numpy(x_np)
		x = normalize_flow_tensor(x, cfg.normalize, cfg.mean, cfg.std)
		return x

	with torch.no_grad():
		frame_idx = 0
		while True:
			if len(frame_buffer) < window + 1:
				break
			frames_list = list(frame_buffer)[- (window + 1):]
			x_rgb = build_rgb_clip(frames_list[:-1])  # take the first window frames for RGB
			x_flow = build_flow_clip(frames_list)
			batch_rgb.append(x_rgb)
			batch_flow.append(x_flow)
			batch_centers.append(frame_idx + window // 2)

			if len(batch_rgb) >= batch_size:
				rgb_batch = torch.stack(batch_rgb).to(device)
				flow_batch = torch.stack(batch_flow).to(device)
				out = torch.sigmoid(model(rgb_batch, flow_batch)).squeeze(1).cpu().numpy()
				preds.extend(out.tolist())
				centers.extend(batch_centers)
				batch_rgb.clear()
				batch_flow.clear()
				batch_centers.clear()

			# advance by stride
			ret = True
			for _ in range(stride):
				ret, fr = cap.read()
				frame_idx += 1
				if not ret:
					break
				if (fr.shape[1], fr.shape[0]) != (resize[0], resize[1]):
					fr = cv2.resize(fr, (resize[0], resize[1]))
				frame_buffer.append(fr)
			if not ret:
				break

		# flush remaining
		if batch_rgb:
			rgb_batch = torch.stack(batch_rgb).to(device)
			flow_batch = torch.stack(batch_flow).to(device)
			out = torch.sigmoid(model(rgb_batch, flow_batch)).squeeze(1).cpu().numpy()
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
				soft = float(seg.mean()) if seg.size > 0 else 0.0
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

			print("\n--- Einzelvideo-Metriken ---")
			print(f"Windows: {len(probs)} | Pos-rate(window): {float((soft_labels>=cfg.pos_threshold_eval).mean()):.4f}")
			print(f"Accuracy @fix={cfg.pos_threshold_eval:.2f}: {acc:.4f}")
			print(f"Precision @fix: {prec:.4f} | Recall @fix: {rec:.4f} | F1 @fix: {f1:.4f}")
			print(f"Average Precision (AP): {ap:.4f} | ROC AUC: {auc if not np.isnan(auc) else 'nan'}")
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