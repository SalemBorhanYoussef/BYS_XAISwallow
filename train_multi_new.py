# train_multi_new.py
# Vollständiges Training mit Umschalter: "slowfast" | "slowraft"
# - slowfast: pytorchvideo SlowFast R50
# - slowraft: ResNet18 (Slow) + RAFT (Fast) nach SlowFast-Prinzip (improved)
#
# Anforderungen:
#   pip install torch torchvision pytorchvideo tensorboard opencv-python scikit-learn matplotlib pillow

import os
import math
import time
import json
import logging
import argparse
from dataclasses import dataclass, asdict, replace
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from glob import glob
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

# Improved SlowRAFT components
from slowraft_improved import SlowRAFTModelImproved

logger = logging.getLogger("train_multi_new")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)


@dataclass
class Config:
    # Daten
    dataset_dir: str = "./Dataset_all/Dataset_l"
    dataset_val_dir: Optional[str] = None
    dataset_test_dir: str = "./Dataset_all/Dataset_l_test"
    val_split_frac: float = 0.15

    # Video / Sampling
    fps: int = 32
    resize_h: int = 224
    resize_w: int = 224
    window_size: int = 32
    train_stride: int = 4
    val_stride: int = 8
    test_stride: int = 8

    # SlowFast Zeitsampling
    t_fast: int = 32
    alpha: int = 4

    # Optimierung
    batch_size: int = 8
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 0.0

    scheduler: str = "cosine"  # none|cosine|plateau_val
    plateau_metric: str = "val_loss"
    plateau_patience: int = 3
    plateau_factor: float = 0.5
    plateau_min_lr: float = 1e-7

    early_stopping_patience: int = 7

    # System
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0
    pin_memory: bool = torch.cuda.is_available()
    use_amp: bool = True

    # Checkpoints / Out
    out_dir: str = "./models/l/"
    ckpt: Optional[str] = None

    # Labels
    positive_label: str = "none"
    pos_threshold_eval: float = 0.5

    # Normalisierung
    normalize: bool = True
    mean: Tuple[float, float, float] = (0.45, 0.45, 0.45)
    std: Tuple[float, float, float] = (0.225, 0.225, 0.225)

    # Tricks
    epoch_jitter: bool = True

    # TensorBoard
    log_dir_base: str = "runs/train_multi"
    log_every_n_steps: int = 20
    log_weight_hist_every_n_epochs: int = 5
    print_every_10: int = 10
    print_every_100: int = 100

    # Freeze / Finetune
    freeze_policy: str = "partial"  # linear|partial|none
    pretrained_backbone: bool = True
    bn_eval_freeze_affine: bool = True

    backbone_lr: float = 1e-5
    head_lr: float = 1e-4

    # Modellwahl
    model_type: str = "slowraft"  # slowfast|slowraft

    # SlowRAFT Optionen
    raft_weights: Optional[str] = None
    raft_delta: int = 2
    raft_downscale: float = 1.0
    raft_pair_stride: int = 2
    raft_max_pairs: int = 32

    # Auswahl Kriterium für "best"
    best_metric: str = "val_f1_best"  # val_f1_best|val_ap

    # Diskrete Threshold-Kandidaten
    thresh_candidates: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def list_video_label_pairs(ds_dir: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if not os.path.isdir(ds_dir):
        return pairs
    for v in sorted(glob(os.path.join(ds_dir, "*_roi.mp4"))):
        base = os.path.splitext(os.path.basename(v))[0]
        t = os.path.join(ds_dir, base.replace("_roi", "") + ".txt")
        if os.path.exists(t):
            pairs.append((v, t))
        else:
            logger.warning("Missing label txt for %s", v)
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
                    flags[si : ei + 1] = 1
    return flags


def sample_indices(num_src: int, num_out: int) -> List[int]:
    if num_src <= 0:
        return [0] * num_out
    if num_out <= 1:
        return [0]
    if num_src == num_out:
        return list(range(num_src))
    idx = np.linspace(0, max(0, num_src - 1), num_out)
    return np.round(idx).astype(int).tolist()


def build_tfm_cpu(normalize: bool, size_hw: Tuple[int, int], mean, std):
    from torchvision import transforms as T
    tfms = [T.ToPILImage(), T.Resize(size_hw), T.ToTensor()]
    if normalize:
        tfms.append(T.Normalize(mean=mean, std=std))
    return T.Compose(tfms)


def make_slowfast_tensors_cpu(frames_rgb, t_fast: int, t_slow: int, tfm):
    idx_fast = sample_indices(len(frames_rgb), t_fast)
    fast_clip = [tfm(frames_rgb[i]) for i in idx_fast]
    fast = torch.stack(fast_clip, dim=1)
    idx_slow = sample_indices(len(fast_clip), t_slow)
    slow_clip = [fast_clip[i] for i in idx_slow]
    slow = torch.stack(slow_clip, dim=1)
    return slow, fast


class SlowFastDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        window_size: int,
        stride: int,
        fps: int,
        t_fast: int,
        alpha: int,
        positive_label: str,
        resize_hw: Tuple[int, int],
        normalize: bool,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        epoch_jitter: bool = True,
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

        self.index: List[Tuple[str, int, np.ndarray, int]] = []
        self.epoch: int = 0

        for vpath, tpath in pairs:
            cap = cv2.VideoCapture(vpath)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            cap.release()
            if n <= 0:
                logger.warning("Empty video: %s", vpath)
                continue
            flags = elan_flags_from_txt(tpath, n, self.fps, positive_label)
            max_start = max(1, n - self.window + 1)
            for s in range(0, max_start, self.stride):
                self.index.append((vpath, s, flags, n))

        self.tfm_cpu = build_tfm_cpu(self.normalize, self.resize_hw, self.mean, self.std)
        self.soft_labels = np.array(
            [float(flags[s : s + self.window].mean()) for (_, s, flags, _) in self.index], dtype=np.float32
        )

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _epoch_jitter(self, idx: int) -> int:
        if not self.enable_epoch_jitter or self.stride <= 1:
            return 0
        return ((idx * 1315423911) ^ (self.epoch * 2654435761)) % self.stride

    def epoch_soft_labels(self) -> np.ndarray:
        labs = np.zeros(len(self.index), dtype=np.float32)
        for i, (vpath, start, flags, n) in enumerate(self.index):
            j = self._epoch_jitter(i)
            s = min(start + j, max(0, n - self.window))
            labs[i] = float(flags[s : s + self.window].mean())
        return labs

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        vpath, start, flags, n = self.index[idx]
        j = self._epoch_jitter(idx)
        s = min(start + j, max(0, n - self.window))

        cap = cv2.VideoCapture(vpath)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        except Exception:
            pass

        frames = []
        last = None
        for _ in range(self.window):
            ok, frame = cap.read()
            if not ok or frame is None:
                frames.append(last if last is not None else np.zeros((self.resize_hw[0], self.resize_hw[1], 3), np.uint8))
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last = frame
            frames.append(frame)
        cap.release()

        slow, fast = make_slowfast_tensors_cpu(frames, self.t_fast, self.t_slow, self.tfm_cpu)
        soft = float(flags[s : s + self.window].mean())
        return [slow, fast], torch.tensor(soft, dtype=torch.float32)


# --- A) Original SlowFast Hub ---

def build_model_slowfast(device: torch.device, pretrained: bool = True) -> nn.Module:
    try:
        from pytorchvideo.models.hub import slowfast_r50
    except Exception as e:
        raise RuntimeError("Bitte 'pytorchvideo' installieren: pip install pytorchvideo") from e
    m = slowfast_r50(pretrained=pretrained)
    # Head auf 1 Logit
    m.blocks[-1].proj = nn.Linear(m.blocks[-1].proj.in_features, 1)
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


# --- B) SlowRAFT (Improved) ---

def build_model_slowraft(device: torch.device, cfg: Config) -> nn.Module:
    try:
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        try:
            weights = Raft_Small_Weights.DEFAULT
        except Exception:
            weights = None
        raft = raft_small(weights=weights, progress=True).to(device).eval()
        logger.info("Loaded torchvision RAFT-Small with weights=%s", str(weights))
    except Exception:
        from torchvision.models.optical_flow import raft_small
        raft = raft_small(pretrained=True).to(device).eval()
        logger.info("Loaded torchvision RAFT-Small with pretrained=True (legacy API)")

    raft = raft.to(torch.float32)  # RAFT zwingend Float32

    if getattr(cfg, "raft_weights", None):
        state = torch.load(cfg.raft_weights, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        raft.load_state_dict(state, strict=False)
        logger.info("Loaded custom RAFT weights (%s).", cfg.raft_weights)

    m = SlowRAFTModelImproved(
        raft_model=raft,
        pretrained_resnet=cfg.pretrained_backbone,
        delta=cfg.raft_delta,
        pair_stride=cfg.raft_pair_stride,
        max_pairs=cfg.raft_max_pairs,
        raft_scale=cfg.raft_downscale,
        beta=0.25,
        use_batched_raft=True,
    )
    return m.to(device)


def apply_freeze_policy_slowraft(model: nn.Module, policy: str, bn_freeze_affine: bool = True):
    def set_req_grad(m, flag: bool):
        for p in m.parameters():
            p.requires_grad = flag

    def bn_eval_freeze_affine(m, freeze_affine=True):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            if freeze_affine and getattr(m, "affine", False):
                if m.weight is not None:
                    m.weight.requires_grad = False
                if m.bias is not None:
                    m.bias.requires_grad = False
        for c in m.children():
            bn_eval_freeze_affine(c, freeze_affine)

    resnet = model.slow
    raft = model.fast_raft.raft
    flow_h = model.fast_head
    head = model.head

    if policy == "linear":
        set_req_grad(resnet, False)
        set_req_grad(raft, False)
        set_req_grad(flow_h, False)
        set_req_grad(head, True)
    elif policy == "partial":
        # For improved backbone, enable later layers
        for name, m in model.slow.named_children():
            if name in ("layer3", "layer4", "temporal_agg"):
                set_req_grad(m, True)
            else:
                set_req_grad(m, False)
        set_req_grad(raft, False)
        set_req_grad(flow_h, True)
        set_req_grad(head, True)
    elif policy == "none":
        set_req_grad(model, True)
    else:
        raise ValueError(f"Unbekannte freeze_policy: {policy}")

    if bn_freeze_affine:
        bn_eval_freeze_affine(model, freeze_affine=True)


def _build_weighted_loader(ds: SlowFastDataset, cfg: Config, cur_soft: np.ndarray) -> DataLoader:
    pos_mask = cur_soft >= cfg.pos_threshold_eval
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
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )


def _split_train_val_pairs(train_pairs: List[Tuple[str, str]], frac: float, seed: int):
    if not 0.0 < frac < 1.0:
        return train_pairs, []
    rng = np.random.RandomState(seed)
    idx = np.arange(len(train_pairs))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(train_pairs) * frac)))
    val_idx = set(idx[:n_val].tolist())
    train_sub = [p for i, p in enumerate(train_pairs) if i not in val_idx]
    val_sub = [p for i, p in enumerate(train_pairs) if i in val_idx]
    return train_sub, val_sub


def make_dataloaders(cfg: Config, device: torch.device):
    all_train_pairs = list_video_label_pairs(cfg.dataset_dir)
    if not all_train_pairs:
        raise RuntimeError("Keine (video, txt) Paare im Trainingsordner gefunden.")

    if cfg.dataset_val_dir:
        val_pairs = list_video_label_pairs(cfg.dataset_val_dir)
        train_pairs = all_train_pairs
    else:
        train_pairs, val_pairs = _split_train_val_pairs(all_train_pairs, cfg.val_split_frac, cfg.seed)

    test_pairs = list_video_label_pairs(cfg.dataset_test_dir)

    logger.info(
        "Found train videos=%d | val videos=%d | test videos=%d",
        len(train_pairs),
        len(val_pairs),
        len(test_pairs),
    )

    train_ds = SlowFastDataset(
        train_pairs,
        cfg.window_size,
        cfg.train_stride,
        cfg.fps,
        cfg.t_fast,
        cfg.alpha,
        cfg.positive_label,
        (cfg.resize_h, cfg.resize_w),
        cfg.normalize,
        cfg.mean,
        cfg.std,
        epoch_jitter=cfg.epoch_jitter,
    )
    val_ds = SlowFastDataset(
        val_pairs,
        cfg.window_size,
        cfg.val_stride,
        cfg.fps,
        cfg.t_fast,
        cfg.alpha,
        cfg.positive_label,
        (cfg.resize_h, cfg.resize_w),
        cfg.normalize,
        cfg.mean,
        cfg.std,
        epoch_jitter=False,
    )
    test_ds = SlowFastDataset(
        test_pairs,
        cfg.window_size,
        cfg.test_stride,
        cfg.fps,
        cfg.t_fast,
        cfg.alpha,
        cfg.positive_label,
        (cfg.resize_h, cfg.resize_w),
        cfg.normalize,
        cfg.mean,
        cfg.std,
        epoch_jitter=False,
    )

    train_ds.set_epoch(0)
    cur_soft = train_ds.epoch_soft_labels() if cfg.epoch_jitter else train_ds.soft_labels

    train_loader = _build_weighted_loader(train_ds, cfg, cur_soft)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    logger.info(
        "Windows: Train=%d | Val=%d | Test=%d | pos_mean(train)=%.4f | win_pos_rate(train)=%.4f",
        len(train_ds),
        len(val_ds),
        len(test_ds),
        float(cur_soft.mean()),
        float((cur_soft >= cfg.pos_threshold_eval).mean()),
    )
    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds


@torch.no_grad()

def evaluate_windows_with_loss(
    loader: DataLoader,
    model: nn.Module,
    criterion,
    device: torch.device,
    pos_threshold: float,
    thresh_candidates: Tuple[float, ...],
    log_pr_roc: bool = True,
):
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        roc_auc_score,
        confusion_matrix,
        precision_recall_curve,
        roc_curve,
        auc as sk_auc,
        average_precision_score,
    )
    model.eval()
    total_loss, total_n = 0.0, 0
    all_probs, all_soft = [], []

    for (slow, fast), y in loader:
        slow, fast = slow.to(device, non_blocking=True), fast.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).unsqueeze(1)
        logits = model([slow, fast])
        loss = criterion(logits, y)
        probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
        soft = y.squeeze(1).detach().cpu().numpy()

        total_loss += float(loss.item()) * y.size(0)
        total_n += y.size(0)
        all_probs.append(probs)
        all_soft.append(soft)

    if not all_probs:
        return {
            "loss": 0.0,
            "acc": 0.0,
            "prec": 0.0,
            "rec": 0.0,
            "f1": 0.0,
            "auc": float("nan"),
            "ap": float("nan"),
            "best_thresh": pos_threshold,
            "f1_best": float("nan"),
            "curves": None,
            "cm": np.array([[0, 0], [0, 0]], dtype=int),
            "y_true": np.array([], dtype=np.int32),
            "y_prob": np.array([], dtype=np.float32),
        }

    probs = np.concatenate(all_probs)
    soft = np.concatenate(all_soft)
    gt_bin = (soft >= pos_threshold).astype(int)
    pr_bin_fix = (probs >= pos_threshold).astype(int)

    acc = accuracy_score(gt_bin, pr_bin_fix)
    prec, rec, f1, _ = precision_recall_fscore_support(gt_bin, pr_bin_fix, average="binary", zero_division=0)

    precision, recall, pr_thresholds = precision_recall_curve(gt_bin, probs)
    ap = average_precision_score(gt_bin, probs)

    best_thresh, f1_best = _best_threshold_from_list(gt_bin, probs, thresh_candidates)

    try:
        fpr, tpr, _ = roc_curve(gt_bin, probs)
        roc_auc = sk_auc(fpr, tpr) if len(np.unique(gt_bin)) > 1 else float("nan")
        auc = roc_auc_score((soft > 0.0).astype(int), probs)
    except Exception:
        fpr, tpr, roc_auc, auc = np.array([0, 1]), np.array([0, 1]), float("nan"), float("nan")

    cm = confusion_matrix(gt_bin, pr_bin_fix, labels=[0, 1])

    curves = None
    if log_pr_roc:
        curves = {"precision": precision, "recall": recall, "fpr": fpr, "tpr": tpr, "roc_auc": float(roc_auc)}

    return {
        "loss": total_loss / max(1, total_n),
        "acc": float(acc),
        "prec": float(prec),
        "rec": float(rec),
        "f1": float(f1),
        "auc": float(auc),
        "ap": float(ap),
        "best_thresh": float(best_thresh),
        "f1_best": float(f1_best),
        "curves": curves,
        "cm": cm,
        "y_true": gt_bin.astype(np.int32),
        "y_prob": probs.astype(np.float32),
    }


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


def _add_confusion_matrix_image(writer: SummaryWriter, tag: str, cm: np.ndarray, class_names=("neg", "pos"), global_step: int = 0):
    import matplotlib.pyplot as plt
    import io

    fig = plt.figure(figsize=(3, 3))
    ax = plt.gca()
    ax.imshow(cm, interpolation="nearest")
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
    plt.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    import PIL.Image as Image

    img = Image.open(buf)
    img = np.array(img).transpose(2, 0, 1)
    writer.add_image(tag, img, global_step)


def evaluate_confusion_matrix(metrics_dict, threshold_used):
    if "cm" in metrics_dict:
        return metrics_dict["cm"]
    return np.array([[0, 0], [0, 0]], dtype=int)


def _global_grad_norm(parameters, norm_type: float = 2.0) -> float:
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0
    device = params[0].grad.device
    total = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in params]), norm_type)
    return total.item()


def init_head_bias_slowfastlike(model: nn.Module, prior: float):
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
    ts = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(cfg.log_dir_base, f"{cfg.model_type}_{ts}")
    os.makedirs(log_dir, exist_ok=True)
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


def train(cfg: Config):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)

    writer, log_dir = _make_run_dirs_and_writer(cfg)

    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = make_dataloaders(cfg, device)

    if cfg.model_type.lower() == "slowfast":
        model = build_model_slowfast(device, pretrained=cfg.pretrained_backbone)
        if cfg.ckpt and os.path.isfile(cfg.ckpt):
            state = torch.load(cfg.ckpt, map_location=device)
            model.load_state_dict(state, strict=False)
            logger.info("Loaded weights from %s", cfg.ckpt)
        apply_freeze_policy_slowfast(model, cfg.freeze_policy, bn_freeze_affine=cfg.bn_eval_freeze_affine)
        head_modules = [model.blocks[-1].proj]
    elif cfg.model_type.lower() == "slowraft":
        model = build_model_slowraft(device, cfg)
        if cfg.ckpt and os.path.isfile(cfg.ckpt):
            state = torch.load(cfg.ckpt, map_location=device)
            model.load_state_dict(state, strict=False)
            logger.info("Loaded weights from %s", cfg.ckpt)
        apply_freeze_policy_slowraft(model, cfg.freeze_policy, bn_freeze_affine=cfg.bn_eval_freeze_affine)
        head_modules = [model.head]
    else:
        raise ValueError("Config.model_type muss 'slowfast' oder 'slowraft' sein")

    init_soft = train_ds.epoch_soft_labels() if cfg.epoch_jitter else train_ds.soft_labels
    init_pos_rate = float((init_soft >= cfg.pos_threshold_eval).mean())
    init_head_bias_slowfastlike(model, max(init_pos_rate, 1e-6))
    writer.add_scalar("train/init_pos_rate", init_pos_rate, 0)

    def make_criterion(pos_rate: float):
        pos_rate = max(min(float(pos_rate), 1.0 - 1e-6), 1e-6)
        pw = (1.0 - pos_rate) / pos_rate
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))

    criterion = make_criterion(init_pos_rate)

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
        # mode used only for step metric selection
    else:
        scheduler = None

    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp and (device.type == "cuda"))

    global_step = 0
    best_metric_value = -1.0
    best_path = None
    best_val_thresh = cfg.pos_threshold_eval
    es_counter = 0

    for epoch in range(1, cfg.epochs + 1):
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
        for bidx, ((slow, fast), y) in enumerate(train_loader, 1):
            slow, fast = slow.to(device, non_blocking=True), fast.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                with torch.amp.autocast("cuda", enabled=True):
                    logits = model([slow, fast])
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                grad_norm = _global_grad_norm(model.parameters())
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model([slow, fast])
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

        val_metrics = evaluate_windows_with_loss(
            val_loader, model, criterion, device, cfg.pos_threshold_eval, cfg.thresh_candidates, log_pr_roc=True
        )

        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("val/acc", val_metrics["acc"], epoch)
        writer.add_scalar("val/precision@fix", val_metrics["prec"], epoch)
        writer.add_scalar("val/recall@fix", val_metrics["rec"], epoch)
        writer.add_scalar("val/f1@fix", val_metrics["f1"], epoch)
        writer.add_scalar("val/auc", val_metrics["auc"], epoch)
        writer.add_scalar("val/AP", val_metrics["ap"], epoch)
        writer.add_scalar("val/f1_best_from_list", val_metrics["f1_best"], epoch)
        writer.add_scalar("val/best_thresh_from_list", val_metrics["best_thresh"], epoch)

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
            writer,
            "val/confusion_matrix@fix",
            evaluate_confusion_matrix(val_metrics, cfg.pos_threshold_eval),
            class_names=("neg", "pos"),
            global_step=epoch,
        )

        current_metric = val_metrics["f1_best"] if cfg.best_metric == "val_f1_best" else val_metrics["ap"]
        improved = current_metric > best_metric_value
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
            writer.add_text(
                "checkpoints/best",
                f"epoch={epoch}, {cfg.best_metric}={best_metric_value:.4f}, thresh*={best_val_thresh:.3f}, path={best_path}",
                epoch,
            )
            logger.info(
                "✅ New BEST (%s=%.3f, thresh_from_list=%.3f) -> %s",
                cfg.best_metric,
                best_metric_value,
                best_val_thresh,
                best_path,
            )
            es_counter = 0
        else:
            es_counter += 1

        if cfg.scheduler == "cosine":
            if isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR):
                scheduler.step()
        elif cfg.scheduler == "plateau_val":
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                monitored = val_metrics["loss"] if cfg.plateau_metric == "val_loss" else val_metrics["f1_best"]
                scheduler.step(monitored)

        if cfg.log_weight_hist_every_n_epochs > 0 and (epoch % cfg.log_weight_hist_every_n_epochs == 0):
            for name, p in model.named_parameters():
                if p is not None and p.requires_grad and p.data is not None:
                    writer.add_histogram(f"model/weights/{name}", p.detach().cpu().numpy(), epoch)
                if p is not None and p.grad is not None:
                    writer.add_histogram(f"model/grads/{name}", p.grad.detach().cpu().numpy(), epoch)

        ep_path = os.path.join(cfg.out_dir, f"{cfg.model_type}_ws{cfg.window_size}_e{epoch}.pth")
        torch.save(model.state_dict(), ep_path)
        writer.add_text("checkpoints/epoch", f"epoch={epoch}, path={ep_path}", epoch)

        if cfg.early_stopping_patience > 0 and es_counter >= cfg.early_stopping_patience:
            logger.info("⏹️ Early stopping after no improvement for %d epochs.", es_counter)
            break

        writer.flush()

    if best_path and os.path.isfile(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device), strict=False)
        logger.info("Loaded BEST checkpoint for final test: %s", best_path)

    test_metrics_fix = evaluate_windows_with_loss(
        test_loader, model, criterion, device, cfg.pos_threshold_eval, cfg.thresh_candidates, log_pr_roc=True
    )
    test_metrics_valbest = evaluate_windows_with_loss(
        test_loader, model, criterion, device, best_val_thresh, cfg.thresh_candidates, log_pr_roc=False
    )

    writer.add_scalar("test/loss@fix", test_metrics_fix["loss"])
    writer.add_scalar("test/f1@fix", test_metrics_fix["f1"])
    writer.add_scalar("test/AP", test_metrics_fix["ap"])
    writer.add_scalar("test/auc", test_metrics_fix["auc"])
    writer.add_scalar("test/f1@val_best_thresh", test_metrics_valbest["f1"])
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
            writer,
            "test/confusion_matrix@fix",
            evaluate_confusion_matrix(test_metrics_fix, cfg.pos_threshold_eval),
            class_names=("neg", "pos"),
            global_step=0,
        )

    logger.info(
        "[TEST] @fix=%.2f | Loss %.4f F1 %.3f AP %.3f AUC %.3f",
        cfg.pos_threshold_eval,
        test_metrics_fix["loss"],
        test_metrics_fix["f1"],
        test_metrics_fix["ap"],
        test_metrics_fix["auc"],
    )
    logger.info(
        "[TEST] @val* thresh_from_list=%.3f | F1 %.3f",
        best_val_thresh,
        test_metrics_valbest["f1"],
    )

    writer.close()
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SlowFast/SlowRAFT")
    p.add_argument("--config-json", type=str, default=None, help="Override Config via JSON string")
    p.add_argument("--config", type=str, default=None, help="Path to JSON file with Config overrides")
    return p.parse_args()


def apply_overrides(cfg: Config, overrides: dict) -> Config:
    for k, v in overrides.items():
        if hasattr(cfg, k):
            cfg = replace(cfg, **{k: v})
    return cfg


if __name__ == "__main__":
    args = parse_args()
    cfg = Config()
    try:
        if args.config:
            with open(args.config, "r", encoding="utf-8") as f:
                overrides = json.load(f)
            cfg = apply_overrides(cfg, overrides)
        if args.config_json:
            overrides = json.loads(args.config_json)
            cfg = apply_overrides(cfg, overrides)
    except Exception as ex:
        logger.warning("Config overrides failed: %s", ex)

    logger.info("Config:\n%s", cfg.to_json())
    train(cfg)
