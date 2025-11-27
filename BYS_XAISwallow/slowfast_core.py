# slowfast_core.py
# -----------------------------------------------------------------------------
# Minimaler, aber vollständiger Kern für Live-Inferenz mit SlowFast:
# - Config
# - load_model(cfg, device)
# - make_slowfast_tensors_gpu(...)
# - make_slowfast_tensors_cpu(...)
#
# Optional: SimpleSlowFast (2D-Encoder je Pfad), SimpleSlowFast3D,
#           oder Hub-Modell (pytorchvideo.slowfast_r50), je nach cfg.model.
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# torchvision wird für 2D-Backbones benötigt (simple-Variante)
from torchvision import transforms

# pytorchvideo (nur, wenn cfg.model == "hub" und Lib vorhanden)
try:
    from pytorchvideo.models.hub import slowfast_r50 as hub_slowfast_r50
except Exception:
    hub_slowfast_r50 = None


# =============================================================================
# Konfiguration
# =============================================================================

@dataclass
class Config:
    # Video & Sampling
    fps: int = 32
    resize_h: int = 244
    resize_w: int = 244
    window_size: int = 32

    # SlowFast-Zeitachsen
    t_fast: int = 32
    alpha: int = 4  # t_slow = t_fast // alpha

    # Inferenz
    threshold: float = 0.5
    # Für Konsistenz mit Trainings-Skripten (eval cutoff)
    pos_threshold_eval: float = 0.5
    use_amp: bool = True
    normalize: bool = False
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Modellwahl
    model: str = "hub"             # Standard: PyTorchVideo SlowFast R50
    slow_backbone: str = "resnet18"
    fast_backbone: str = "mobilenet_v3_small"
    pretrained_backbones: bool = True
    freeze_backbones: bool = True

    # Checkpoint
    ckpt: Optional[str] = None

    # (Optional) Performance
    compile: bool = False

    # (optional) nur zur App-UI: Test-Stride (Sampling-Schritt zwischen Fenstern)
    test_stride: int = 8

    def to_dict(self):
        return asdict(self)


# =============================================================================
# Preprocessing-Helfer
# =============================================================================

def sample_indices(num_src: int, num_out: int) -> List[int]:
    """Gleichmäßig verteilte Indizes von 0..num_src-1 mit Länge num_out."""
    if num_src <= 0:
        return [0] * max(1, num_out)
    if num_out <= 1:
        return [0]
    if num_src == num_out:
        return list(range(num_src))
    idx = np.linspace(0, max(0, num_src - 1), num_out)
    return np.round(idx).astype(int).tolist()


def build_tfm_cpu(size: Tuple[int, int],
                  normalize: bool,
                  mean: Tuple[float, float, float],
                  std: Tuple[float, float, float]):
    """Torchvision-Compose für CPU-Preprocessing (Resize/ToTensor/Norm)."""
    tfms = [transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.ToTensor()]
    if normalize:
        tfms.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(tfms)


def make_slowfast_tensors_cpu(
    frames_rgb: List[np.ndarray],
    t_fast: int,
    t_slow: int,
    tfm: Optional[transforms.Compose] = None,
    out_hw: Optional[Tuple[int, int]] = None,
    normalize: bool = False,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CPU-Pfad: nimmt eine Liste RGB-Frames [H,W,3] (uint8) und erzeugt:
      slow: [3, Ts, H, W], fast: [3, Tf, H, W]  (float32, 0..1, optional normalisiert)

    Entweder gib 'tfm' (Torchvision Compose) an ODER nutze out_hw/normalize/mean/std.
    """
    if tfm is None:
        if out_hw is None:
            raise ValueError("make_slowfast_tensors_cpu: Entweder 'tfm' ODER 'out_hw' angeben.")
        tfm = build_tfm_cpu(out_hw, normalize, mean, std)

    # Fast-Pfad: Tf Frames, gleichmäßig aus frames_rgb gesampelt
    idx_fast = sample_indices(len(frames_rgb), t_fast)
    fast_clip = [tfm(frames_rgb[i]) for i in idx_fast]   # Liste [C,H,W] (float)
    fast = torch.stack(fast_clip, dim=1).contiguous()    # [C,Tf,H,W]

    # Slow-Pfad: Ts Frames aus den bereits (resized/norm) fast_clip
    idx_slow = sample_indices(len(fast_clip), t_slow)
    slow_clip = [fast_clip[i] for i in idx_slow]
    slow = torch.stack(slow_clip, dim=1).contiguous()    # [C,Ts,H,W]

    return slow, fast


def make_slowfast_tensors_gpu(
    frames_rgb: List[np.ndarray],
    out_hw: Tuple[int, int],
    t_fast: int,
    t_slow: int,
    device: torch.device,
    normalize: bool = False,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-Pfad: arbeitet komplett auf 'device'.
    Gibt slow, fast mit Shape [3,T,H,W] (float32) zurück.
    """
    dev = device if isinstance(device, torch.device) else torch.device(device)
    x = torch.from_numpy(np.stack(frames_rgb, axis=0)).to(dev, dtype=torch.float32, non_blocking=True)
    x = x.permute(0, 3, 1, 2).contiguous() / 255.0            # [T,3,H,W]
    x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)

    if normalize:
        mean_t = torch.tensor(mean, device=dev, dtype=x.dtype).view(1, 3, 1, 1)
        std_t  = torch.tensor(std,  device=dev, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean_t) / std_t

    T = x.shape[0]
    idx_fast = torch.linspace(0, max(0, T - 1), steps=t_fast, device=dev).round().long()
    fast = x.index_select(0, idx_fast).permute(1, 0, 2, 3).contiguous()  # [3,Tf,H,W]

    idx_slow = torch.linspace(0, fast.shape[1] - 1, steps=t_slow, device=dev).round().long()
    slow = fast.index_select(1, idx_slow).contiguous()                    # [3,Ts,H,W]

    return slow, fast


# =============================================================================
# Modellvarianten
# =============================================================================

class ImageNetNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


def _load_backbone_2d(name: str, pretrained: bool = True) -> nn.Module:
    """Kleiner Loader für 2D-Backbones aus torchvision; gibt ein Feature-Extraktor-Modul zurück."""
    from torchvision import models

    def _weights(klass, flag):
        try:
            return klass.DEFAULT if flag else None
        except Exception:
            return None

    name = name.lower()
    if name in ("resnet18", "resnet-18"):
        try:
            m = models.resnet18(weights=_weights(models.ResNet18_Weights, pretrained))
        except Exception:
            m = models.resnet18(pretrained=pretrained)
        trunk = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4,
            m.avgpool, nn.Flatten()
        )
        return trunk

    if name in ("mobilenet_v3_small", "mobilenetv3_small"):
        try:
            m = models.mobilenet_v3_small(weights=_weights(models.MobileNet_V3_Small_Weights, pretrained))
        except Exception:
            m = models.mobilenet_v3_small(pretrained=pretrained)
        return nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())

    if name in ("mobilenet_v2", "mobilenetv2"):
        try:
            m = models.mobilenet_v2(weights=_weights(models.MobileNet_V2_Weights, pretrained))
        except Exception:
            m = models.mobilenet_v2(pretrained=pretrained)
        return nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())

    if name in ("shufflenet_v2_x0_5", "shufflenet0_5"):
        try:
            m = models.shufflenet_v2_x0_5(weights=_weights(models.ShuffleNet_V2_X0_5_Weights, pretrained))
        except Exception:
            m = models.shufflenet_v2_x0_5(pretrained=pretrained)
        return nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())

    raise ValueError(f"Unknown backbone: {name}")


class FrameEncoder(nn.Module):
    """2D-Backbone pro Frame -> temporal mean über T (leichtgewichtig)."""
    def __init__(self, backbone_name: str, pretrained: bool = True, freeze: bool = True):
        super().__init__()
        self.norm = ImageNetNorm()
        self.backbone = _load_backbone_2d(backbone_name, pretrained=pretrained)
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x_bcthw: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x_bcthw.shape
        x = x_bcthw.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W).contiguous()
        x = self.norm(x)
        feats = self.backbone(x)           # [B*T, D]
        D = feats.shape[-1]
        feats = feats.view(B, T, D)
        out = feats.mean(dim=1)            # [B, D]
        return out


class SimpleSlowFast(nn.Module):
    """
    Leichtgewichtige SlowFast-Variante:
      - je Pfad ein 2D-Encoder über Frames, temporal gemittelt
      - concat + kleiner MLP-Head -> 1 Logit
    Erwarteter Input:
      model([slow, fast]) mit
      slow: [B,3,Ts,H,W], fast: [B,3,Tf,H,W]
    """
    def __init__(
        self,
        slow_backbone: str = "resnet18",
        fast_backbone: str = "mobilenet_v3_small",
        pretrained_backbones: bool = True,
        freeze_backbones: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.slow_enc = FrameEncoder(slow_backbone, pretrained=pretrained_backbones, freeze=freeze_backbones)
        self.fast_enc = FrameEncoder(fast_backbone, pretrained=pretrained_backbones, freeze=freeze_backbones)
        self.head = nn.Sequential(
            nn.LazyLinear(256), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(256, 1)
        )

    def forward(self, inputs):
        slow, fast = inputs
        f_slow = self.slow_enc(slow)
        f_fast = self.fast_enc(fast)
        f = torch.cat([f_slow, f_fast], dim=1)
        return self.head(f)  # [B,1]


# (Optional) sehr kompakte 3D-Version – in dieser Datei nicht zwingend genutzt
class ImageNetNorm3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


# =============================================================================
# Modell-Erzeugung und Laden
# =============================================================================

def _build_model_hub(device: torch.device) -> nn.Module:
    if hub_slowfast_r50 is None:
        raise RuntimeError("pytorchvideo ist nicht installiert. Installiere 'pytorchvideo' oder nutze model='simple'.")
    model = hub_slowfast_r50(pretrained=False)
    # 1-Logit Binary Head
    model.blocks[-1].proj = nn.Linear(model.blocks[-1].proj.in_features, 1)
    return model.to(device)


def _build_model_simple(cfg: Config, device: torch.device) -> nn.Module:
    model = SimpleSlowFast(
        slow_backbone=cfg.slow_backbone,
        fast_backbone=cfg.fast_backbone,
        pretrained_backbones=cfg.pretrained_backbones,
        freeze_backbones=cfg.freeze_backbones,
        dropout=0.2,
    ).to(device)
    return model


def build_model(cfg: Config, device: torch.device) -> nn.Module:
    if cfg.model == "hub":
        model = _build_model_hub(device)
    elif cfg.model == "simple":
        model = _build_model_simple(cfg, device)
    else:
        raise ValueError("cfg.model must be 'simple' or 'hub'")

    if cfg.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            pass
    return model


def load_model(cfg: Config, device: torch.device) -> nn.Module:
    """
    Baut das gewünschte Modell (cfg.model) und lädt Gewichte aus cfg.ckpt.
    Gibt das Modell im eval()-Modus zurück.
    """
    if not cfg.ckpt or not os.path.isfile(cfg.ckpt):
        raise ValueError(f"Checkpoint nicht gefunden oder nicht gesetzt: {cfg.ckpt}")

    model = build_model(cfg, device)
    state = torch.load(cfg.ckpt, map_location=device)

    # Tolerantes Laden (falls Keys minimal abweichen)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load_model] Warn: fehlende Keys: {missing}")
    if unexpected:
        print(f"[load_model] Warn: unerwartete Keys: {unexpected}")

    model = model.to(device)
    model.eval()
    print(f"✅ Modell geladen: '{cfg.model}' | ckpt='{cfg.ckpt}' | device={device}")
    return model