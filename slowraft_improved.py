import torch
import torch.nn as nn
import torch.nn.functional as F


class RAFTWrapper(nn.Module):
    """RAFT immer in Float32 und Eval, optional Downscale intern, mit einfachem Caching."""
    def __init__(self, raft_model: nn.Module, scale_ratio: float = 1.0,
                 align_mult: int = 8, max_cache_size: int = 32):
        super().__init__()
        self.raft = raft_model
        self.scale = float(max(0.125, min(1.0, scale_ratio)))
        self.mult = align_mult
        self.cache = {}
        self.max_cache = max_cache_size

    def _down(self, img: torch.Tensor):
        B, C, H, W = img.shape
        if self.scale < 1.0:
            Hs = int((H * self.scale) // self.mult) * self.mult
            Ws = int((W * self.scale) // self.mult) * self.mult
            Hs = max(self.mult, Hs)
            Ws = max(self.mult, Ws)
        else:
            Hs = (H // self.mult) * self.mult
            Ws = (W // self.mult) * self.mult
            if Hs == H and Ws == W:
                return img, (H, W), (H, W)
        img_small = F.interpolate(img, size=(Hs, Ws), mode="bilinear", align_corners=False)
        return img_small, (H, W), (Hs, Ws)

    def _up_flow(self, flow: torch.Tensor, orig_hw):
        H, W = orig_hw
        h, w = flow.shape[-2:]
        scale_h = H / h
        scale_w = W / w
        flow_up = F.interpolate(flow, size=(H, W), mode="bilinear", align_corners=False)
        flow_up[:, 0] *= scale_w
        flow_up[:, 1] *= scale_h
        return flow_up

    @torch.no_grad()
    def forward(self, im1: torch.Tensor, im2: torch.Tensor) -> torch.Tensor:
        self.raft.eval()
        key = (im1.data_ptr(), im2.data_ptr())
        if key in self.cache:
            return self.cache[key]

        i1, orig_hw, _ = self._down(im1)
        i2, _, _ = self._down(im2)
        with torch.cuda.amp.autocast(enabled=False):
            out = self.raft(i1.float(), i2.float())
        flow = out[-1] if isinstance(out, (list, tuple)) else out
        flow = self._up_flow(flow, orig_hw)

        if len(self.cache) >= self.max_cache:
            self.cache.clear()
        self.cache[key] = flow
        return flow


class FlowHeadSmall(nn.Module):
    """Kleiner CNN-Encoder für Flusskarten mit LayerNorm für stabilere Features."""
    def __init__(self, in_ch: int = 3, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, out_dim, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.norm = nn.LayerNorm(out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x).flatten(1)
        return self.norm(feat)


class TemporalConv1D(nn.Module):
    """1D-Faltung über Zeit für Feature-Aggregation."""
    def __init__(self, in_dim, hidden_dim=128, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):  # x: [B, T, C]
        x = x.permute(0, 2, 1)
        x = self.pool(self.act(self.conv(x)))
        return x.squeeze(-1)


class ResNet18BackboneImproved(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision import models as tvm
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4
        self.avgpool = m.avgpool
        self.temporal_agg = TemporalConv1D(in_dim=512, hidden_dim=256)
        self.out_dim = 256

    def forward(self, x):  # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B*T, C, H, W)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feat = self.avgpool(x).flatten(1).view(B, T, 512)
        return self.temporal_agg(feat)  # [B, 256]


class SlowRAFTModelImproved(nn.Module):
    """SlowRAFT++ mit Lateral Fusion, TemporalConv, Caching und Flow-Normalisierung."""
    def __init__(self,
                 raft_model: nn.Module,
                 pretrained_resnet: bool = True,
                 delta: int = 1,
                 pair_stride: int = 1,
                 max_pairs: int = 0,
                 raft_scale: float = 0.5,
                 beta: float = 0.25,
                 use_batched_raft: bool = True):
        super().__init__()
        self.slow = ResNet18BackboneImproved(pretrained=pretrained_resnet)
        self.fast_raft = RAFTWrapper(raft_model, scale_ratio=raft_scale)
        fast_dim = int(256 * beta)
        self.fast_head = FlowHeadSmall(in_ch=3, out_dim=fast_dim)
        self.delta = int(max(1, delta))
        self.pair_stride = int(max(1, pair_stride))
        self.max_pairs = int(max(0, max_pairs))
        self.use_batched_raft = use_batched_raft
        self.lateral = nn.Linear(fast_dim, self.slow.out_dim)
        self.head = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(self.slow.out_dim + fast_dim, 1))

    def forward(self, inputs):
        slow, fast = inputs
        B, C, Ts, Hs, Ws = slow.shape
        _, _, Tf, Hf, Wf = fast.shape
        slow_feat = self.slow(slow)

        t_candidates = list(range(0, Tf - self.delta, self.pair_stride))
        if self.max_pairs > 0 and len(t_candidates) > self.max_pairs:
            idx = torch.linspace(0, len(t_candidates)-1, self.max_pairs).round().long()
            t_candidates = [t_candidates[i] for i in idx]

        if len(t_candidates) == 0:
            fast_feat = torch.zeros(B, self.fast_head.out_dim, device=slow.device)
        elif self.use_batched_raft:
            im1_list, im2_list = [], []
            for t in t_candidates:
                im1_list.append(fast[:, :, t, :, :])
                im2_list.append(fast[:, :, t + self.delta, :, :])
            im1 = torch.cat(im1_list, dim=0)
            im2 = torch.cat(im2_list, dim=0)
            with torch.cuda.amp.autocast(enabled=False):
                flow = self.fast_raft(im1, im2)
            flow = flow.view(B, len(t_candidates), 2, Hf, Wf)
            mag = torch.sqrt(flow[:, :, 0:1]**2 + flow[:, :, 1:2]**2 + 1e-6)
            fmap = torch.cat([flow, mag], dim=2).view(B * len(t_candidates), 3, Hf, Wf)
            fast_feat = self.fast_head(fmap).view(B, len(t_candidates), -1).mean(dim=1)
        else:
            feats = []
            for t in t_candidates:
                im1, im2 = fast[:, :, t, :, :], fast[:, :, t + self.delta, :, :]
                with torch.cuda.amp.autocast(enabled=False):
                    flow = self.fast_raft(im1, im2)
                mag = torch.sqrt(flow[:, 0:1]**2 + flow[:, 1:2]**2 + 1e-6)
                fmap = torch.cat([flow, mag], dim=1)
                feats.append(self.fast_head(fmap))
            fast_feat = torch.stack(feats, dim=1).mean(dim=1)

        lateral = self.lateral(fast_feat)
        fused = slow_feat + lateral
        x = torch.cat([fused, fast_feat], dim=1)
        return self.head(x)
