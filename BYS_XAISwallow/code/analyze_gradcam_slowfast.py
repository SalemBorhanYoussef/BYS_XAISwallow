"""
Analyze SlowFast predictions with Grad-CAM on slow and fast pathways.

- Loads a trained SlowFast model (pytorchvideo SlowFast R50 with 1-logit head)
- Picks multiple positive and multiple negative windows from a SINGLE ROI video provided explicitly
- Computes Grad-CAM for:
    * Slow path: center time step (single frame CAM)
    * Fast path: full temporal sequence (per-frame CAMs)
- Saves visualization overlays to an output folder.

Requirements (same as training):
    pip install torch torchvision pytorchvideo opencv-python numpy
Optionally for nice color maps:
    pip install matplotlib

Usage (no argparse):
    1) Set `video_path` and `annotation_path` plus `num_pos` / `num_neg` in the AnalyzeConfig at the bottom.
    2) Run: python analyze_gradcam_slowfast.py
    3) Or import and call run_analysis(AnalyzeConfig(...)).

Notes:
- This script reuses utilities from train_slowfast.py (Config, dataset helpers, model builder).
- Provide the exact ROI video file and matching ELAN annotation .txt instead of a dataset directory scan.
"""

import os
import json
import time
from typing import Tuple, List, Optional, Dict

import h5py
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict

import numpy as np
import cv2
import torch
import torch.nn as nn

# Reuse code from training script
from train_slowfast import (
    Config,
    build_model_slowfast,
    SlowFastDataset,
    build_tfm_cpu,
    make_slowfast_tensors_cpu,
)


@dataclass
class AnalyzeConfig:
    """Configuration for Grad-CAM analysis (single ROI video, multiple examples, no argparse).

    Fields:
        ckpt: Path to trained SlowFast checkpoint; auto-resolved if None.
        video_path: Path to a single ROI video file.
        annotation_path: Path to the matching ELAN annotation .txt for the video.
        positive_label: ELAN label string considered positive.
        out_dir: Output directory for overlays/videos.
        window_size, t_fast, alpha, resize_h, resize_w: Optional overrides of training config.
    threshold: Window-level soft label threshold (used for positive/negative selection). If None use training cfg.pos_threshold_eval.
    num_pos: Number of positive windows to visualize.
    num_neg: Number of negative windows to visualize.
    random_sample: If True sample without replacement among candidates, otherwise take deterministic sorted order.
    seed: RNG seed for sampling when random_sample is True.
        device: 'cuda' or 'cpu'. If None falls back to training Config default.
    """
    ckpt: Optional[str] = None
    video_path: str = "D:/Master_Arbeit/TSwallowDataset/sample_roi.mp4"  # <-- update to real file
    annotation_path: str = "D:/Master_Arbeit/TSwallowDataset/sample_annotation.txt"  # <-- update to real file
    positive_label: str = "none"
    out_dir: str = "runs/gradcam"
    window_size: Optional[int] = None
    t_fast: Optional[int] = None
    alpha: Optional[int] = None
    resize_h: Optional[int] = None
    resize_w: Optional[int] = None
    threshold: Optional[float] = None
    num_pos: int = 1
    num_neg: int = 1
    random_sample: bool = False
    seed: int = 0
    device: Optional[str] = None
    # If True, normalize Grad-CAM maps (min-max per-frame / per-center) before using as masks
    normalize_cam: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def print(self, header: Optional[str] = None) -> None:
        if header:
            print(header)
        print(self.to_json())


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _read_window_frames(video_path: str, start: int, window: int, resize_hw: Tuple[int, int]) -> List[np.ndarray]:
    """Read a fixed window of frames starting at index 'start' (inclusive).
    Returns a list of RGB uint8 frames with shape (H, W, 3), resized to resize_hw (h, w).
    """
    cap = cv2.VideoCapture(video_path)
    if start > 0:
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
        except Exception:
            pass
    frames = []
    last = None
    for _ in range(window):
        ok, fr = cap.read()
        if not ok or fr is None:
            if last is not None:
                frames.append(last.copy())
            else:
                frames.append(np.zeros((resize_hw[0], resize_hw[1], 3), dtype=np.uint8))
            continue
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        fr = cv2.resize(fr, (resize_hw[1], resize_hw[0]), interpolation=cv2.INTER_LINEAR)
        frames.append(fr)
        last = fr
    cap.release()
    return frames


def _find_flow_h5(video_path: str) -> Optional[str]:
    """Guess common HDF5 filenames that may contain precomputed flows for a video.

    Tries several candidates and returns first existing path or None.
    """
    base, ext = os.path.splitext(video_path)
    candidates = [
        base + "_flow.h5",
        base + "_flows.h5",
        base + "_flow_raft.h5",
        base + ".h5",
        base + "_optflow.h5",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def _read_flow_h5_segment(h5_path: str, start: int, length: int) -> Optional[np.ndarray]:
    """Read a segment of optical flow from an HDF5 file.

    Attempts to support datasets stored as:
      - 'flow' or 'flows' shape [T, H, W, 2]
      - 'u' and 'v' datasets shape [T, H, W]
      - 'flow' shape [2, T, H, W]

    Returns numpy array shape [T, H, W, 2] or None if not found/compatible.
    """
    try:
        with h5py.File(h5_path, "r") as hf:
            # helper to read dataset slice and dequantize if needed
            def _read_ds(ds) -> np.ndarray:
                # ds is an h5py.Dataset
                # read slice
                arr = ds[start: start + length]
                # convert to float32 and dequantize if int dtype with scale attr
                try:
                    dtype_kind = getattr(ds.dtype, 'kind', None)
                except Exception:
                    dtype_kind = None
                scale = float(ds.attrs.get('scale', 1.0)) if hasattr(ds, 'attrs') else 1.0
                if dtype_kind in ('i', 'u'):
                    return np.asarray(arr, dtype=np.float32) / (scale if scale != 0 else 1.0)
                else:
                    # float16 or float32
                    return np.asarray(arr, dtype=np.float32)

            # prefer 'flow' or 'flows'
            if 'flow' in hf:
                d = hf['flow']
                arr = _read_ds(d)
                # possible shapes: [T, H, W, 2] or [2, T, H, W]
                if arr.ndim == 4 and arr.shape[-1] == 2:
                    return arr.astype(np.float32)
                if arr.ndim == 4 and arr.shape[0] == 2:
                    # [2, T, H, W] -> transpose
                    arr = np.transpose(arr, (1, 2, 3, 0))
                    return arr.astype(np.float32)
            if 'flows' in hf:
                d = hf['flows']
                arr = _read_ds(d)
                if arr.ndim == 4 and arr.shape[-1] == 2:
                    return arr.astype(np.float32)
            # try u/v (separate components)
            if 'u' in hf and 'v' in hf:
                du = hf['u']
                dv = hf['v']
                u = np.asarray(du[start: start + length], dtype=np.float32)
                v = np.asarray(dv[start: start + length], dtype=np.float32)
                # check for scale attrs on u/v
                scale_u = float(du.attrs.get('scale', 1.0)) if hasattr(du, 'attrs') else 1.0
                scale_v = float(dv.attrs.get('scale', 1.0)) if hasattr(dv, 'attrs') else 1.0
                if scale_u != 1.0:
                    u = u / (scale_u if scale_u != 0 else 1.0)
                if scale_v != 1.0:
                    v = v / (scale_v if scale_v != 0 else 1.0)
                if u.ndim == 3 and v.ndim == 3:
                    return np.stack([u, v], axis=-1).astype(np.float32)
    except Exception as e:
        print(f"[WARN] Could not read flows from {h5_path}: {e}")
    return None


# ===== CAM-masked optical-flow exports (parity with optflow analyzer) =====
def _save_heatmap_image(arr2d: np.ndarray, out_path: str):
    a = arr2d.astype(np.float32)
    a = a - a.min()
    mx = a.max()
    if mx > 0:
        a = a / mx
    img = (a * 255.0).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    cv2.imwrite(out_path, img)


def _map_time_indices(src_len: int, dst_len: int) -> List[int]:
    if src_len <= 1:
        return [0] * max(1, dst_len)
    if dst_len <= 1:
        return [0]
    idx = np.round(np.linspace(0, src_len - 1, dst_len)).astype(int).tolist()
    return idx


def _export_flow_cam_csv(flow_np: np.ndarray, heat_t: Optional[np.ndarray], out_csv: str):
    """Export per-pixel raw and CAM-masked flow with per-frame heat.
    Columns: frame,y,x,u,v,heat,u_masked,v_masked
    """
    if flow_np is None:
        return
    T, H, W, _ = flow_np.shape
    Npix = H * W
    y_grid, x_grid = np.meshgrid(np.arange(H, dtype=np.int32), np.arange(W, dtype=np.int32), indexing='ij')
    y_flat = y_grid.reshape(-1)
    x_flat = x_grid.reshape(-1)

    rows = []
    header = 'frame,y,x,u,v,heat,u_masked,v_masked'
    for t in range(T):
        u = flow_np[t, :, :, 0]
        v = flow_np[t, :, :, 1]
        if heat_t is not None:
            h2d = heat_t[t].astype(np.float32)
            # ensure [0,1]
            h2d = h2d - h2d.min()
            mx = h2d.max()
            if mx > 0:
                h2d = h2d / mx
        else:
            h2d = np.zeros((H, W), dtype=np.float32)
        u_flat = u.reshape(-1)
        v_flat = v.reshape(-1)
        h_flat = h2d.reshape(-1)
        um = u_flat * h_flat
        vm = v_flat * h_flat
        t_col = np.full((Npix,), t, dtype=np.int32)
        block = np.stack([t_col, y_flat, x_flat, u_flat, v_flat, h_flat, um, vm], axis=1)
        rows.append(block)
    if rows:
        data = np.concatenate(rows, axis=0)
        np.savetxt(
            out_csv,
            data,
            delimiter=',',
            header=header,
            comments='',
            fmt=['%d','%d','%d','%.6f','%.6f','%.6f','%.6f','%.6f']
        )


def _flow_to_bgr(flow: np.ndarray) -> np.ndarray:
    """Visualize optical flow (u,v) as BGR image using HSV mapping (H=angle, V=mag)."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
    ang_deg = ang * 180.0 / np.pi / 2.0
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = np.clip(ang_deg, 0, 179).astype(np.uint8)
    m = mag
    if m.max() > 0:
        m = (m / m.max()) * 255.0
    hsv[..., 2] = np.clip(m, 0, 255).astype(np.uint8)
    hsv[..., 1] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def _export_magnitude_2d(
    mag_arr: np.ndarray,
    cam_t: Optional[np.ndarray],
    out_csv: str,
    out_mag_png: Optional[str] = None,
    out_heat_png: Optional[str] = None,
    out_masked_png: Optional[str] = None,
    agg: str = 'mean',
):
    if mag_arr is None:
        return
    T, H, W = mag_arr.shape
    if agg == 'max':
        mag2d = mag_arr.max(axis=0)
    else:
        mag2d = mag_arr.mean(axis=0)
    if cam_t is not None and cam_t.shape[0] == T:
        heat2d = cam_t.mean(axis=0)
    else:
        heat2d = np.ones_like(mag2d, dtype=np.float32)
    h = heat2d - heat2d.min()
    mx = h.max()
    heat2d = h / mx if mx > 0 else np.zeros_like(h)
    mag_masked = mag2d * heat2d
    y_grid, x_grid = np.meshgrid(np.arange(H, dtype=np.int32), np.arange(W, dtype=np.int32), indexing='ij')
    data = np.stack([
        y_grid.reshape(-1),
        x_grid.reshape(-1),
        mag2d.reshape(-1),
        heat2d.reshape(-1),
        mag_masked.reshape(-1)
    ], axis=1)
    np.savetxt(out_csv, data, delimiter=',', header='y,x,mag,heat,mag_masked', comments='', fmt=['%d','%d','%.6f','%.6f','%.6f'])
    if out_mag_png:
        _save_heatmap_image(mag2d, out_mag_png)
    if out_heat_png:
        _save_heatmap_image(heat2d, out_heat_png)
    if out_masked_png:
        _save_heatmap_image(mag_masked, out_masked_png)


def _export_flow_images(flow_np: np.ndarray, heat_t: Optional[np.ndarray], indices: List[int], out_dir: str):
    if flow_np is None or not indices:
        return
    T, H, W, _ = flow_np.shape
    for t in indices:
        if t < 0 or t >= T:
            continue
        raw_bgr = _flow_to_bgr(flow_np[t])
        cv2.imwrite(os.path.join(out_dir, f"flow_raw_t{t:03d}.png"), raw_bgr)
        if heat_t is not None:
            h2d = heat_t[t].astype(np.float32)
            h2d = h2d - h2d.min()
            mx = h2d.max()
            if mx > 0:
                h2d = h2d / mx
            masked = np.empty_like(flow_np[t])
            masked[..., 0] = flow_np[t][..., 0] * h2d
            masked[..., 1] = flow_np[t][..., 1] * h2d
            masked_bgr = _flow_to_bgr(masked)
            cv2.imwrite(os.path.join(out_dir, f"flow_masked_t{t:03d}.png"), masked_bgr)


def _export_motion_energy(flow_np: np.ndarray, heat_t: Optional[np.ndarray], out_csv: str):
    if flow_np is None:
        return
    T, H, W, _ = flow_np.shape
    rows = []
    for t in range(T):
        u = flow_np[t, :, :, 0].astype(np.float32)
        v = flow_np[t, :, :, 1].astype(np.float32)
        raw_e = float((u*u + v*v).sum())
        if heat_t is not None:
            h2d = heat_t[t].astype(np.float32)
            h2d = h2d - h2d.min()
            mx = h2d.max()
            if mx > 0:
                h2d = h2d / mx
            masked_e = float((((u*h2d)**2) + ((v*h2d)**2)).sum())
        else:
            masked_e = raw_e
        rows.append([t, masked_e, raw_e])
    data = np.array(rows, dtype=np.float32)
    np.savetxt(out_csv, data, delimiter=',', header='frame,energy_masked,energy_raw', comments='', fmt=['%d','%.6f','%.6f'])


def _export_fastpath_flow_curves(
    flows: np.ndarray,                # [T_flow,H,W,2]
    cam_fast_seq: Optional[np.ndarray],  # [T_cam,Hc,Wc]
    out_csv: str,
    out_png: Optional[str] = None,
    fps: float = 32.0,
):
    """Export flow magnitude curves over fast-path time: raw vs CAM-masked.
    CSV columns: t_fast,flow_index,time_s,raw_mean,masked_mean,raw_sum,masked_sum
    """
    if flows is None:
        return
    T_flow, H, W, _ = flows.shape
    T_cam = 0 if cam_fast_seq is None else int(np.asarray(cam_fast_seq).shape[0])
    if T_cam <= 0:
        # fallback: use flow frames as 'fast' time
        T_cam = T_flow
        idx_map = list(range(T_flow))
        cam_fast_res = None
    else:
        idx_map = _map_time_indices(T_flow, T_cam)
        # upsample CAM to flow res per fast t
        cam_fast_res = np.zeros((T_cam, H, W), dtype=np.float32)
        for t in range(T_cam):
            fr = np.asarray(cam_fast_seq[t]).astype(np.float32)
            fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_LINEAR)
            # min-max per-frame to [0,1]
            mn = float(np.nanmin(fr))
            mx = float(np.nanmax(fr))
            if (mx - mn) > 1e-6:
                fr = (fr - mn) / (mx - mn)
            else:
                fr = np.zeros_like(fr)
            cam_fast_res[t] = fr

    rows = []
    raw_curve = []
    masked_curve = []
    times = []
    for t in range(T_cam):
        i = int(idx_map[t])
        u = flows[i, :, :, 0].astype(np.float32)
        v = flows[i, :, :, 1].astype(np.float32)
        mag = np.sqrt(u*u + v*v)
        raw_mean = float(mag.mean())
        raw_sum = float(mag.sum())
        if cam_fast_res is not None:
            m = cam_fast_res[t]
            um = u * m
            vm = v * m
            magm = np.sqrt(um*um + vm*vm)
            masked_mean = float(magm.mean())
            masked_sum = float(magm.sum())
        else:
            masked_mean = raw_mean
            masked_sum = raw_sum
        time_s = float(i) / float(max(fps, 1e-6))
        times.append(time_s)
        raw_curve.append(raw_mean)
        masked_curve.append(masked_mean)
        rows.append([t, i, time_s, raw_mean, masked_mean, raw_sum, masked_sum])

    data = np.array(rows, dtype=np.float32)
    np.savetxt(
        out_csv,
        data,
        delimiter=',',
        header='t_fast,flow_index,time_s,raw_mean,masked_mean,raw_sum,masked_sum',
        comments='',
        fmt=['%d','%d','%.6f','%.6f','%.6f','%.6f','%.6f']
    )

    if out_png is not None:
        try:
            plt.figure(figsize=(10, 3))
            plt.plot(times, raw_curve, label='raw mean |F|', color='gray')
            plt.plot(times, masked_curve, label='masked mean |M·F|', color='red')
            plt.xlabel('Time (s)')
            plt.ylabel('Magnitude')
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_png)
            plt.close()
        except Exception as e:
            print(f"[WARN] fastpath curves plot failed: {e}")


def _sample_indices(num_src: int, num_out: int) -> List[int]:
    if num_src <= 0:
        return [0] * num_out
    if num_out <= 1:
        return [0]
    if num_src == num_out:
        return list(range(num_src))
    idx = np.linspace(0, max(0, num_src - 1), num_out)
    return np.round(idx).astype(int).tolist()


def analyze_flow_with_cam(h5_path: str,
                          cam_fast_seq: np.ndarray,
                          slow_cam_center: Optional[np.ndarray],
                          combine_mode: str,
                          out_dir: str,
                          start_frame: int,
                          window: int,
                          fps: float = 32.0,
                          smooth_window: int = 5,
                          show_plot: bool = False,
                          save_plot: bool = True,
                          normalize_cam: bool = True):
    """Analyze optical flow in HDF5 using a Grad-CAM sequence as attention mask.

    Args:
        h5_path: path to HDF5 containing flow arrays.
        cam_seq: numpy array [T_cam, H_cam, W_cam] with values in [0,1].
        out_dir: directory to save plots/results.
        start_frame: start frame index in the original video corresponding to the first flow frame to read.
        window: number of frames to analyze (flow frames to read: start_frame .. start_frame+window-1)
        fps: framerate for plotting/time axis.
    Returns:
        result dict similar to analyze_flow_video or None on error.
    """
    _ensure_dir(out_dir)
    flows = _read_flow_h5_segment(h5_path, start_frame, window)
    if flows is None:
        print(f"[WARN] No flow data read from {h5_path}")
        return None

    # flows shape: [T, H, W, 2]
    T, H, W, C = flows.shape
    # compute per-frame magnitude
    mags = np.sqrt(np.square(flows[..., 0]) + np.square(flows[..., 1]))  # [T,H,W]

    # cam_seq: optional fast CAM sequence [T_cam, Hc, Wc]
    cam = None
    if cam_fast_seq is not None:
        try:
            cam_tmp = np.asarray(cam_fast_seq)
            if cam_tmp.ndim == 3:
                cam = cam_tmp.astype(np.float32)
            else:
                print(f"[WARN] cam_fast_seq unexpected ndim={cam_tmp.ndim}; proceeding without fast CAM")
                cam = None
        except Exception as e:
            print(f"[WARN] could not parse cam_fast_seq: {e}; proceeding without fast CAM")
            cam = None

    # Optionally normalize CAMs (per-frame min-max for fast sequence; min-max for slow center)
    if normalize_cam:
        try:
            if cam is not None:
                # normalize each frame in the fast sequence to [0,1]
                if cam.ndim == 3:
                    cam_n = np.zeros_like(cam, dtype=np.float32)
                    for ti in range(cam.shape[0]):
                        f = cam[ti].astype(np.float32)
                        mn = float(np.nanmin(f))
                        mx = float(np.nanmax(f))
                        if (mx - mn) > 1e-6:
                            cam_n[ti] = (f - mn) / (mx - mn)
                        else:
                            cam_n[ti] = np.zeros_like(f)
                    cam = cam_n
        except Exception as e:
            print(f"[WARN] CAM fast-seq normalization failed: {e}")

        try:
            if slow_cam_center is not None:
                sc = np.asarray(slow_cam_center).astype(np.float32)
                mn = float(np.nanmin(sc))
                mx = float(np.nanmax(sc))
                if (mx - mn) > 1e-6:
                    slow_cam_center = (sc - mn) / (mx - mn)
                else:
                    slow_cam_center = np.zeros_like(sc)
        except Exception as e:
            print(f"[WARN] CAM slow-center normalization failed: {e}")

    # Prepare slow center mask if provided (replicated across frames)
    slow_mask_resized = None
    if slow_cam_center is not None:
        sc = np.asarray(slow_cam_center)
        if sc.ndim == 2:
            slow_mask_resized = cv2.resize(sc.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
            slow_mask_resized = np.clip(slow_mask_resized, 0.0, 1.0)

    masked_means = []

    # Precompute common masks
    center_fast_mask = None
    if cam is not None and cam.ndim == 3:
        try:
            center_idx = cam.shape[0] // 2
            center_heat = cam[center_idx]
            center_r = cv2.resize(center_heat.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
            center_fast_mask = np.clip(center_r, 0.0, 1.0)
        except Exception:
            center_fast_mask = None

    # Use actual available flow frames (T) to avoid indexing past EOF in HDF5
    analysis_len = min(window, int(T))

    for t in range(analysis_len):
        # Build final mask according to combine_mode without requiring per-frame fast CAM
        if combine_mode == 'none':
            final_mask = np.ones((H, W), dtype=np.float32)
        elif combine_mode == 'slow':
            if slow_mask_resized is not None:
                final_mask = slow_mask_resized
            else:
                final_mask = np.ones((H, W), dtype=np.float32)
        elif combine_mode == 'fast_center':
            if center_fast_mask is not None:
                final_mask = center_fast_mask
            else:
                # fallback to ones if no fast CAM available
                final_mask = np.ones((H, W), dtype=np.float32)
        else:
            final_mask = np.ones((H, W), dtype=np.float32)

        masked = mags[t] * final_mask
        masked_means.append(float(np.mean(masked)))

    masked_means = np.array(masked_means, dtype=np.float32)

    # smoothing
    smoothed = pd.Series(masked_means).rolling(window=smooth_window, min_periods=1).mean().to_numpy()

    # dynamic threshold
    mean_mag = float(np.mean(smoothed))
    std_mag = float(np.std(smoothed))
    threshold = mean_mag + 4.5 * std_mag

    peaks, _ = find_peaks(smoothed, height=threshold, distance=int(fps * 1.0))

    # time axis for plots — use actual analyzed length
    time_sec = (np.arange(len(smoothed)) + 0) / float(fps)

    # save plot for this mode
    if save_plot:
        plt.figure(figsize=(10, 4))
        plt.plot(time_sec, smoothed, label=f"Masked mean magnitude ({combine_mode})", color='blue')
        if peaks.size > 0:
            plt.scatter(time_sec[peaks], smoothed[peaks], color='red', label='Detected peaks')
        plt.axhline(y=threshold, color='gray', linestyle='--', label='Threshold')
        plt.xlabel("Time (s)")
        plt.ylabel("Masked magnitude")
        plt.title(os.path.basename(h5_path))
        plt.legend()
        plt.tight_layout()
        out_plot = os.path.join(out_dir, os.path.basename(h5_path) + f"_masked_magnitude_plot_{combine_mode}.png")
        plt.savefig(out_plot)
        plt.close()

    result = {
        "h5": h5_path,
        "start_frame": int(start_frame),
        "window_requested": int(window),
        "window_used": int(len(smoothed)),
        "fps": float(fps),
        "mean_masked_magnitude": mean_mag,
        "threshold": threshold,
        "n_peaks": int(len(peaks)),
        "peaks": peaks.tolist(),
        "combine_mode": combine_mode,
        "masked_means": masked_means.tolist(),
        "smoothed": smoothed.tolist(),
        "time_sec": time_sec.tolist(),
    }
    # save JSON
    with open(os.path.join(out_dir, "flow_masked_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def _minmax_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))
    if (mx - mn) <= 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def analyze_combined_flow_gradcam(h5_path: str,
                                  cam_fast_seq: Optional[np.ndarray],
                                  slow_cam_center: Optional[np.ndarray],
                                  out_dir: str,
                                  start_frame: int,
                                  window: int = 128,
                                  fps: float = 32.0,
                                  smooth_window: int = 5,
                                  save_plot: bool = True) -> Optional[dict]:
    """Simplified combined analysis: normalize flow and CAM per-frame, combine them, smooth and detect peaks.

    Returns a compact result dict with normalized time series and peaks.
    """
    _ensure_dir(out_dir)
    flows = _read_flow_h5_segment(h5_path, start_frame, window)
    if flows is None:
        print(f"[WARN] No flow data read from {h5_path}")
        return None

    T, H, W, C = flows.shape
    analysis_len = min(window, int(T))
    # per-frame flow magnitude mean
    mags = np.sqrt(np.square(flows[..., 0]) + np.square(flows[..., 1]))  # [T,H,W]
    flow_mean = mags[:analysis_len].reshape(analysis_len, -1).mean(axis=1)

    # CAM energy per frame: prefer fast sequence, else replicate slow center
    cam_mean = None
    if cam_fast_seq is not None:
        try:
            cam = np.asarray(cam_fast_seq)
            if cam.ndim == 3:
                cm = []
                for t in range(min(cam.shape[0], analysis_len)):
                    fr = cv2.resize(cam[t].astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
                    cm.append(float(np.mean(np.clip(fr, 0.0, 1.0))))
                if len(cm) < analysis_len:
                    cm.extend([0.0] * (analysis_len - len(cm)))
                cam_mean = np.array(cm, dtype=np.float32)
        except Exception as e:
            print(f"[WARN] could not compute fast CAM means: {e}")

    if cam_mean is None and slow_cam_center is not None:
        try:
            sc = cv2.resize(np.asarray(slow_cam_center).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
            v = float(np.mean(np.clip(sc, 0.0, 1.0)))
            cam_mean = np.array([v] * analysis_len, dtype=np.float32)
        except Exception:
            cam_mean = np.zeros(analysis_len, dtype=np.float32)

    if cam_mean is None:
        cam_mean = np.zeros(analysis_len, dtype=np.float32)

    # normalize both signals to [0,1]
    flow_n = _minmax_normalize(flow_mean)
    cam_n = _minmax_normalize(cam_mean)

    # combined signal (elementwise product to highlight flow where CAM is strong)
    combined = flow_n * cam_n

    # smoothing
    sm_flow = pd.Series(flow_n).rolling(window=smooth_window, min_periods=1).mean().to_numpy()
    sm_cam = pd.Series(cam_n).rolling(window=smooth_window, min_periods=1).mean().to_numpy()
    sm_comb = pd.Series(combined).rolling(window=smooth_window, min_periods=1).mean().to_numpy()

    # dynamic threshold on combined
    mean_c = float(np.mean(sm_comb))
    std_c = float(np.std(sm_comb))
    thresh = mean_c + 3.0 * std_c
    peaks, _ = find_peaks(sm_comb, height=thresh, distance=int(fps * 0.5))

    time_sec = (np.arange(len(sm_comb)) + 0) / float(fps)

    if save_plot:
        plt.figure(figsize=(12, 4))
        plt.plot(time_sec, sm_flow, label='flow (norm)', color='gray')
        plt.plot(time_sec, sm_cam, label='cam (norm)', color='blue')
        plt.plot(time_sec, sm_comb, label='combined', color='red')
        if peaks.size > 0:
            plt.scatter(time_sec[peaks], sm_comb[peaks], color='black', label='peaks')
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized value')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'combined_flow_cam_norm.png'))
        plt.close()

    result = {
        'h5': h5_path,
        'start_frame': int(start_frame),
        'window_requested': int(window),
        'window_used': int(len(sm_comb)),
        'fps': float(fps),
        'flow_norm': flow_n.tolist(),
        'cam_norm': cam_n.tolist(),
        'combined': sm_comb.tolist(),
        'time_sec': time_sec.tolist(),
        'n_peaks': int(len(peaks)),
        'peaks': peaks.tolist(),
        'threshold': float(thresh),
    }

    with open(os.path.join(out_dir, 'combined_flow_cam.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    return result


def analyze_fastpath_motion(cam_fast_seq: np.ndarray,
                            out_dir: str,
                            fps: float = 32.0,
                            smooth_window: int = 3,
                            downsample_w: int = 128) -> Optional[dict]:
    """Analyze fast-path Grad-CAM sequence and produce a 2D space-time image + motion graph.

    Args:
        cam_fast_seq: numpy [T, H, W] raw CAMs (not necessarily normalized)
        out_dir: output directory to save artifacts
        fps: frames per second for time axis
        smooth_window: rolling window for smoothing time series
        downsample_w: width to downsample the spatial axis for a compact space-time image
    Returns:
        dict with cam_mean, diff_energy, peaks, and paths to saved artifacts
    """
    _ensure_dir(out_dir)
    try:
        cam = np.asarray(cam_fast_seq).astype(np.float32)
    except Exception as e:
        print(f"[WARN] analyze_fastpath_motion: invalid cam data: {e}")
        return None

    if cam.ndim != 3:
        print(f"[WARN] analyze_fastpath_motion: expected cam_fast_seq ndim=3, got {cam.ndim}")
        return None

    T, H, W = cam.shape

    # Per-frame min-max normalize (preserve spatial pattern per frame)
    cam_n = np.zeros_like(cam)
    for t in range(T):
        f = cam[t]
        mn = float(np.nanmin(f))
        mx = float(np.nanmax(f))
        if (mx - mn) > 1e-6:
            cam_n[t] = (f - mn) / (mx - mn)
        else:
            cam_n[t] = np.zeros_like(f)

    # per-frame mean energy and frame-to-frame diff energy
    cam_mean = cam_n.reshape(T, -1).mean(axis=1)
    diff_energy = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        diff_energy[t] = float(np.sqrt(np.mean((cam_n[t] - cam_n[t-1]) ** 2)))

    # create a space-time image: collapse vertical axis by mean -> [T, W]
    st = cam_n.mean(axis=1)  # [T, W]
    # resize to (T, downsample_w)
    try:
        st_img = cv2.resize(st.astype(np.float32), (downsample_w, T), interpolation=cv2.INTER_LINEAR)
    except Exception:
        st_img = st

    # smooth time series
    sm_cam = pd.Series(cam_mean).rolling(window=smooth_window, min_periods=1).mean().to_numpy()
    sm_diff = pd.Series(diff_energy).rolling(window=smooth_window, min_periods=1).mean().to_numpy()

    # detect peaks on the diff (motion) signal
    thresh = float(np.mean(sm_diff)) + 2.5 * float(np.std(sm_diff))
    peaks, _ = find_peaks(sm_diff, height=thresh, distance=int(fps * 0.25))

    time_sec = (np.arange(T) + 0) / float(fps)

    # save space-time image
    st_path = os.path.join(out_dir, 'fast_space_time.png')
    plt.figure(figsize=(10, 4))
    plt.imshow(st_img, aspect='auto', cmap='viridis', origin='lower', extent=[0, downsample_w, 0, T/fps])
    plt.xlabel('spatial (downsampled)')
    plt.ylabel('time (s)')
    plt.title('Fast-path CAM space-time')
    plt.colorbar(label='norm CAM')
    plt.tight_layout()
    plt.savefig(st_path)
    plt.close()

    # save motion plot
    plot_path = os.path.join(out_dir, 'fast_cam_motion.png')
    plt.figure(figsize=(10, 3))
    plt.plot(time_sec, sm_cam, label='cam mean (norm)', color='blue')
    plt.plot(time_sec, sm_diff, label='frame-diff energy', color='red')
    if peaks.size > 0:
        plt.scatter(time_sec[peaks], sm_diff[peaks], color='black', label='peaks')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    result = {
        'T': int(T), 'H': int(H), 'W': int(W),
        'cam_mean': cam_mean.tolist(),
        'diff_energy': diff_energy.tolist(),
        'sm_cam': sm_cam.tolist(),
        'sm_diff': sm_diff.tolist(),
        'peaks': peaks.tolist(),
        'time_sec': time_sec.tolist(),
        'space_time_image': st_path,
        'plot': plot_path,
    }

    with open(os.path.join(out_dir, 'fast_cam_motion.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    return result


class GradCAMHook:
    """Minimal Grad-CAM hook for 3D conv features.

    Captures activations and gradients from a target module. After a backward pass
    on the model outputs, call make_cam(...) to obtain CAM volumes.
    """
    def __init__(self, target_module: nn.Module):
        self.target_module = target_module
        self.activations = None  # [B, C, T, H, W]
        self.gradients = None    # [B, C, T, H, W]
        self._fwd = target_module.register_forward_hook(self._save_activation)
        self._bwd = target_module.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        # output assumed to be a tensor [B, C, T, H, W]
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        # grad_output[0] has shape [B, C, T, H, W]
        self.gradients = grad_output[0].detach()

    def remove(self):
        try:
            self._fwd.remove()
        except Exception:
            pass
        try:
            self._bwd.remove()
        except Exception:
            pass

    @torch.no_grad()
    def make_cam(self) -> Optional[torch.Tensor]:
        """Return CAM tensor of shape [B, T, H, W] normalized to [0,1]."""
        if self.activations is None or self.gradients is None:
            return None
        A = self.activations          # [B, C, T, H, W]
        G = self.gradients            # [B, C, T, H, W]
        if A.ndim != 5 or G.ndim != 5:
            return None
        # weights: global-average pool over (T,H,W)
        weights = G.mean(dim=(2, 3, 4), keepdim=True)  # [B, C, 1, 1, 1]
        cam = (weights * A).sum(dim=1)                 # [B, T, H, W]
        cam = torch.relu(cam)
        # normalize per-sample to [0,1]
        B = cam.shape[0]
        for i in range(B):
            c = cam[i]
            mx = torch.max(c)
            mn = torch.min(c)
            if (mx - mn) > 1e-6:
                cam[i] = (c - mn) / (mx - mn)
            else:
                cam[i] = torch.zeros_like(c)
        return cam


def _find_target_layers(model: nn.Module) -> Dict[str, nn.Module]:
    """Heuristically pick the last Conv3d for pathway0 (slow) and pathway1 (fast).
    If names are not available, fallback to the last Conv3d modules found.
    Returns a dict with keys 'slow' and 'fast'.
    """
    slow_convs = []
    fast_convs = []
    all_convs = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv3d,)):
            all_convs.append((name, m))
            lname = name.lower()
            if "pathway0" in lname or "slow" in lname:
                slow_convs.append((name, m))
            if "pathway1" in lname or "fast" in lname:
                fast_convs.append((name, m))
    # pick last ones if available
    target = {}
    if slow_convs:
        target["slow"] = slow_convs[-1][1]
    if fast_convs:
        target["fast"] = fast_convs[-1][1]
    # fallback
    if "slow" not in target and all_convs:
        target["slow"] = all_convs[-1][1]
    if "fast" not in target and all_convs:
        target["fast"] = all_convs[-1][1]
    return target


def _overlay_heatmap(img_rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.4, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Overlay a heatmap [0,1] onto an RGB uint8 image."""
    h = np.clip((heat * 255.0).astype(np.uint8), 0, 255)
    h = cv2.applyColorMap(h, colormap)
    h = cv2.cvtColor(h, cv2.COLOR_BGR2RGB)
    out = (img_rgb.astype(np.float32) * (1 - alpha) + h.astype(np.float32) * alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def _save_video(frames_rgb: List[np.ndarray], out_path: str, fps: int = 16):
    if len(frames_rgb) == 0:
        return
    h, w = frames_rgb[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for fr in frames_rgb:
        bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
        vw.write(bgr)
    vw.release()


def analyze_one(model: nn.Module,
                device: torch.device,
                frames_rgb: List[np.ndarray],
                cfg: Config,
                example_dir: str,
                pos_label_threshold: float = 0.5,
                flow_h5: Optional[str] = None,
                start_frame: Optional[int] = None,
                normalize_cam: bool = True) -> None:
    """Build slow/fast inputs for frames, run prediction, and write Grad-CAM overlays.

    All artifacts for this example are written into the directory `example_dir`.
    Files created:
        slow_center_cam_prob*.png
        fast_seq_cam_prob*.mp4
        fast_first.png / fast_center.png (if sequence exists)
        center_raw.png
        meta.json (probability)
    """
    t_fast = cfg.t_fast
    t_slow = max(1, t_fast // cfg.alpha)
    tfm = build_tfm_cpu(cfg.normalize, (cfg.resize_h, cfg.resize_w), cfg.mean, cfg.std)

    slow, fast = make_slowfast_tensors_cpu(frames_rgb, t_fast=t_fast, t_slow=t_slow, tfm=tfm)
    slow = slow.unsqueeze(0).to(device)  # [1, C, T, H, W]
    fast = fast.unsqueeze(0).to(device)

    model.eval()

    # pick target modules
    targets = _find_target_layers(model)
    slow_hook = GradCAMHook(targets.get("slow", list(model.modules())[-1]))
    fast_hook = GradCAMHook(targets.get("fast", list(model.modules())[-1]))

    with torch.enable_grad():
        slow.requires_grad_(True)
        fast.requires_grad_(True)
        logits = model([slow, fast])  # [1, 1]
        prob = torch.sigmoid(logits)[0, 0].item()
        # backprop wrt the single logit
        model.zero_grad(set_to_none=True)
        logits[0, 0].backward(retain_graph=False)

    cam_slow = slow_hook.make_cam()  # [1, T, H, W]
    cam_fast = fast_hook.make_cam()  # [1, T, H, W]

    slow_hook.remove()
    fast_hook.remove()

    # placeholder for fast-path motion analysis result
    fast_motion_res = None

    # choose center time for slow
    center_idx = 0
    if cam_slow is not None and cam_slow.shape[1] > 0:
        center_idx = int(cam_slow.shape[1] // 2)
    # We overlay on the raw center frame (approx alignment)
    center_frame = frames_rgb[len(frames_rgb)//2]

    # Ensure per-example output directory exists
    _ensure_dir(example_dir)

    # Save slow-path CAM on center frame
    if cam_slow is not None:
        heat_slow = cam_slow[0, center_idx].cpu().numpy()  # [H, W]
        heat_slow = cv2.resize(heat_slow, (center_frame.shape[1], center_frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        overlay_slow = _overlay_heatmap(center_frame, heat_slow, alpha=0.45)
        cv2.imwrite(os.path.join(example_dir, f"slow_center_cam_prob{prob:.3f}.png"), cv2.cvtColor(overlay_slow, cv2.COLOR_RGB2BGR))

    # Save fast-path CAM as a short video or sequence of frames
    if cam_fast is not None:
        heat_seq = cam_fast[0].cpu().numpy()  # [T, H, W]
        # map each fast-time heatmap to its approximate raw frame index
        # Build fast indices used by make_slowfast_tensors_cpu
        # (same sampling as in dataset: linear sampling across the window)
        def sample_indices(num_src: int, num_out: int) -> List[int]:
            if num_src <= 0:
                return [0] * num_out
            if num_out <= 1:
                return [0]
            if num_src == num_out:
                return list(range(num_src))
            idx = np.linspace(0, max(0, num_src - 1), num_out)
            return np.round(idx).astype(int).tolist()

        idx_fast = sample_indices(len(frames_rgb), heat_seq.shape[0])
        vis_seq = []
        for t, idx_r in enumerate(idx_fast):
            fr = frames_rgb[idx_r]
            heat = heat_seq[t]
            heat = cv2.resize(heat, (fr.shape[1], fr.shape[0]), interpolation=cv2.INTER_LINEAR)
            ov = _overlay_heatmap(fr, heat, alpha=0.45)
            vis_seq.append(ov)
        # save mp4 and first grid preview
        _save_video(vis_seq, os.path.join(example_dir, f"fast_seq_cam_prob{prob:.3f}.mp4"), fps=cfg.fps // 2 if cfg.fps >= 4 else 8)
        # Also save first/center frame overlays for quick glance
        if vis_seq:
            cv2.imwrite(os.path.join(example_dir, "fast_first.png"), cv2.cvtColor(vis_seq[0], cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(example_dir, "fast_center.png"), cv2.cvtColor(vis_seq[len(vis_seq)//2], cv2.COLOR_RGB2BGR))
        # Analyze fast-path motion (space-time and motion plot)
        try:
            fm_dir = os.path.join(example_dir, 'fast_motion')
            _ensure_dir(fm_dir)
            fast_motion_res = analyze_fastpath_motion(heat_seq, fm_dir, fps=cfg.fps, smooth_window=3, downsample_w=128)
        except Exception as e:
            print(f"[WARN] analyze_fastpath_motion failed: {e}")

    # Save raw center frame for reference
    cv2.imwrite(os.path.join(example_dir, "center_raw.png"), cv2.cvtColor(center_frame, cv2.COLOR_RGB2BGR))

    # Write a small json sidecar with the probability
    meta = {"prob": prob}
    # include fast-path motion result (if computed earlier)
    try:
        if 'fast_motion_res' in locals() and fast_motion_res is not None:
            meta['fast_motion'] = fast_motion_res
    except Exception:
        pass
    # If flow HDF5 is provided, run masked-flow analysis using CAMs
    # Note: allow running even if fast CAM is None so we always get 'none' and 'slow' modes.
    if flow_h5 is not None and start_frame is not None:
        try:
            cam_np = cam_fast[0].cpu().numpy()  # [T, H, W]
        except Exception:
            cam_np = None
        try:
            slow_np = None
            if cam_slow is not None:
                slow_np = cam_slow[0, center_idx].cpu().numpy()  # [H, W]

        except Exception:
            slow_np = None

    # --- Debug metadata: record shapes and min/max before/after normalization ---
        cam_debug = {"fast": None, "slow": None}
        try:
            # fast CAM stats
            if cam_np is None:
                cam_debug["fast"] = {"exists": False}
            else:
                f = np.asarray(cam_np).astype(np.float32)
                f_min = float(np.nanmin(f))
                f_max = float(np.nanmax(f))
                f_shape = list(f.shape)
                # simulate per-frame min-max normalization used in analyze_flow_with_cam
                f_after_min = None
                f_after_max = None
                try:
                    f_n = np.zeros_like(f, dtype=np.float32)
                    for ti in range(f.shape[0]):
                        fr = f[ti]
                        mn = float(np.nanmin(fr))
                        mx = float(np.nanmax(fr))
                        if (mx - mn) > 1e-6:
                            f_n[ti] = (fr - mn) / (mx - mn)
                        else:
                            f_n[ti] = np.zeros_like(fr)
                    f_after_min = float(np.nanmin(f_n))
                    f_after_max = float(np.nanmax(f_n))
                except Exception:
                    f_after_min = None
                    f_after_max = None
                cam_debug["fast"] = {"exists": True, "shape": f_shape, "min_before": f_min, "max_before": f_max, "min_after": f_after_min, "max_after": f_after_max}
        except Exception as e:
            cam_debug["fast"] = {"error": str(e)}

        try:
            # slow CAM stats
            if slow_np is None:
                cam_debug["slow"] = {"exists": False}
            else:
                s = np.asarray(slow_np).astype(np.float32)
                s_min = float(np.nanmin(s))
                s_max = float(np.nanmax(s))
                s_shape = list(s.shape)
                # simulate slow-center normalization
                s_after_min = None
                s_after_max = None
                try:
                    if (np.nanmax(s) - np.nanmin(s)) > 1e-6:
                        s_n = (s - float(np.nanmin(s))) / (float(np.nanmax(s)) - float(np.nanmin(s)))
                        s_after_min = float(np.nanmin(s_n))
                        s_after_max = float(np.nanmax(s_n))
                    else:
                        s_after_min = 0.0
                        s_after_max = 0.0
                except Exception:
                    s_after_min = None
                    s_after_max = None
                cam_debug["slow"] = {"exists": True, "shape": s_shape, "min_before": s_min, "max_before": s_max, "min_after": s_after_min, "max_after": s_after_max}
        except Exception as e:
            cam_debug["slow"] = {"error": str(e)}

        meta["cam_debug"] = cam_debug

        # Run a simplified combined analysis over a larger context (128 frames by default)
        try:
            subdir = os.path.join(example_dir, "combined")
            _ensure_dir(subdir)
            res_comb = analyze_combined_flow_gradcam(flow_h5, cam_np, slow_np, subdir, int(start_frame), window=max(128, len(frames_rgb)), fps=cfg.fps, smooth_window=5, save_plot=True)
            meta['flow_masked'] = {'combined': res_comb}
            if res_comb is not None:
                print(f"[INFO] combined -> len={len(res_comb.get('combined', []))} peaks={res_comb.get('n_peaks')} mean={np.mean(res_comb.get('combined', [])) if res_comb.get('combined') else 0.0:.3f}")
        except Exception as e:
            print(f"[WARN] Combined analyze failed: {e}")

        # Per-example detailed exports: per-pixel CSV, 2D magnitude, images, motion energy
        try:
            flows = _read_flow_h5_segment(flow_h5, int(start_frame), len(frames_rgb))
        except Exception:
            flows = None
        if flows is not None:
            T, Hf, Wf, _ = flows.shape
            # Build per-frame heat at flow resolution using fast CAM if available; otherwise slow center
            heat_t = None
            try:
                if cam_np is not None:
                    frame_to_cam = _map_time_indices(cam_np.shape[0], T)
                    small = cam_np[frame_to_cam]
                    heat_t = np.zeros((T, Hf, Wf), dtype=np.float32)
                    for ti in range(T):
                        heat_t[ti] = cv2.resize(small[ti].astype(np.float32), (Wf, Hf), interpolation=cv2.INTER_LINEAR)
                elif slow_np is not None:
                    h2 = cv2.resize(np.asarray(slow_np).astype(np.float32), (Wf, Hf), interpolation=cv2.INTER_LINEAR)
                    heat_t = np.repeat(h2[None, ...], T, axis=0)
            except Exception:
                heat_t = None

            fusion_dir = os.path.join(example_dir, 'flow_cam_fusion')
            _ensure_dir(fusion_dir)

            # per-pixel CSV
            try:
                _export_flow_cam_csv(flows, heat_t, os.path.join(fusion_dir, 'flow_cam_masked.csv'))
            except Exception as e:
                print(f"[WARN] flow_cam CSV export failed: {e}")

            # 2D magnitude CSV/PNGs
            try:
                mag_arr = np.sqrt(flows[..., 0]**2 + flows[..., 1]**2)
                _export_magnitude_2d(
                    mag_arr,
                    heat_t,
                    out_csv=os.path.join(fusion_dir, 'flow_magnitude_2d.csv'),
                    out_mag_png=os.path.join(fusion_dir, 'flow_magnitude_2d.png'),
                    out_heat_png=os.path.join(fusion_dir, 'cam_2d.png'),
                    out_masked_png=os.path.join(fusion_dir, 'flow_magnitude_2d_masked.png'),
                    agg='mean',
                )
            except Exception as e:
                print(f"[WARN] 2D magnitude export failed: {e}")

            # per-frame raw/masked images (center frame only to limit I/O)
            try:
                _export_flow_images(flows, heat_t, indices=[T//2], out_dir=fusion_dir)
            except Exception as e:
                print(f"[WARN] flow image export failed: {e}")

            # motion energy curve
            try:
                _export_motion_energy(flows, heat_t, os.path.join(fusion_dir, 'motion_energy.csv'))
            except Exception as e:
                print(f"[WARN] motion energy export failed: {e}")

            # fast-path per-frame magnitude curves (raw vs masked)
            try:
                _export_fastpath_flow_curves(
                    flows,
                    cam_np if cam_np is not None else None,
                    out_csv=os.path.join(fusion_dir, 'fastpath_flow_curves.csv'),
                    out_png=os.path.join(fusion_dir, 'fastpath_flow_curves.png'),
                    fps=float(getattr(cfg, 'fps', 32.0))
                )
            except Exception as e:
                print(f"[WARN] fastpath flow curves export failed: {e}")

    with open(os.path.join(example_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def run_analysis(acfg: AnalyzeConfig):
    # Optional: print current analysis config
    try:
        acfg.print("[AnalyzeConfig] Current settings")
    except Exception:
        pass
    # Build baseline cfg from training defaults
    cfg = Config()
    # dataset_dir not used; we operate on a single (video, annotation) pair
    if acfg.positive_label:
        cfg.positive_label = acfg.positive_label
    if acfg.window_size:
        cfg.window_size = int(acfg.window_size)
    if acfg.t_fast:
        cfg.t_fast = int(acfg.t_fast)
    if acfg.alpha:
        cfg.alpha = int(acfg.alpha)
    if acfg.resize_h and acfg.resize_w:
        cfg.resize_h, cfg.resize_w = int(acfg.resize_h), int(acfg.resize_w)
    if acfg.threshold is not None:
        cfg.pos_threshold_eval = float(acfg.threshold)
    if acfg.device:
        cfg.device = acfg.device

    # Resolve checkpoint
    ckpt = acfg.ckpt
    if ckpt is None:
        # prefer local models dir, otherwise notebooks/models
        candidate1 = os.path.join("models", "best_slowfast.pth")
        candidate2 = os.path.join("notebooks", "models", "best_slowfast.pth")
        ckpt = candidate1 if os.path.isfile(candidate1) else candidate2
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    device = torch.device(cfg.device)
    print(f"Using device: {device}")
    model = build_model_slowfast(device, pretrained=False)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Build dataset index (no jitter) to pick pos/neg samples
    # Validate provided paths
    if not os.path.isfile(acfg.video_path):
        raise FileNotFoundError(f"ROI video not found: {acfg.video_path}")
    if not os.path.isfile(acfg.annotation_path):
        raise FileNotFoundError(f"Annotation file not found: {acfg.annotation_path}")

    # Build single-pair list for dataset
    pairs = [(acfg.video_path, acfg.annotation_path)]

    ds = SlowFastDataset(
        pairs=pairs,
        window_size=cfg.window_size,
        stride=cfg.test_stride,
        fps=cfg.fps,
        t_fast=cfg.t_fast,
        alpha=cfg.alpha,
        positive_label=cfg.positive_label,
        resize_hw=(cfg.resize_h, cfg.resize_w),
        normalize=cfg.normalize,
        mean=cfg.mean,
        std=cfg.std,
        epoch_jitter=False,
    )

    # Select multiple positive / negative windows
    soft = ds.soft_labels
    if soft.size == 0:
        raise RuntimeError("No windows extracted from video.")

    pos_thresh = cfg.pos_threshold_eval
    rng = np.random.RandomState(acfg.seed)

    pos_candidates = np.where(soft >= pos_thresh)[0]
    neg_candidates = np.where(soft < pos_thresh)[0]

    # Fallback if none meet threshold: use top-N / bottom-N by soft score
    if pos_candidates.size == 0:
        pos_candidates = np.argsort(-soft)  # descending
    if neg_candidates.size == 0:
        neg_candidates = np.argsort(soft)   # ascending

    def _pick(candidates: np.ndarray, k: int, random_flag: bool) -> List[int]:
        if candidates.size == 0 or k <= 0:
            return []
        if k >= candidates.size:
            return candidates.tolist()
        if random_flag:
            sel = candidates.copy()
            rng.shuffle(sel)
            return sel[:k].tolist()
        # deterministic: keep original order (already threshold order), or sorted by score
        return candidates[:k].tolist()

    pos_selected = _pick(pos_candidates, max(1, acfg.num_pos), acfg.random_sample)
    neg_selected = _pick(neg_candidates, max(1, acfg.num_neg), acfg.random_sample)

    print(f"Selected {len(pos_selected)} positive and {len(neg_selected)} negative windows (threshold={pos_thresh:.3f}).")
    if pos_selected:
        print("Positive indices (soft):", [(int(i), float(soft[i])) for i in pos_selected])
    if neg_selected:
        print("Negative indices (soft):", [(int(i), float(soft[i])) for i in neg_selected])

    # Recover window start for those indices
    def _get_window_frames_for_idx(dataset: SlowFastDataset, idx: int):
        vpath, start, flags, n = dataset.index[idx]
        # mimic __getitem__ window selection (centered window)
        window = dataset.window
        center = min(max(0, start + (window // 2)), n - 1)
        half = window // 2
        read_start = center - half
        if read_start < 0:
            read_start = 0
        if read_start + window > n:
            read_start = max(0, n - window)
        frames = _read_window_frames(vpath, start=read_start, window=window, resize_hw=dataset.resize_hw)
        return frames, vpath, read_start

    out_dir = acfg.out_dir
    _ensure_dir(out_dir)

    # Process positives
    for rank, idx in enumerate(pos_selected, 1):
        t0 = time.time()
        frames_pos, vpath_pos, read_start_pos = _get_window_frames_for_idx(ds, idx)
        example_dir_pos = os.path.join(out_dir, f"pos_idx{idx}_r{rank}")
        flow_h5_pos = _find_flow_h5(vpath_pos)
        analyze_one(model, device, frames_pos, cfg, example_dir_pos, cfg.pos_threshold_eval, flow_h5=flow_h5_pos, start_frame=read_start_pos)
        print(f"[POS {rank}/{len(pos_selected)}] idx={idx} soft={soft[idx]:.3f} processed in {time.time()-t0:.2f}s -> {example_dir_pos}")

    # Process negatives
    for rank, idx in enumerate(neg_selected, 1):
        t1 = time.time()
        frames_neg, vpath_neg, read_start_neg = _get_window_frames_for_idx(ds, idx)
        example_dir_neg = os.path.join(out_dir, f"neg_idx{idx}_r{rank}")
        flow_h5_neg = _find_flow_h5(vpath_neg)
        analyze_one(model, device, frames_neg, cfg, example_dir_neg, cfg.pos_threshold_eval, flow_h5=flow_h5_neg, start_frame=read_start_neg)
        print(f"[NEG {rank}/{len(neg_selected)}] idx={idx} soft={soft[idx]:.3f} processed in {time.time()-t1:.2f}s -> {example_dir_neg}")

if __name__ == "__main__":
    # Edit these values as needed (no argparse)
    user_cfg = AnalyzeConfig(
        ckpt=None,  # auto-resolve if None
        video_path="D:/Master_Arbeit/TSwallowDataset/your_video_roi.mp4",  # <-- set real ROI video
        annotation_path="D:/Master_Arbeit/TSwallowDataset/your_video_annotation.txt",  # <-- set real annotation
        positive_label="none",
        out_dir="runs/gradcam",
        window_size=None,  # e.g., 32
        t_fast=None,       # e.g., 32
        alpha=None,        # e.g., 4
        resize_h=None,     # e.g., 224
        resize_w=None,     # e.g., 224
        threshold=None,    # e.g., 0.5
        num_pos=3,         # number of positive windows to visualize
        num_neg=3,         # number of negative windows to visualize
        random_sample=False,  # set True to randomly sample candidates
        seed=42,
        device=None,       # 'cuda' or 'cpu'
    )
    run_analysis(user_cfg)
