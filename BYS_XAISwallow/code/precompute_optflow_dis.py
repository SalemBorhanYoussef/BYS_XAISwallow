
#!/usr/bin/env python3
"""
precompute_optflow_dis.py

Berechnet optical flow (DIS) zwischen aufeinanderfolgenden Frames von Videos
oder Frame-Ordnern und speichert die Flows platzoptimiert ab.

Features:
- DIS optical flow (fall-back auf mehrere OpenCV APIs)
- Optionen: downsample, speichern als float16 (.npz) oder quantisiert int16 in HDF5 (gzip)
- Schreibt pro Video eine Datei (besser als viele kleine Dateien)
- Fortschrittsanzeige und einfache Abschätzung der Speichergröße

Empfehlung (für große Mengen): Halbierung der Auflösung (z.B. 224->112) + quantisierte int16 Speicherung
mit HDF5 gzip führt zu großen Einsparungen und guter Kompression.

Benötigte Pakete: opencv-contrib-python, numpy, h5py (optional für HDF5), tqdm

Usage example:
python precompute_optflow_dis.py --input path/to/video_or_frame_dir --output runs/optflow --backend h5 --dtype int16 --downscale 0.5

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Union, Tuple
import re
import tempfile
import shutil
import sys
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

try:
    import h5py
    H5PY_AVAILABLE = True
except Exception:
    H5PY_AVAILABLE = False


def get_dis_flow():
    """Versuche verschiedene Aufrufe, um DIS Optical Flow zu erhalten.
    Liefert eine Instanz, oder wirft eine Exception, wenn nicht verfügbar.
    """
    # Verschiedene OpenCV-Versionen bieten unterschiedliche APIs
    if hasattr(cv2, 'optflow'):
        try:
            return cv2.optflow.createOptFlow_DIS()
        except Exception:
            pass
    # Fallback
    if hasattr(cv2, 'DISOpticalFlow_create'):
        try:
            return cv2.DISOpticalFlow_create()
        except Exception:
            pass
    raise RuntimeError("DIS Optical Flow API nicht gefunden. Bitte opencv-contrib-python installieren.")


def read_frame_from_source(cap_or_dir, idx=None):
    """Lese Frame entweder aus VideoCapture oder aus einem Verzeichnis (numerische Dateinamen).
    Wenn cap_or_dir ist cv2.VideoCapture -> idx wird ignoriert.
    """
    # Note: hasattr check because cv2.VideoCapture is a class; isinstance may fail across cv2 builds
    if hasattr(cap_or_dir, 'read') and callable(cap_or_dir.read):
        ret, frame = cap_or_dir.read()
        if not ret:
            return None
        return frame
    else:
        # directory of frames
        p = cap_or_dir / idx
        if not p.exists():
            return None
        return cv2.imread(str(p))


def natural_sort_key(path: Union[str, Path]) -> List[Union[int, str]]:
    """Return a sort key that handles numeric filename parts naturally.
    Example: frame_2.png < frame_10.png
    """
    s = str(path)
    parts = re.split(r"(\d+)", s)
    key = [int(p) if p.isdigit() else p.lower() for p in parts]
    return key


def maybe_gray_and_resize(frame, downscale: float):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if downscale != 1.0:
        h, w = frame_gray.shape[:2]
        new_w = max(1, int(w * downscale))
        new_h = max(1, int(h * downscale))
        frame_gray = cv2.resize(frame_gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return frame_gray


def estimate_size(frames: int, h: int, w: int, channels: int = 2, dtype_bytes: int = 4, compression_ratio: float = 1.0):
    # Rohgröße in Bytes, dann Kompression
    raw = frames * h * w * channels * dtype_bytes
    return raw / compression_ratio


def save_h5_int16(output_file: Path, total_pairs: int, h: int, w: int, scale: float = 64.0, compression_level: int = 4):
    # legacy: keep for compatibility, but prefer save_h5_dataset
    return save_h5_dataset(output_file, total_pairs, h, w, dtype_str='int16', scale=scale, compression_level=compression_level)


def save_h5_dataset(output_file: Path, total_pairs: int, h: int, w: int,
                    dtype_str: str = 'int16', scale: float = 64.0, compression_level: int = 4):
    """Create an HDF5 file with a 'flow' dataset.

    dtype_str: 'int16' (quantized with scale attr) or 'float16' (raw float16 values)
    Returns (file, dataset).
    """
    f = h5py.File(str(output_file), 'w')
    if dtype_str == 'int16':
        ds = f.create_dataset('flow', shape=(total_pairs, h, w, 2), dtype=np.int16,
                              compression='gzip', compression_opts=compression_level,
                              chunks=(1, max(1, h//4), max(1, w//4), 2))
        ds.attrs['scale'] = float(scale)
        ds.attrs['dtype'] = 'int16'
    elif dtype_str == 'float16':
        ds = f.create_dataset('flow', shape=(total_pairs, h, w, 2), dtype=np.float16,
                              compression='gzip', compression_opts=compression_level,
                              chunks=(1, max(1, h//4), max(1, w//4), 2))
        # For float storage we set scale=1.0 for downstream compatibility
        ds.attrs['scale'] = 1.0
        ds.attrs['dtype'] = 'float16'
    else:
        f.close()
        raise ValueError(f"Unsupported dtype_str for HDF5 backend: {dtype_str}")
    return f, ds


def process_video_or_frames(src: Path, out_dir: Path, backend: str = 'h5', dtype: str = 'int16', downscale: float = 1.0,
                            scale: float = 64.0, compression_level: int = 4, max_frames: int | None = None):
    """Verarbeitet ein Video (Datei) oder einen Frame-Ordner (mit numerischen Dateinamen).

    backend: 'h5' (HDF5 int16), 'npz' (float16 npz per video)
    dtype: 'int16' oder 'float16'
    scale: für int16 -> flow_quant = np.round(flow * scale).astype(np.int16)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    name = src.stem
    out_file = out_dir / f"{name}.h5" if backend == 'h5' else out_dir / f"{name}.npz"

    # Determine frames count and whether src is video or folder
    if src.is_dir():
        # collect frame filenames sorted (natural sort)
        frames = [p for p in src.iterdir() if p.suffix.lower() in ['.jpg', '.png', '.jpeg', '.bmp']]
        frames = sorted(frames, key=natural_sort_key)
        total_frames = len(frames)
        frame_source = frames
        is_video = False
    else:
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError(f"Video {src} konnte nicht geöffnet werden")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_source = cap
        is_video = True

    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    total_pairs = max(0, total_frames - 1)

    if total_pairs <= 0:
        print(f"Keine Paare in {src}")
        if is_video:
            frame_source.release()
        return

    # read first frame
    if is_video:
        ret, prev_frame = frame_source.read()
        if not ret:
            frame_source.release()
            raise RuntimeError("Fehler beim Lesen des ersten Frames")
        prev_gray = maybe_gray_and_resize(prev_frame, downscale)
    else:
        prev_frame = cv2.imread(str(frame_source[0]))
        prev_gray = maybe_gray_and_resize(prev_frame, downscale)

    h, w = prev_gray.shape[:2]

    # Prepare storage
    if backend == 'h5' and not H5PY_AVAILABLE:
        raise RuntimeError("h5py nicht installiert. Installiere h5py oder wähle backend=npz")

    if backend == 'h5':
        # honor the requested dtype for HDF5 storage
        if dtype not in ('int16', 'float16'):
            raise ValueError("For backend='h5' dtype must be 'int16' or 'float16'")
        f, ds = save_h5_dataset(out_file, total_pairs, h, w, dtype_str=dtype, scale=scale, compression_level=compression_level)
    else:
        # Use memmap for large datasets to avoid using too much RAM.
        dtype_np = np.float16 if dtype == 'float16' else np.float32
        # estimate byte size
        est_bytes = total_pairs * h * w * 2 * np.dtype(dtype_np).itemsize
        memmap_threshold = 200 * 1024 ** 2  # 200 MB
        if est_bytes > memmap_threshold:
            tf = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
            tf.close()
            memmap_path = Path(tf.name)
            flows = np.memmap(str(memmap_path), dtype=dtype_np, mode='w+', shape=(total_pairs, h, w, 2))
            use_memmap = True
        else:
            flows = np.empty((total_pairs, h, w, 2), dtype=dtype_np)
            use_memmap = False

    # DIS flow engine
    flow_engine = get_dis_flow()

    # Write pairs using a zero-based pair_index for clarity
    pair_index = 0
    pbar = tqdm(total=total_pairs, desc=f"{name}")
    while pair_index < total_pairs:
        # read next frame (we already have prev_gray loaded)
        if is_video:
            ret, frame = frame_source.read()
            if not ret:
                break
        else:
            frame = cv2.imread(str(frame_source[pair_index + 1]))
            if frame is None:
                break

        gray = maybe_gray_and_resize(frame, downscale)

        # compute flow (returns 2-channel float32)
        flow = flow_engine.calc(prev_gray, gray, None)

        if backend == 'h5':
            if dtype == 'int16':
                # quantize to int16 (clip to avoid overflow)
                q = np.round(flow * scale)
                q = np.clip(q, -32767, 32767).astype(np.int16)
                ds[pair_index, :, :, :] = q
            else:
                # store as float16 directly
                ds[pair_index, :, :, :] = flow.astype(np.float16)
        else:
            if dtype == 'float16':
                flows[pair_index] = flow.astype(np.float16)
            else:
                flows[pair_index] = flow.astype(np.float32)

        prev_gray = gray
        pair_index += 1
        pbar.update(1)

    pbar.close()

    if is_video:
        frame_source.release()

    if backend == 'h5':
        # close and report
        try:
            dtype_written = ds.attrs.get('dtype', 'unknown')
            reported_scale = ds.attrs.get('scale', scale)
        except Exception:
            dtype_written = 'unknown'
            reported_scale = scale
        f.close()
        if dtype_written == 'int16':
            print(f"Gespeichert: {out_file} (dtype=int16, scale={reported_scale}, gzip={compression_level})")
        else:
            print(f"Gespeichert: {out_file} (dtype={dtype_written}, gzip={compression_level})")
    else:
        # If we used a memmap, persist it as an uncompressed .npy for speed and low RAM.
        if use_memmap:
            final_npy = out_file.with_suffix('.npy')
            # memmap is already on disk at memmap_path; rename to target
            try:
                shutil.move(str(memmap_path), str(final_npy))
                print(f"Gespeichert (uncompressed memmap): {final_npy} (dtype={dtype})")
            except Exception:
                # fallback: save via numpy.save (this will load into memory in worst case)
                np.save(str(final_npy), flows)
                print(f"Gespeichert (fallback): {final_npy} (dtype={dtype})")
        else:
            # small enough to compress
            np.savez_compressed(str(out_file), flow=flows)
            print(f"Gespeichert: {out_file} (dtype={dtype})")


def load_h5_flow(path: Union[str, Path]) -> Tuple[h5py.File, 'h5py.Dataset', float]:
    """Öffnet eine HDF5-Flow-Datei und gibt (file, dataset, scale) zurück.

    Beispiel zum Dequantisieren:
        f, ds, scale = load_h5_flow('video.h5')
        flow = ds[0].astype(np.float32) / scale
    Achtung: ds ist ein h5py.Dataset - nicht die ganze Datei in den Speicher laden, iteriere in Blöcken.
    """
    if not H5PY_AVAILABLE:
        raise RuntimeError('h5py nicht verfügbar')
    f = h5py.File(str(path), 'r')
    ds = f['flow']
    # If dataset was quantized int16 it will have scale attr set >1.0
    scale = ds.attrs.get('scale', 1.0)
    return f, ds, float(scale)


@dataclass
class Config:
    # Bearbeite diese Werte direkt im Code (kein CLI nötig)
    input: str = 'path/to/video_or_frame_dir'
    output: str = 'runs/optflow'
    backend: str = 'h5'  # 'h5' oder 'npz'
    dtype: str = 'float16'  # 'int16' oder 'float16'
    downscale: float = 1.0
    scale: float = 64.0
    compression: int = 4
    max_frames: Optional[int] = None


# Default configuration: ändere diese Werte direkt im Skript
DEFAULT_CONFIG = Config()


def pretty_bytes(n: float) -> str:
    for unit in ('B','KB','MB','GB','TB'):
        if abs(n) < 1024.0:
            return f"{n:3.2f}{unit}"
        n /= 1024.0
    return f"{n:.2f}PB"


def main():
    # Benutze DEFAULT_CONFIG (ändere Werte direkt oben im Skript)
    cfg = DEFAULT_CONFIG
    inp = Path(cfg.input)
    out = Path(cfg.output)
    out.mkdir(parents=True, exist_ok=True)

    # estimate
    # find a sample frame to get size
    if inp.is_dir():
        sample = next((p for p in inp.iterdir() if p.suffix.lower() in ['.jpg', '.png', '.jpeg', '.bmp']), None)
        if sample is None:
            print("Kein Bild im Input-Ordner gefunden.")
            sys.exit(1)
        tmp = cv2.imread(str(sample))
    else:
        cap = cv2.VideoCapture(str(inp))
        if not cap.isOpened():
            print("Video konnte nicht geöffnet werden")
            sys.exit(1)
        ret, tmp = cap.read()
        cap.release()
        if not ret:
            print("Konnte ersten Frame nicht lesen")
            sys.exit(1)

    tmp_gray = maybe_gray_and_resize(tmp, cfg.downscale)
    h, w = tmp_gray.shape[:2]

    # frames count
    if inp.is_dir():
        total_frames = len([p for p in inp.iterdir() if p.suffix.lower() in ['.jpg', '.png', '.jpeg', '.bmp']])
    else:
        cap = cv2.VideoCapture(str(inp))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    total_pairs = max(0, total_frames - 1)

    # dtype bytes estimate
    if cfg.dtype == 'float16':
        dtype_bytes = 2
    else:
        # we store int16 physically (2 bytes)
        dtype_bytes = 2

    # very conservative compression ratio estimate
    compression_ratio = 3.0 if cfg.backend == 'h5' else 2.0
    est = estimate_size(total_pairs, h, w, channels=2, dtype_bytes=dtype_bytes, compression_ratio=compression_ratio)
    print(f"Input: {inp} -> frames={total_frames}, pairs={total_pairs}, frame-size={w}x{h}")
    print(f"Geschätzter Speicher (Backend={cfg.backend}, dtype={cfg.dtype}): ~{pretty_bytes(est)} (Kompressionsannahme ~{compression_ratio}x)")

    # Start processing (Ändere Parameter oben im Skript in DEFAULT_CONFIG)
    process_video_or_frames(inp, out, backend=cfg.backend, dtype=cfg.dtype, downscale=cfg.downscale,
                            scale=cfg.scale, compression_level=cfg.compression, max_frames=cfg.max_frames)


def find_roi_videos(dataset_dir: Union[str, Path]) -> List[Path]:
    """Findet rekursiv alle ROI-Videos im Dataset-Ordner.

    Sucht nach Dateinamen, die 'roi' enthalten und gängige Video-Extensions besitzen.
    """
    dataset_dir = Path(dataset_dir)
    exts = ('.mp4', '.avi', '.mov', '.mkv')
    matches: List[Path] = []
    for ext in exts:
        matches.extend(list(dataset_dir.rglob(f"*roi*{ext}")))
    # Remove duplicates and sort naturally
    unique = sorted(set(matches), key=natural_sort_key)
    return unique


def batch_precompute_from_dataset(dataset_dir: Union[str, Path], output_root: Union[str, Path] | None = None,
                                  cfg: Optional[Config] = None, inplace: bool = False):
    """Durchläuft alle ROI-Videos im dataset_dir und berechnet für jedes die Optical-Flow Datei.

    Standard: wenn inplace=True, wird die Flow-Datei direkt in dasselbe Verzeichnis wie das ROI-Video geschrieben.
    Andernfalls wird die Ausgabe unter output_root/<relpath_to_video_dir>/<video_stem>.<backend_ext> gespeichert.
    """
    cfg = cfg or DEFAULT_CONFIG
    dataset_dir = Path(dataset_dir)
    output_root = None if output_root is None else Path(output_root)
    videos = find_roi_videos(dataset_dir)
    if not videos:
        print(f"Keine ROI-Videos im Verzeichnis {dataset_dir} gefunden.")
        return

    print(f"Gefundene ROI-Videos: {len(videos)}. Beginne Precompute...")
    for v in tqdm(videos, desc="Videos", unit="video"):
        try:
            rel = v.relative_to(dataset_dir)
        except Exception:
            rel = v.name
        # Determine output directory
        if inplace:
            out_dir = v.parent
        else:
            # Preserve subfolder structure under output_root
            rel_parent = Path(rel).parent
            if output_root is None:
                # fallback to video parent
                out_dir = v.parent
            else:
                out_dir = output_root / rel_parent
        out_dir.mkdir(parents=True, exist_ok=True)

        # Choose output filename (same stem)
        backend = cfg.backend
        ext = '.h5' if backend == 'h5' else '.npz'
        out_file = out_dir / f"{v.stem}{ext}"

        # Skip if already exists
        if out_file.exists():
            print(f"Skip existing: {out_file}")
            continue

        # Run computation for this video
        try:
            process_video_or_frames(v, out_dir, backend=cfg.backend, dtype=cfg.dtype, downscale=cfg.downscale,
                                    scale=cfg.scale, compression_level=cfg.compression, max_frames=cfg.max_frames)
        except Exception as e:
            print(f"Fehler bei {v}: {e}")

    print("Precompute abgeschlossen.")


if __name__ == '__main__':
    main()
