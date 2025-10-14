"""Helper script to prepare model & YOLO pose weights.

This does not automatically download proprietary checkpoints.
Edit the URLS or copy your files manually.

Example usage (PowerShell):
  python scripts/get_models.py --yolo
  python scripts/get_models.py --custom slowfastHub_ws32_e20.pth C:/path/to/local.ckpt
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path
import urllib.request

YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt"

def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[DOWNLOAD] {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)

def copy_local(src: Path, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[COPY] {src} -> {dest}")
    shutil.copy2(src, dest)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", action="store_true", help="Download YOLO pose weights yolov8n-pose.pt")
    ap.add_argument("--custom", nargs=2, metavar=("DEST_NAME", "SRC"), help="Copy a local checkpoint into models dir")
    ap.add_argument("--models-dir", default="BYS_XAISwallow/models", help="Target models directory")
    args = ap.parse_args()

    mdir = Path(args.models_dir)
    if args.yolo:
        download(YOLO_URL, mdir / "yolov8n-pose.pt")
    if args.custom:
        dest_name, src = args.custom
        copy_local(Path(src), mdir / dest_name)
    if not args.yolo and not args.custom:
        ap.print_help()

if __name__ == "__main__":
    main()