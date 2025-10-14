"""Utility helpers for repository cleanup.

This script does NOT rewrite history automatically. It:
 1. Lists large tracked files (default extensions & size threshold)
 2. Optionally moves them to an external folder (keeping a manifest)
 3. Prints suggested git-filter-repo commands if you decide to purge

Usage (PowerShell / bash):
  python scripts/cleanup_large_files.py --list
  python scripts/cleanup_large_files.py --move-out ./_extracted_large

After verifying you may manually run (outside this script):
  git filter-repo --path <path/to/file> --invert-paths
Repeat for each path or craft a combined command.

NOTE: Install git-filter-repo first: https://github.com/newren/git-filter-repo
"""

from __future__ import annotations
import argparse
import subprocess
import json
import shutil
from pathlib import Path

DEFAULT_EXTS = {".mp4", ".ravi", ".pth", ".pt"}

def git_tracked_files() -> list[Path]:
    try:
        out = subprocess.check_output(["git", "ls-files"], text=True)
        return [Path(p) for p in out.strip().splitlines() if p.strip()]
    except Exception as ex:
        print(f"[WARN] git ls-files failed: {ex}")
        return []

def human(n: int) -> str:
    for unit in ["B","KB","MB","GB"]:
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024
    return f"{n}B"

def list_large(exts: set[str], min_size_mb: float):
    files = git_tracked_files()
    min_bytes = min_size_mb * 1024 * 1024
    rows = []
    for p in files:
        if p.suffix.lower() in exts and p.exists():
            size = p.stat().st_size
            if size >= min_bytes:
                rows.append((size, p))
    rows.sort(reverse=True)
    return rows

def move_out(rows, target: Path):
    target.mkdir(parents=True, exist_ok=True)
    manifest = []
    for size, p in rows:
        rel = p
        dest = target / rel.name
        print(f"[MOVE] {rel} -> {dest}")
        shutil.move(str(rel), dest)
        manifest.append({"original": str(rel), "moved_to": str(dest), "size": size})
    with open(target / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[INFO] Manifest written: {target / 'manifest.json'}")

def suggest_filter_repo(rows):
    if not rows:
        return
    print("\nSuggested git-filter-repo commands (one per path):")
    for _, p in rows:
        print(f"git filter-repo --path {p} --invert-paths")
    print("\nAfter running, force-push: git push --force origin main (coordinate with collaborators!)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exts", nargs="*", default=sorted(DEFAULT_EXTS), help="File extensions to scan")
    ap.add_argument("--min-size-mb", type=float, default=1.0, help="Minimum size in MB")
    ap.add_argument("--list", action="store_true", help="List large files")
    ap.add_argument("--move-out", type=Path, help="Move matching files out of repo (working tree only)")
    args = ap.parse_args()

    rows = list_large(set(e.lower() for e in args.exts), args.min_size_mb)
    if args.list or (not args.list and not args.move_out):
        if not rows:
            print("No large tracked files found meeting criteria.")
        else:
            print(f"Found {len(rows)} large tracked file(s):")
            for size, p in rows:
                print(f"  {human(size):>8}  {p}")
    if args.move_out:
        if not rows:
            print("Nothing to move.")
        else:
            move_out(rows, args.move_out)
    suggest_filter_repo(rows)

if __name__ == "__main__":
    main()
