"""Utility helpers for the elephant detection project."""

import os
from pathlib import Path


def ensure_dirs(*paths: str) -> None:
    """Create directories (including parents) if they do not already exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def verify_dataset(data_yaml: str) -> bool:
    """
    Perform a basic sanity-check on the dataset layout described by *data_yaml*.

    Returns True when the train and val image directories exist and contain at
    least one image file, otherwise prints a warning and returns False.
    """
    import yaml  # imported lazily so the module is usable without PyYAML

    with open(data_yaml, "r") as fh:
        cfg = yaml.safe_load(fh)

    root = Path(cfg.get("path", "."))
    ok = True
    for split in ("train", "val"):
        split_path = cfg.get(split)
        if not split_path:
            continue
        img_dir = root / split_path
        if not img_dir.exists():
            print(f"[WARNING] {split} image directory not found: {img_dir}")
            ok = False
            continue
        images = [
            f for f in img_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        ]
        if not images:
            print(f"[WARNING] No images found in {split} directory: {img_dir}")
            ok = False
        else:
            print(f"[INFO] {split}: {len(images)} image(s) found in {img_dir}")

    return ok


def get_latest_run(project: str = "runs/detect") -> Path | None:
    """Return the path of the most-recently created run directory."""
    project_dir = Path(project)
    if not project_dir.exists():
        return None
    runs = sorted(
        (p for p in project_dir.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return runs[0] if runs else None
