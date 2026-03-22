"""
train.py – Train a YOLOv8 model for elephant detection.

Usage
-----
Basic (fine-tune the nano model):
    python train.py

Custom epochs / image size / batch:
    python train.py --epochs 100 --imgsz 640 --batch 16

Resume an interrupted run:
    python train.py --resume

Full option list:
    python train.py --help
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from utils import verify_dataset


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLOv8 elephant detection model."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/elephant.yaml",
        help="Path to the dataset YAML config (default: data/elephant.yaml).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help=(
            "YOLOv8 model variant to fine-tune. Use a pretrained checkpoint "
            "(e.g. yolov8n.pt, yolov8s.pt, yolov8m.pt) or a custom .yaml "
            "architecture to train from scratch (default: yolov8n.pt)."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size in pixels (default: 640).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size; use -1 for AutoBatch (default: 16).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help=(
            "Training device: '' auto-selects GPU if available, "
            "or specify 'cpu', '0', '0,1', etc."
        ),
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Root directory for saving training runs (default: runs/detect).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="elephant",
        help="Sub-directory name for this run (default: elephant).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early-stopping patience in epochs (default: 20).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader workers (default: 8).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Sanity-check the dataset layout (non-fatal; training will still run)
    if not verify_dataset(args.data):
        print(
            "[WARNING] Dataset verification failed. "
            "Please ensure images and labels are in place before training."
        )

    # Load model
    model = YOLO(args.model)

    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device if args.device else None,
        project=args.project,
        name=args.name,
        resume=args.resume,
        patience=args.patience,
        workers=args.workers,
        verbose=True,
    )

    print("\n[INFO] Training complete.")
    print(f"[INFO] Results saved to: {Path(args.project) / args.name}")

    # Evaluate on the validation set
    try:
        metrics = model.val()
        print(f"\n[INFO] Validation mAP50   : {metrics.box.map50:.4f}")
        print(f"[INFO] Validation mAP50-95: {metrics.box.map:.4f}")
    except Exception as exc:
        print(
            f"\n[WARNING] Validation could not be completed: {exc}\n"
            "Check that your validation images and labels are in place "
            "as described in data/elephant.yaml."
        )


if __name__ == "__main__":
    main()
