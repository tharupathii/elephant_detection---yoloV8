"""
detect.py – Run elephant detection inference with a trained YOLOv8 model.

Usage
-----
Detect in a single image:
    python detect.py --source path/to/image.jpg

Detect in a directory of images:
    python detect.py --source path/to/images/

Run on a video file:
    python detect.py --source path/to/video.mp4

Use a specific model checkpoint:
    python detect.py --source image.jpg --model runs/detect/elephant/weights/best.pt

Adjust confidence / IoU thresholds:
    python detect.py --source image.jpg --conf 0.4 --iou 0.5

Full option list:
    python detect.py --help
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 elephant detection on images or video."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to an image, directory of images, or video file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="runs/detect/elephant/weights/best.pt",
        help=(
            "Path to a trained YOLOv8 weights file "
            "(default: runs/detect/elephant/weights/best.pt)."
        ),
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum confidence threshold for detections (default: 0.25).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for Non-Maximum Suppression (default: 0.45).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size in pixels (default: 640).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help=(
            "Inference device: '' auto-selects GPU if available, "
            "or specify 'cpu', '0', etc."
        ),
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Root directory for saving detection results (default: runs/detect).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="predict",
        help="Sub-directory name for this inference run (default: predict).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save annotated images / video (enabled by default).",
    )
    parser.add_argument(
        "--no-save",
        dest="save",
        action="store_false",
        help="Disable saving annotated outputs.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display results in a window during inference.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {model_path}\n"
            "Train the model first with:  python train.py"
        )

    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")

    model = YOLO(str(model_path))

    results = model.predict(
        source=str(source_path),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device if args.device else None,
        project=args.project,
        name=args.name,
        save=args.save,
        show=args.show,
        verbose=True,
    )

    # Summarize detections
    total_elephants = 0
    for r in results:
        count = len(r.boxes)
        total_elephants += count

    print(f"\n[INFO] Detection complete.")
    print(f"[INFO] Total elephants detected across all inputs: {total_elephants}")
    if args.save:
        print(f"[INFO] Annotated results saved to: {Path(args.project) / args.name}")


if __name__ == "__main__":
    main()
