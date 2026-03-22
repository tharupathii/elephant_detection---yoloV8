# Elephant Detection — YOLOv8

A fine-tuned [YOLOv8](https://github.com/ultralytics/ultralytics) model for detecting elephants in images and video.

---

## Project Structure

```
elephant_detection---yoloV8/
├── data/
│   ├── elephant.yaml        # Dataset configuration
│   ├── images/
│   │   ├── train/           # Training images
│   │   ├── val/             # Validation images
│   │   └── test/            # (optional) Test images
│   └── labels/
│       ├── train/           # YOLO-format label files for training images
│       ├── val/             # YOLO-format label files for validation images
│       └── test/            # (optional) label files for test images
├── detect.py                # Inference script
├── train.py                 # Training script
├── utils.py                 # Shared utility helpers
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare your dataset

Organise images and labels under `data/`:

```
data/images/train/   ← training images  (.jpg / .png)
data/images/val/     ← validation images
data/labels/train/   ← matching YOLO .txt label files
data/labels/val/     ← matching YOLO .txt label files
```

Each label file contains one detection per line in YOLO format:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalised to `[0, 1]` relative to the image dimensions.
For elephant detection there is only **one class** (`class_id = 0`).

> **Tip:** The [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html)
> and [LILA BC](https://lila.science/) are good sources for labelled elephant images.

---

## Training

```bash
# Fine-tune the YOLOv8-nano model for 50 epochs (default)
python train.py

# Larger model, more epochs, custom batch size
python train.py --model yolov8s.pt --epochs 100 --batch 16

# Resume an interrupted run
python train.py --resume
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `data/elephant.yaml` | Dataset config |
| `--model` | `yolov8n.pt` | Pretrained weights to fine-tune |
| `--epochs` | `50` | Training epochs |
| `--imgsz` | `640` | Input image size |
| `--batch` | `16` | Batch size (`-1` = AutoBatch) |
| `--device` | auto | `cpu`, `0`, `0,1`, … |
| `--resume` | off | Resume from last checkpoint |

Trained weights are saved to `runs/detect/elephant/weights/best.pt`.

---

## Inference

```bash
# Single image
python detect.py --source path/to/elephant.jpg

# Directory of images
python detect.py --source path/to/images/

# Video file
python detect.py --source path/to/video.mp4

# Use a specific checkpoint, lower confidence threshold
python detect.py --source image.jpg \
                 --model runs/detect/elephant/weights/best.pt \
                 --conf 0.3
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | *(required)* | Image / directory / video path |
| `--model` | `runs/detect/elephant/weights/best.pt` | Weights file |
| `--conf` | `0.25` | Confidence threshold |
| `--iou` | `0.45` | NMS IoU threshold |
| `--imgsz` | `640` | Inference image size |
| `--save` / `--no-save` | save on | Save annotated outputs |
| `--show` | off | Display results in a window |

Annotated outputs are saved to `runs/detect/predict/`.

---

## Model Variants

| Model | Speed | Accuracy |
|-------|-------|---------|
| `yolov8n.pt` | Fastest | Good |
| `yolov8s.pt` | Fast | Better |
| `yolov8m.pt` | Medium | Best for most use cases |
| `yolov8l.pt` | Slow | Higher accuracy |
| `yolov8x.pt` | Slowest | Highest accuracy |

Start with `yolov8n.pt` for quick experiments and scale up as needed.

---

## License

This project uses [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
which is released under the [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html).
