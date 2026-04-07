from pathlib import Path
import argparse
import platform

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect elephants in a local video using YOLOv8")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--model", default="yolov8n.pt", help="Path to YOLO model file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--save", action="store_true", help="Save annotated output video")
    parser.add_argument("--show", action="store_true", help="Show live preview while processing")
    return parser.parse_args()


def _play_alert_if_supported() -> None:
    if platform.system() != "Windows":
        return

    try:
        import winsound

        winsound.Beep(2000, 350)
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)

    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return

    model = YOLO(args.model)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Unable to open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_writer = None
    output_path = None

    if args.save:
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"detected_{video_path.stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0
    elephant_frames = 0

    print("Processing video. Press 'q' to stop preview window.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model.predict(source=frame, conf=args.conf, verbose=False)
        result = results[0]

        elephant_detected = False
        for box in result.boxes:
            cls = int(box.cls[0])
            if model.names[cls] == "elephant":
                elephant_detected = True
                break

        if elephant_detected:
            elephant_frames += 1

        annotated = result.plot()

        if output_writer is not None:
            output_writer.write(annotated)

        if args.show:
            cv2.imshow("Elephant Detection (Video)", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    cap.release()
    if output_writer is not None:
        output_writer.release()
    cv2.destroyAllWindows()

    if elephant_frames > 0:
        _play_alert_if_supported()
        print(f"ELEPHANT DETECTED in {elephant_frames} frame(s) out of {frame_count}.")
    else:
        print(f"No elephant detected across {frame_count} processed frame(s).")

    if output_path is not None:
        print(f"Saved annotated video: {output_path}")


if __name__ == "__main__":
    main()
