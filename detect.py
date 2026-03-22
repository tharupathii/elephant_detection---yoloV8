#--this is for detect using a jpg image
#--command: py detect.py --image elephant.jpg --show


from pathlib import Path
import argparse
import cv2
import winsound
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect elephants in a JPG image using YOLOv8")
    parser.add_argument("--image", required=True, help="Path to input JPG/PNG image")
    parser.add_argument("--model", default="yolov8n.pt", help="Path to YOLO model file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--show", action="store_true", help="Show result image in a window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    model = YOLO(args.model)
    results = model.predict(source=str(image_path), conf=args.conf, verbose=False)
    result = results[0]

    elephant_boxes = []
    for box in result.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        if label == "elephant":
            conf = float(box.conf[0])
            elephant_boxes.append(conf)

    if elephant_boxes:
        print(f"ELEPHANT DETECTED! Count: {len(elephant_boxes)}")
        print("Confidences:", ", ".join(f"{c:.2f}" for c in elephant_boxes))
        winsound.Beep(2000, 500)
    else:
        print("No elephant detected in this image.")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"detected_{image_path.name}"

    annotated = result.plot()
    cv2.imwrite(str(output_path), annotated)
    print(f"Saved result image: {output_path}")

    if args.show:
        cv2.imshow("Elephant Detection", annotated)
        print("Press any key to close image window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
