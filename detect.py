from ultralytics import YOLO
import cv2
import winsound

# ----- LOAD AI MODEL -----
model = YOLO("yolov8n.pt")  # Pretrained YOLOv8 nano model

# ----- OPEN WEBCAM -----
cap = cv2.VideoCapture(0)  # 0 = default laptop webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Starting real-time elephant detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ----- DETECTION -----
    results = model(frame)

    elephant_detected = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "elephant":
                elephant_detected = True

    # ----- ALERT -----
    if elephant_detected:
        print("ELEPHANT DETECTED!")
        winsound.Beep(2000, 500)  # Alert sound

    # ----- SHOW CAMERA FEED -----
    annotated_frame = results[0].plot()
    cv2.imshow("Elephant Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
