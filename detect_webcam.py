#--this is for detect using the webcam

from ultralytics import YOLO
import cv2
import winsound
import threading
import time
import os

# ----- LOAD AI MODEL -----
model = YOLO("yolov8n.pt")  # Pretrained YOLOv8 nano model

# ----- OPEN WEBCAM -----
cap = cv2.VideoCapture(0)  # 0 = default laptop webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Starting real-time elephant detection. Press 'q' to quit.")

ALERT_WAV_PATH = r"C:\Windows\Media\Windows Exclamation.wav"
BEEP_COOLDOWN_SEC = 1.0
last_beep_time = 0.0


def play_beep_async():
    # Play a real WAV file through system audio; this is more reliable than Beep.
    if os.path.exists(ALERT_WAV_PATH):
        winsound.PlaySound(ALERT_WAV_PATH, winsound.SND_FILENAME)
    else:
        winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)

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
        now = time.time()
        if now - last_beep_time >= BEEP_COOLDOWN_SEC:
            # Run tone beep in a background thread so frame updates stay responsive.
            threading.Thread(target=play_beep_async, daemon=True).start()
            last_beep_time = now

    # ----- SHOW CAMERA FEED -----
    annotated_frame = results[0].plot()
    cv2.imshow("Elephant Detection", annotated_frame)

    # Press 'q'/'Q' or ESC to quit (window must be focused).
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q'), 27):
        break

    # Also stop if the OpenCV window is closed from the title bar.
    if cv2.getWindowProperty("Elephant Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
