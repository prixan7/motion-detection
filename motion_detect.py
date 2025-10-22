import cv2
import imutils
import os
from datetime import datetime

# --- Choose video source ---
# Set to 0 for webcam, or "sample/footage.mp4" for your video file
VIDEO_SOURCE = "sample/footage.mp4"  # Change to 0 to use webcam

# Create output folder
OUTPUT_DIR = "events"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Open video or webcam
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print(f"❌ Error: Could not open {VIDEO_SOURCE}.")
    exit()

print(f"✅ Source opened: {VIDEO_SOURCE}. Press 'q' to quit.")

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100)

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Warning: No frame captured. Exiting...")
        break

    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)
    fgmask = cv2.medianBlur(fgmask, 5)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"motion_{timestamp}.jpg"), frame)

    cv2.imshow("Motion Detection", frame)
    cv2.imshow("Mask", fgmask)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
