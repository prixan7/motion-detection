import cv2
import imutils
import os
from datetime import datetime

# ---------------------------
# CONFIGURATION
# ---------------------------
VIDEO_SOURCE = "sample/footage.mp4"  # 0 = webcam, or video path
OUTPUT_DIR = "events"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILE = "motion_log.txt"
MIN_AREA = 500  # smaller to capture tiny motion
FRAME_WIDTH = 800
ROI = None  # Set to (x, y, w, h) to detect motion only in that region

SAVE_VIDEO = True  # Save short video clips of motion
VIDEO_CLIP_LENGTH = 5  # seconds

# ---------------------------
# UTILITY FUNCTIONS
# ---------------------------
def log_motion(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as log:
        log.write(f"{timestamp} - {message}\n")
    print(f"{timestamp} - {message}")

def save_snapshot(frame, prefix="motion"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{prefix}_{timestamp}.jpg"
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), frame)
    print(f"💾 Snapshot saved: {filename}")

def initialize_video_writer(frame, prefix="clip"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(OUTPUT_DIR, f"{prefix}_{timestamp}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 20
    height, width = frame.shape[:2]
    return cv2.VideoWriter(filename, fourcc, fps, (width, height))

# ---------------------------
# INITIALIZE VIDEO SOURCE
# ---------------------------
cap = cv2.VideoCapture(VIDEO_SOURCE)
if VIDEO_SOURCE == 0:
    cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(f"❌ Cannot open {VIDEO_SOURCE}")
    exit()

print(f"✅ Video source opened: {VIDEO_SOURCE}")
print("Press 'q' to quit, 's' to save manual snapshot")

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100)

motion_clip_writer = None
motion_timer = 0
fps = 20  # approximate FPS for video clip

# ---------------------------
# MAIN LOOP
# ---------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ End of video or no frame captured.")
        break

    frame = imutils.resize(frame, width=FRAME_WIDTH)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply ROI if defined
    roi_frame = frame
    if ROI:
        x, y, w, h = ROI
        roi_frame = frame[y:y+h, x:x+w]

    fgmask = fgbg.apply(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY))
    fgmask = cv2.medianBlur(fgmask, 5)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_detected = False
    person_id = 1

    for contour in contours:
        if cv2.contourArea(contour) > MIN_AREA:
            motion_detected = True

            # Thin contour (green)
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 1)

            # Thin convex hull (red)
            hull = cv2.convexHull(contour)
            cv2.drawContours(frame, [hull], -1, (0, 0, 255), 1)

            # Small arrow on top of head
            x, y, w, h = cv2.boundingRect(contour)
            head_x = x + w // 2
            head_y = y - 5
            cv2.arrowedLine(frame, (head_x, head_y-10), (head_x, head_y), (255, 0, 0), 2, tipLength=0.3)

            # Optional: label
            cv2.putText(frame, f"Person {person_id}", (head_x-15, head_y-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            person_id += 1

    # Overlay timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show frames
    cv2.imshow("Motion Detection", frame)
    cv2.imshow("Mask", fgmask)

    # Handle motion event
    if motion_detected:
        log_motion("Motion detected")
        save_snapshot(frame)

        # Start video clip recording
        if SAVE_VIDEO:
            if motion_clip_writer is None:
                motion_clip_writer = initialize_video_writer(frame)
                motion_timer = 0

    # Write to video clip if recording
    if motion_clip_writer:
        motion_clip_writer.write(frame)
        motion_timer += 1
        if motion_timer >= VIDEO_CLIP_LENGTH * fps:
            motion_clip_writer.release()
            motion_clip_writer = None
            print(f"🎥 Motion clip saved ({VIDEO_CLIP_LENGTH}s)")

    # Keyboard controls
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        save_snapshot(frame, prefix="manual")

# ---------------------------
# CLEANUP
# ---------------------------
cap.release()
if motion_clip_writer:
    motion_clip_writer.release()
cv2.destroyAllWindows()
print("✅ Motion detection stopped.")
