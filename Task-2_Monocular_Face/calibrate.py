import cv2
import math
import os
from face_landmarks import detect_face_landmarks

os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

REAL_EYE_DIST_M = 0.063   # change if you measure your own interpupillary distance

cap = cv2.VideoCapture(0)
f_values = []

print("Instructions:")
print(" - Stand at a KNOWN distance, hold still, press SPACE to capture")
print(" - Repeat at 3-4 different distances (e.g. 0.5m, 1m, 1.5m, 2m)")
print(" - Press 'q' when done to see averaged focal length\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    eye_dist_px = None
    landmarks = detect_face_landmarks(frame)
    if landmarks:
        x1, y1 = landmarks[33].x * w, landmarks[33].y * h     # left eye outer corner
        x2, y2 = landmarks[263].x * w, landmarks[263].y * h   # right eye outer corner
        eye_dist_px = math.hypot(x2 - x1, y2 - y1)

        cv2.circle(frame, (int(x1), int(y1)), 3, (0, 255, 0), -1)
        cv2.circle(frame, (int(x2), int(y2)), 3, (0, 255, 0), -1)
        cv2.putText(frame, f"eye_dist_px = {eye_dist_px:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(frame, f"Captures so far: {len(f_values)}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imshow("Calibration - SPACE to capture, q to finish", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' ') and eye_dist_px:
        dist_str = input("\nEnter the REAL distance (in meters) you are standing at: ")
        try:
            known_Z = float(dist_str)
            f = (eye_dist_px * known_Z) / REAL_EYE_DIST_M
            f_values.append(f)
            print(f"Captured f = {f:.2f} px at {known_Z} m\n")
        except ValueError:
            print("Invalid number, skipped.\n")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if f_values:
    avg_f = sum(f_values) / len(f_values)
    print("\n" + "=" * 50)
    print(f"Individual f values: {[round(f,2) for f in f_values]}")
    print(f"AVERAGED FOCAL LENGTH = {avg_f:.2f} px")
    print("Paste this into app.py as FOCAL_LENGTH_PX")
    print("=" * 50)
else:
    print("No captures recorded.")