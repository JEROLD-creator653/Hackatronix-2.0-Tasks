from ultralytics import YOLO
import cv2
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "runs" / "ball_detection" / "weights" / "best.pt"
DATA_YAML = ROOT / "data.yaml"

model = YOLO(str(WEIGHTS))

precision = 0.0
recall = 0.0
f1_score = 0.0

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam with DirectShow backend.")

frame_count = 0
latency_sum_ms = 0.0
fps_sum = 0.0


def draw_legend(frame, latency_ms, fps):
    overlay = frame.copy()
    x1, y1 = 12, 12
    x2, y2 = 320, 95
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    lines = [
        f"Latency: {latency_ms:.1f} ms",
        f"FPS: {fps:.1f}",
    ]

    for idx, text in enumerate(lines):
        y = y1 + 28 + idx * 22
        cv2.putText(frame, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)


def enhance_visibility(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    y_channel = clahe.apply(y_channel)

    boosted = cv2.cvtColor(cv2.merge((y_channel, cr_channel, cb_channel)), cv2.COLOR_YCrCb2BGR)

    if boosted.mean() < 40:
        boosted = cv2.convertScaleAbs(boosted, alpha=3.0, beta=40)

    return boosted


def compute_validation_metrics():
    metrics_model = YOLO(str(WEIGHTS))
    metrics = metrics_model.val(data=str(DATA_YAML), imgsz=640, device="cpu", verbose=False)
    precision_value = float(metrics.box.mp)
    recall_value = float(metrics.box.mr)
    f1_value = 0.0 if (precision_value + recall_value) == 0 else (2.0 * precision_value * recall_value) / (precision_value + recall_value)
    return precision_value, recall_value, f1_value

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame
        start_time = time.perf_counter()
        results = model(display_frame, verbose=False)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        fps = 0.0 if latency_ms <= 0 else 1000.0 / latency_ms

        frame_count += 1
        latency_sum_ms += latency_ms
        fps_sum += fps

        annotated = results[0].plot()
        draw_legend(annotated, latency_ms, fps)

        cv2.imshow("Ball Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()

    if frame_count > 0:
        try:
            precision, recall, f1_score = compute_validation_metrics()
        except Exception as exc:
            print(f"\nValidation failed: {exc}")
    else:
        print("\nNo webcam frames captured; skipping validation summary.")

    avg_latency_ms = 0.0 if frame_count == 0 else latency_sum_ms / frame_count
    avg_fps = 0.0 if frame_count == 0 else fps_sum / frame_count

    print("\nDetection summary")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1_score:.3f}")
    print(f"Average latency: {avg_latency_ms:.1f} ms")
    print(f"Average FPS: {avg_fps:.1f}")