import streamlit as st
import cv2
import numpy as np
import math
import os
from PIL import Image
from face_landmarks import detect_face_landmarks

os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")


DEFAULT_FOCAL_LENGTH_PX = 755.59   
DEFAULT_EYE_DIST_M = 0.063
DEFAULT_FACE_WIDTH_M = 0.15

st.set_page_config(page_title="Face Distance Estimator", layout="centered")
st.title("📷 Monocular Face Distance Estimation")
st.caption("Pinhole camera model — estimates depth (Z) and horizontal deviation angle (θ) from a single 2D image.")

st.sidebar.header("⚙️ Calibration Settings")
st.sidebar.markdown("Tune these for YOUR camera and face for best accuracy.")

focal_length_px = st.sidebar.number_input(
    "Focal Length f (px)", value=DEFAULT_FOCAL_LENGTH_PX, step=1.0,
    help="Get this from calibrate.py"
)

method = st.sidebar.radio(
    "Measurement method",
    ["Eye distance (more accurate)", "Face width (fallback)"]
)

if method == "Eye distance (more accurate)":
    real_measure_m = st.sidebar.number_input(
        "Your real interpupillary distance (m)", value=DEFAULT_EYE_DIST_M,
        step=0.001, format="%.3f",
        help="Average is 0.063 m. Measure your own for better accuracy."
    )
else:
    real_measure_m = st.sidebar.number_input(
        "Your real face width (m)", value=DEFAULT_FACE_WIDTH_M,
        step=0.001, format="%.3f",
        help="Measure cheekbone to cheekbone with a ruler for best accuracy."
    )

st.sidebar.markdown("---")
st.sidebar.caption("Tip: run calibrate.py at 3-4 distances and average the focal length for best results.")

st.markdown("---")

# ---------------- INPUT MODE ----------------
mode = st.radio("Choose input mode:", ["Upload Image", "Use Webcam Snapshot"])

def estimate(frame, method, focal_length_px, real_measure_m):
    h, w, _ = frame.shape
    c_x = w / 2

    landmarks = detect_face_landmarks(frame)
    if not landmarks:
        return None, frame

    if method == "Eye distance (more accurate)":
        left_index, right_index = 33, 263
    else:
        left_index, right_index = 234, 454

    x1, y1 = landmarks[left_index].x * w, landmarks[left_index].y * h
    x2, y2 = landmarks[right_index].x * w, landmarks[right_index].y * h
    measured_px = math.hypot(x2 - x1, y2 - y1)

    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    cv2.circle(frame, (int(x1), int(y1)), 5, (0, 255, 0), -1)
    cv2.circle(frame, (int(x2), int(y2)), 5, (0, 255, 0), -1)
    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    if measured_px <= 0:
        return None, frame

    Z = (focal_length_px * real_measure_m) / measured_px
    theta = math.degrees(math.atan((x_center - c_x) / focal_length_px))

    cv2.circle(frame, (int(x_center), int(y_center)), 5, (0, 0, 255), -1)
    cv2.line(frame, (int(c_x), 0), (int(c_x), h), (255, 0, 0), 2)

    return (Z, theta, measured_px), frame


image_input = None

if mode == "Upload Image":
    uploaded = st.file_uploader("Upload a face photo", type=["jpg", "jpeg", "png"])
    if uploaded:
        image_input = Image.open(uploaded).convert("RGB")
else:
    cam_img = st.camera_input("Take a photo")
    if cam_img:
        image_input = Image.open(cam_img).convert("RGB")

if image_input is not None:
    frame = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    result, annotated = estimate(frame.copy(), method, focal_length_px, real_measure_m)

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    st.image(annotated_rgb, caption="Detection Result", use_column_width=True)

    if result:
        Z, theta, measured_px = result
        col1, col2 = st.columns(2)
        col1.metric("Estimated Depth (Z)", f"{Z:.2f} m")
        col2.metric("Deviation Angle (θ)", f"{theta:.2f}°")

        st.success("Face detected and measured successfully ✅")

        with st.expander("Calculation details"):
            st.latex(r"Z = \frac{f \times W}{w_{px}}")
            st.latex(r"\theta = \arctan\left(\frac{x - c_x}{f}\right)")
            st.write(f"f = {focal_length_px} px")
            st.write(f"Real-world measurement (W) = {real_measure_m} m")
            st.write(f"Detected pixel measurement (w_px) = {measured_px:.2f} px")
    else:
        st.error("No face detected. Try better lighting or face the camera more directly.")

st.markdown("---")
st.caption("Built using the pinhole camera model · Monocular Face Distance Estimation · HackTronix 2.0")