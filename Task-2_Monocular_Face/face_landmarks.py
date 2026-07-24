from functools import lru_cache
from pathlib import Path
import urllib.request

import cv2
import mediapipe as mp


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_PATH = Path(__file__).with_name("face_landmarker.task")


def _download_model() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


@lru_cache(maxsize=1)
def get_face_landmarker():
    base_options = mp.tasks.BaseOptions(model_asset_path=str(_download_model()))
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def detect_face_landmarks(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = get_face_landmarker().detect(mp_image)

    if not result.face_landmarks:
        return None

    return result.face_landmarks[0]