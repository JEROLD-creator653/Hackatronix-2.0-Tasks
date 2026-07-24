# Hackatronix 2.0 Tasks

![Computer Vision Project](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Landmarks-orange)

A professional computer vision project that combines two practical vision pipelines:

- Task 1: Real-time ball detection and tracking
- Task 2: Monocular face landmark detection with a Streamlit-based interface

## Project Overview

This repository demonstrates end-to-end deployment of computer vision solutions for two different challenges:

1. Object detection for ball recognition in live video streams
2. Facial landmark estimation from a single camera input for real-time analysis

The project is structured to support both experimentation and evaluation, with reusable scripts, trained weights, and a dependency manifest for easy setup.

## Repository Structure

- Task-1_Ball_Detection/
  - Contains the ball detection pipeline, training script, webcam inference script, and dataset files
- Task-2_Monocular_Face/
  - Contains the face landmark detection app, calibration logic, and MediaPipe-based inference helper
- requirements.txt
  - Python dependencies needed for both tasks
- README.md
  - Project overview, setup instructions, and evaluation notes

## Task 1: Ball Detection

### Objective

Detect cricket or tennis-style balls in a live webcam feed and measure the system's real-time performance.

### Results Summary

| Metric | Value |
| --- | ---: |
| F1 Score | 0.894 |
| Precision | 0.887 |
| Recall | 0.902 |
| Maximum FPS | 77.8 |
| Average Latency | 15.3 ms |

### What the system does

- Loads a YOLO-based detection model
- Captures frames from the webcam
- Runs inference frame-by-frame
- Displays live performance indicators such as latency and FPS
- Prints final evaluation metrics on shutdown

### Evaluation Highlights

Representative results observed during local evaluation:

- F1 Score: 0.894
- Precision: 0.887
- Recall: 0.902
- Maximum FPS observed: 77.8
- Average latency: approximately 15.3 ms per frame

These values indicate a strong balance between detection accuracy and real-time responsiveness for the deployed model.

### Key Files

- Task-1_Ball_Detection/detect.py
- Task-1_Ball_Detection/train.py
- Task-1_Ball_Detection/data.yaml

## Task 2: Monocular Face Landmark Detection

### Objective

Estimate facial landmarks from a single camera stream using a monocular vision approach with MediaPipe Face Landmarker.

### What the system does

- Reads live video frames from the camera
- Detects facial regions and landmark points
- Displays results in a Streamlit web interface
- Supports calibration-related processing for geometric analysis

### Evaluation Approach

The quality of landmark localization can be evaluated using geometric error. For a set of $N$ predicted landmarks and corresponding ground-truth landmarks, the average point error is computed as:

| Metric | Formula | Interpretation |
| --- | --- | --- |
| Mean Point Error | $E_{mean} = \frac{1}{N} \sum_{i=1}^{N} \sqrt{(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2}$ | Lower is better |
| Normalized Error | $E_{norm} = \frac{E_{mean}}{D_{interocular}}$ | Makes comparison fair across different face sizes |

$$
E_{mean} = \frac{1}{N} \sum_{i=1}^{N} \sqrt{(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2}
$$

Where:
- $x_i, y_i$ are the predicted landmark coordinates
- $\hat{x}_i, \hat{y}_i$ are the ground-truth landmark coordinates
- $N$ is the total number of evaluated landmarks

A normalized version of the error is often used for fair comparison across different face sizes:

$$
E_{norm} = \frac{E_{mean}}{D_{interocular}}
$$

Where $D_{interocular}$ is the distance between the eyes. Lower values indicate better landmark accuracy.

### Key Files

- Task-2_Monocular_Face/app.py
- Task-2_Monocular_Face/calibrate.py
- Task-2_Monocular_Face/face_landmarks.py

## Setup Instructions

### 1. Create a virtual environment

```bash
python -m venv BD
```

### 2. Activate the environment

On Windows PowerShell:

```powershell
BD\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Projects

### Run Task 1

```bash
python Task-1_Ball_Detection/detect.py
```

### Run Task 2

```bash
python -m streamlit run Task-2_Monocular_Face/app.py
```

## Notes

- The virtual environment folder is intentionally ignored and will not be uploaded to GitHub.
- The repository includes trained weights and supporting assets needed for local execution.
- For best performance on Windows, Task 1 uses a DirectShow-based camera configuration to improve webcam compatibility.
