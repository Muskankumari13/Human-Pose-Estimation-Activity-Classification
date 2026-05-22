#  Human Pose Estimation & Activity Classification
> Computer Vision Lab Assignment — CLO-3 | GA-4

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-green?logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Overview

A real-time human activity recognition pipeline built using **MediaPipe Pose**
and **OpenCV**. The system extracts 33 body keypoints from a live webcam feed,
computes joint angles (knee, elbow, hip), applies EMA smoothing, and classifies
three activities using rule-based thresholds.

**Video Source:** Live Webcam (CV2 VideoCapture, index 0) — no pre-recorded video used.

---

##  Activities Detected

| Activity      | Knee Angle | Hip Angle | Elbow Angle |
|---------------|------------|-----------|-------------|
| Standing      | > 160°      | > 160°     | > 160°       |
| Squatting     | < 130°      | < 130°     | Any         |
| Hands Raised  | > 160°      | > 160°     | < 160°       |

---

## Repository Structure

```
pose-estimation-activity-classification/
│
├── pose_estimation.py          # Main Python script (all 3 tasks)
├── CV_Lab_Report.pdf           # Short report (angle logic, accuracy, screenshots)
└── README.md                   # This file
```

---

##  Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Muskankumari13/pose-estimation-activity-classification.git
cd pose-estimation-activity-classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the script
```bash
python pose_estimation.py
```
> Make sure your webcam is connected. Press **`q`** to quit the live window.

---

##  Requirements

```
opencv-python>=4.5.0
mediapipe>=0.10.0
numpy>=1.21.0
matplotlib>=3.4.0
```

---

## Pipeline Summary

### Task 1 — Pose Detection & Pre-processing
- MediaPipe Pose extracts **33 landmarks** per frame in real time
- **EMA smoothing** (α = 0.5) applied to keypoint coordinates to reduce jitter
- Skeleton overlay rendered on each frame using OpenCV

### Task 2 — Joint Angle Computation & Tracking
- Angles computed using the **dot-product formula** on vectors from adjacent landmarks
- Three joints tracked: **Knee**, **Elbow**, **Hip**
- Angle vs. frame graph plotted with Matplotlib — activity transitions clearly visible

### Task 3 — Rule-Based Activity Classification
- Priority-cascade classifier: Squatting → Hands Raised → Standing (default)
- Compared against manually labeled ground truth
- **Overall Accuracy: 96.1%** across 1070 frames

---

## 📊 Results

| Activity      | GT Frames | Correct | Accuracy |
|---------------|-----------|---------|----------|
| Standing      | 520       | 503     | 96.7%    |
| Squatting     | 330       | 318     | 96.4%    |
| Hands Raised  | 220       | 207     | 94.1%    |
| **Overall**   | **1070**  | **1028**| **96.1%**|

<img width="1600" height="831" alt="image" src="https://github.com/user-attachments/assets/c5ed2957-02d1-43e7-a32e-f111fcb744b0" />


*(See `CV_Lab_Report.pdf` for full annotated screenshots and angle plots)*



---

## 📄 License
MIT License — feel free to use for academic purposes.
