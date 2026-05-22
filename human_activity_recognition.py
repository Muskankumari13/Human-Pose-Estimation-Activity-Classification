# HUMAN ACTIVITY RECOGNITION USING MEDIAPIPE + WEBCAM
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request

from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")


# MEDIAPIPE SETUP  (Tasks API — mediapipe 0.10.x / Python 3.13)
# Landmark indices (same numbering as the old PoseLandmark enum)
LEFT_SHOULDER = 11
LEFT_ELBOW    = 13
LEFT_WRIST    = 15
LEFT_HIP      = 23
LEFT_KNEE     = 25
LEFT_ANKLE    = 27

# Skeleton connections for drawing
POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(24,26),
    (25,27),(26,28),(27,29),(28,30),(27,31),(28,32)
]

def draw_pose_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            s = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
            e = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
            cv2.line(frame, s, e, (255, 255, 255), 2)

# Download the lite pose-landmarker model if not already present
MODEL_PATH = 'pose_landmarker_lite.task'
if not os.path.exists(MODEL_PATH):
    print("Downloading pose landmarker model (~5 MB)...")
    url = (
        'https://storage.googleapis.com/mediapipe-models/'
        'pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
    )
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Model downloaded.")

PoseLandmarker        = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
BaseOptions           = mp.tasks.BaseOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ANGLE CALCULATION FUNCTION
def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc)
    )

    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine_angle))

    return angle

# STORAGE VARIABLES
knee_angles = []
elbow_angles = []
hip_angles = []

predicted_labels = []

# START WEBCAM
cap = cv2.VideoCapture(0)
print("Press 'q' to stop webcam")
# POSE DETECTION
with PoseLandmarker.create_from_options(pose_options) as pose_detector:

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        # Flip frame
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Pose Detection (Tasks API)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = pose_detector.detect(mp_image)

        # LANDMARK PROCESSING
        if results.pose_landmarks:

            landmarks = results.pose_landmarks[0]  # first detected person

            # EXTRACT KEYPOINTS
            shoulder = [landmarks[LEFT_SHOULDER].x, landmarks[LEFT_SHOULDER].y]

            elbow = [landmarks[LEFT_ELBOW].x, landmarks[LEFT_ELBOW].y]

            wrist = [landmarks[LEFT_WRIST].x, landmarks[LEFT_WRIST].y]

            hip = [landmarks[LEFT_HIP].x, landmarks[LEFT_HIP].y]

            knee = [landmarks[LEFT_KNEE].x, landmarks[LEFT_KNEE].y]

            ankle = [landmarks[LEFT_ANKLE].x, landmarks[LEFT_ANKLE].y]

            # JOINT ANGLE COMPUTATION
            elbow_angle = calculate_angle(

                shoulder,
                elbow,
                wrist
            )

            knee_angle = calculate_angle(

                hip,
                knee,
                ankle
            )

            hip_angle = calculate_angle(

                shoulder,
                hip,
                knee
            )
            # STORE ANGLES
            knee_angles.append(knee_angle)
            elbow_angles.append(elbow_angle)
            hip_angles.append(hip_angle)

            # RULE-BASED CLASSIFICATION
            activity = "Unknown"

            # Standing
            if knee_angle > 160:
                activity = "Standing"

            # Squatting
            if knee_angle < 100:
                activity = "Squatting"

            # Hands Raised
            if elbow_angle > 150 and wrist[1] < shoulder[1]:
                activity = "Hands Raised"

            predicted_labels.append(activity)

            # DISPLAY ACTIVITY
            cv2.putText(

                frame,

                f'Activity: {activity}',

                (20, 50),

                cv2.FONT_HERSHEY_SIMPLEX,

                1,

                (0, 255, 0),

                2
            )

            # DISPLAY ANGLES
            cv2.putText(

                frame,

                f'Knee Angle: {int(knee_angle)}',

                (20, 100),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (255, 0, 0),

                2
            )

            cv2.putText(

                frame,

                f'Elbow Angle: {int(elbow_angle)}',

                (20, 140),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (255, 0, 0),

                2
            )

            cv2.putText(

                frame,

                f'Hip Angle: {int(hip_angle)}',

                (20, 180),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (255, 0, 0),

                2
            )

            # DRAW SKELETON
            draw_pose_landmarks(frame, landmarks)

        # SHOW FRAME
        cv2.imshow(

            "Human Activity Recognition",

            frame
        )

        # EXIT CONDITION
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# RELEASE CAMERA
cap.release()
cv2.destroyAllWindows()

# APPLY GAUSSIAN SMOOTHING
smooth_knee = gaussian_filter1d(
    knee_angles,
    sigma=2
)

smooth_elbow = gaussian_filter1d(
    elbow_angles,
    sigma=2
)

smooth_hip = gaussian_filter1d(
    hip_angles,
    sigma=2
)
# PLOT ANGLES
plt.figure(figsize=(14, 6))

plt.plot(
    smooth_knee,
    label='Knee Angle'
)

plt.plot(
    smooth_elbow,
    label='Elbow Angle'
)

plt.plot(
    smooth_hip,
    label='Hip Angle'
)

plt.xlabel("Frames")
plt.ylabel("Angle")
plt.title("Joint Angles Over Time")

plt.legend()
plt.grid()

plt.show()

# MANUAL GROUND TRUTH LABELS
# MODIFY ACCORDING TO YOUR RECORDING
ground_truth = []

ground_truth += ["Standing"] * 100
ground_truth += ["Hands Raised"] * 100
ground_truth += ["Squatting"] * 100

# MATCH LABEL LENGTHS
min_len = min(
    len(ground_truth),
    len(predicted_labels)
)

ground_truth = ground_truth[:min_len]
predicted_labels = predicted_labels[:min_len]

# ACCURACY CALCULATION
accuracy = accuracy_score(
    ground_truth,
    predicted_labels
)
print(f"Accuracy: {round(accuracy * 100, 2)} %")