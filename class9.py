import os
import sys
import numpy as np
import cv2

# --- Point Python at the zoo's helper classes and models ---
ZOO = '/Users/ladielsonrenner/Documents/CODING/proj/opencv_zoo/models'
PALM_DIR = os.path.join(ZOO, 'palm_detection_mediapipe')
HAND_DIR = os.path.join(ZOO, 'handpose_estimation_mediapipe')
sys.path.append(PALM_DIR)
sys.path.append(HAND_DIR)

from mp_palmdet import MPPalmDet
from mp_handpose import MPHandPose

backend = cv2.dnn.DNN_BACKEND_OPENCV
target = cv2.dnn.DNN_TARGET_CPU

palm_detector = MPPalmDet(
    modelPath=os.path.join(PALM_DIR, 'palm_detection_mediapipe_2023feb.onnx'),
    nmsThreshold=0.3,
    scoreThreshold=0.6,
    backendId=backend,
    targetId=target)

handpose_detector = MPHandPose(
    modelPath=os.path.join(HAND_DIR, 'handpose_estimation_mediapipe_2023feb.onnx'),
    confThreshold=0.9,
    backendId=backend,
    targetId=target)

# Pairs of landmark indices that form the skeleton (wrist=0, then each finger)
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (0, 9), (9, 10), (10, 11), (11, 12),    # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Stage 1: find palms. Stage 2: read the skeleton of each palm.
    palms = palm_detector.infer(frame)

    for palm in palms:
        hand = handpose_detector.infer(frame, palm)
        if hand is None:
            continue

        handedness = hand[-2]                       # <= 0.5 means Left
        label = 'Left' if handedness <= 0.5 else 'Right'
        bbox = hand[0:4].astype(np.int32)
        landmarks = hand[4:67].reshape(21, 3).astype(np.int32)[:, :2]  # x, y only

        # bounding box + label
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # skeleton lines, then joint dots
        for a, b in CONNECTIONS:
            cv2.line(frame, tuple(landmarks[a]), tuple(landmarks[b]), (255, 255, 255), 2)
        for point in landmarks:
            cv2.circle(frame, tuple(point), 5, (0, 0, 255), -1)

        # Example of using the data: thumb tip (4) and index tip (8) distance
        dist = np.linalg.norm(landmarks[4] - landmarks[8])
        cv2.putText(frame, f'Pinch dist: {int(dist)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow('Hands', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()