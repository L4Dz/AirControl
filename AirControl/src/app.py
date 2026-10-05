import cv2
import os
import sys
import numpy as np
import joblib

# 1. Add base directory to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    from src.mp_palmdet import MPPalmDet
    from src.mp_handpose import MPHandPose
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# 2. Paths to ONNX and Trained Model
PALM_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'palm_detection_mediapipe_2023feb.onnx')
HAND_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'handpose_estimation_mediapipe_2023feb.onnx')
CLASSIFIER_PATH = os.path.join(BASE_DIR, 'models', 'rps_point_model.pkl')

# 3. Load Models
palm_detector = MPPalmDet(modelPath=PALM_MODEL_PATH)
hand_estimator = MPHandPose(modelPath=HAND_MODEL_PATH)
classifier = joblib.load(CLASSIFIER_PATH)

CONF_THRESHOLD = 0.7

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

def parse_hand_result(hand_res):
    if hand_res is None:
        return None, None, 0.0
    arr = np.asarray(hand_res, dtype=np.float32).flatten()
    if arr.size < 132:
        return None, None, 0.0
    bbox = arr[0:4].astype(int)
    landmarks = arr[4:67].reshape(21, 3)
    confidence = float(arr[131])
    return landmarks, bbox, confidence

def normalize(lm):
    pts = lm[:, :2] - lm[0, :2]
    scale = np.max(np.linalg.norm(pts, axis=1)) + 1e-6
    xy = pts / scale
    z = (lm[:, 2] - lm[0, 2]) / scale
    return np.column_stack([xy, z])

cap = cv2.VideoCapture(0)
canvas = None
prev_x, prev_y = 0, 0

print("\n==========================================")
print("     AIRCONTROL LIVE INFERENCE READY      ")
print("==========================================")
print("Point: Draw | Rock: Clear Canvas | Press 'q': Quit\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    palms = palm_detector.infer(frame)
    if palms is not None and len(palms) > 0:
        best_palm = palms[np.argmax(palms[:, -1])]
        hand_res = hand_estimator.infer(frame, best_palm)
        landmarks, bbox, confidence = parse_hand_result(hand_res)

        if landmarks is not None and confidence >= CONF_THRESHOLD:
            # Draw landmarks
            for a, b in HAND_CONNECTIONS:
                pa = (int(landmarks[a][0]), int(landmarks[a][1]))
                pb = (int(landmarks[b][0]), int(landmarks[b][1]))
                cv2.line(frame, pa, pb, (0, 0, 255), 2)

            for pt in landmarks:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)

            # Normalize landmarks
            norm_landmarks = normalize(landmarks)
            feature_vector = norm_landmarks.flatten().reshape(1, -1)

            # Predict gesture class
            prediction = classifier.predict(feature_vector)[0]

            # Get index tip position (landmark 8)
            index_tip = landmarks[8]
            cx, cy = int(index_tip[0]), int(index_tip[1])

            # Handle Actions
            if prediction == 'point':
                cv2.circle(frame, (cx, cy), 8, (255, 0, 0), cv2.FILLED)
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = cx, cy
                cv2.line(canvas, (prev_x, prev_y), (cx, cy), (0, 255, 0), 4)
                prev_x, prev_y = cx, cy

            elif prediction == 'paper':
                # Rock (fist) clears/erases the entire canvas
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                prev_x, prev_y = 0, 0

            else:
                prev_x, prev_y = 0, 0

            # Display prediction banner
            cv2.putText(frame, f"Gesture: {prediction.upper()}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    frame = cv2.add(frame, canvas)
    cv2.imshow("AirControl Live", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()