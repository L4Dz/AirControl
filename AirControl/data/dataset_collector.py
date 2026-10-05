import cv2
import csv
import os
import sys
import numpy as np

#this is just a way of getting the path to the src folder, so we can import the modules from there 
# ( i believe we could do like absolute path but this is more elegant)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    from src.mp_palmdet import MPPalmDet
    from src.mp_handpose import MPHandPose
except ImportError as e:
    print(f"Import Error: {e}")
    print("Make sure mp_palmdet.py and mp_handpose.py are placed in the 'src/' folder!")
    sys.exit(1)

PALM_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'palm_detection_mediapipe_2023feb.onnx')
HAND_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'handpose_estimation_mediapipe_2023feb.onnx')

palm_detector = MPPalmDet(modelPath=PALM_MODEL_PATH)
hand_estimator = MPHandPose(modelPath=HAND_MODEL_PATH)

LABEL_MAP = {
    ord('0'): 'fist',
    ord('1'): 'palm',
    ord('2'): 'point',
    ord('3'): 'peace'
}

CONF_THRESHOLD = 0.8 #min confidence for hand detection to be considered valid 

DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
CSV_FILE = os.path.join(DATA_DIR, 'raw_landmarks.csv')

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['label']
        for i in range(21):
            header.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
        writer.writerow(header)


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # palm edge
]


def parse_hand_result(hand_res):
    """
    MPHandPose.infer() returns a flat array of 132 values:
      [0:4]     palm bounding box (x1, y1, x2, y2)
      [4:67]    21 screen landmarks (x, y, z) in image pixels
      [67:130]  21 world landmarks (x, y, z)
      [130]     handedness
      [131]     confidence
    Returns (screen_landmarks (21,3), bbox (x1,y1,x2,y2), confidence) or (None, None, 0.0).
    """
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
    """
    Make landmarks independent of hand position and size in the frame:
    wrist (point 0) becomes the origin, and everything is scaled so the
    farthest point from the wrist is at distance 1.
    """
    pts = lm[:, :2] - lm[0, :2]
    scale = np.max(np.linalg.norm(pts, axis=1)) + 1e-6
    xy = pts / scale
    z = (lm[:, 2] - lm[0, 2]) / scale
    return np.column_stack([xy, z])



cap = cv2.VideoCapture(0)

print("\n==========================================")
print("     HAND DATASET COLLECTOR READY         ")
print("==========================================")
print("Press '0': Fist | '1': Palm | '2': Point | '3': Peace | 'q': Quit\n")

sample_counts = {'fist': 0, 'palm': 0, 'point': 0, 'peace': 0}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error reading camera frame.")
        break

    frame = cv2.flip(frame, 1)

    # Detect palms and use only the strongest one (score is the last column)
    palms = palm_detector.infer(frame)
    current_landmarks = None
    bbox = None
    confidence = 0.0

    if palms is not None and len(palms) > 0:
        best_palm = palms[np.argmax(palms[:, -1])]
        hand_res = hand_estimator.infer(frame, best_palm)
        landmarks, bbox, confidence = parse_hand_result(hand_res)

        if landmarks is not None and confidence >= CONF_THRESHOLD:
            current_landmarks = landmarks

            # Box around the palm
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

            # Red lines connecting the landmarks
            for a, b in HAND_CONNECTIONS:
                pa = (int(current_landmarks[a][0]), int(current_landmarks[a][1]))
                pb = (int(current_landmarks[b][0]), int(current_landmarks[b][1]))
                cv2.line(frame, pa, pb, (0, 0, 255), 2)

            # Green dots on top of the lines
            for pt in current_landmarks:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)

    status_text = "0:fist | 1:palm | 2:point | 3:peace | q:quit"
    if current_landmarks is not None:
        cv2.putText(frame, f"[HAND DETECTED] conf={confidence:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Data Collector", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key in LABEL_MAP and current_landmarks is not None:
        label_name = LABEL_MAP[key]
        norm = normalize(current_landmarks)  # (21, 3), position/scale invariant

        row = [label_name] + [float(v) for v in norm.flatten()]
        with open(CSV_FILE, mode='a', newline='') as f:
            csv.writer(f).writerow(row)

        sample_counts[label_name] += 1
        print(f"Recorded '{label_name}' sample! Total counts: {sample_counts}")

cap.release()
cv2.destroyAllWindows()