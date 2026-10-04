import cv2
import csv
import os
import sys
import numpy as np

# Add parent directory to path so we can import OpenCV Zoo scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../opencv_zoo/models/palm_detection_mediapipe')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../opencv_zoo/models/handpose_estimation_mediapipe')))

try:
    from mp_palmdet import MPPalmDet
    from mp_handpose import MPHandPose
except ImportError:
    print("Error: Could not find opencv_zoo helper scripts. Double check folder paths!")
    sys.exit(1)

# 1. Paths to your ONNX models
PALM_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/palm_detection_mediapipe_2023feb.onnx'))
HAND_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/handpose_estimation_mediapipe_2023feb.onnx'))

# 2. Initialize the detector and estimator models
palm_detector = MPPalmDet(modelPath=PALM_MODEL_PATH)
hand_estimator = MPHandPose(modelPath=HAND_MODEL_PATH)

# 3. Define keyboard controls for recording gestures
# Key '0' = fist, Key '1' = palm, Key '2' = point, Key '3' = peace
LABEL_MAP = {
    ord('0'): 'fist',
    ord('1'): 'palm',
    ord('2'): 'point',
    ord('3'): 'peace'
}

# 4. Prepare the CSV file where data will be saved
CSV_FILE = os.path.join(os.path.dirname(__file__), 'raw_landmarks.csv')

# Write header row (label + 21 keypoints * 3 coordinates = 63 columns)
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['label']
        for i in range(21):
            header.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
        writer.writerow(header)

# 5. Open your webcam
cap = cv2.VideoCapture(0)

print("\n==========================================")
print("     HAND DATASET COLLECTOR STARTED      ")
print("==========================================")
print("Hold a hand gesture in front of the camera and press:")
print("  '0' -> Record FIST sample")
print("  '1' -> Record PALM sample")
print("  '2' -> Record POINT sample")
print("  '3' -> Record PEACE sample")
print("Press 'q' to QUIT.\n")

sample_counts = {'fist': 0, 'palm': 0, 'point': 0, 'peace': 0}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Camera error.")
        break

    # Flip camera so it acts like a mirror
    frame = cv2.flip(frame, 1)
    
    # Run Step 1: Detect palm/hand location
    palms = palm_detector.infer(frame)
    
    current_landmarks = None

    # If a hand is found:
    if palms is not None and len(palms) > 0:
        # Run Step 2: Estimate the 21 landmarks on that hand
        hand = hand_estimator.infer(frame, palms[0])
        if hand is not None:
            # Draw the detected keypoints on screen (landmarks are in hand[0])
            landmarks = hand[0] # Array of 21 points (x, y, z)
            current_landmarks = landmarks
            
            for pt in landmarks:
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    # Show instructions on screen
    cv2.putText(frame, "Keys: 0:fist | 1:palm | 2:point | 3:peace | q:quit", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("Data Collector", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    # Save data if you pressed 0, 1, 2, or 3 AND a hand is detected
    if key in LABEL_MAP and current_landmarks is not None:
        label_name = LABEL_MAP[key]
        
        row = [label_name]
        for pt in current_landmarks:
            row.extend([pt[0], pt[1], pt[2]])

        with open(CSV_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        sample_counts[label_name] += 1
        print(f"Recorded '{label_name}' | Total counts: {sample_counts}")

cap.release()
cv2.destroyAllWindows()

