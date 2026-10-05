import cv2
import os
import sys
import numpy as np
import joblib
import random
import time

# 1. Base Directory Setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    from src.mp_palmdet import MPPalmDet
    from src.mp_handpose import MPHandPose
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# 2. Model Paths
PALM_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'palm_detection_mediapipe_2023feb.onnx')
HAND_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'handpose_estimation_mediapipe_2023feb.onnx')
CLASSIFIER_PATH = os.path.join(BASE_DIR, 'models', 'rps_point_model.pkl')

# 3. Load ONNX and Scikit-Learn Models
palm_detector = MPPalmDet(modelPath=PALM_MODEL_PATH)
hand_estimator = MPHandPose(modelPath=HAND_MODEL_PATH)
classifier = joblib.load(CLASSIFIER_PATH)

CONF_THRESHOLD = 0.8

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

def get_winner(player, ai):
    if player == ai:
        return "DRAW"
    winning_rules = {
        'rock': 'scissors',
        'scissors': 'paper',
        'paper': 'rock'
    }
    if winning_rules.get(player) == ai:
        return "PLAYER"
    return "AI"

# Game State Variables
player_score = 0
ai_score = 0
current_round_state = "WAITING"  # States: WAITING, COUNTDOWN, RESULT
countdown_start_time = 0
COUNTDOWN_DURATION = 3 # 3 seconds

user_move = None
ai_move = None
result_text = ""

cap = cv2.VideoCapture(0)

print("\n==========================================")
print("    ROCK PAPER SCISSORS GAME READY        ")
print("==========================================")
print("Controls: 's' = Start Round | 'r' = Reset Score | 'q' = Quit\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Process Hand Landmarks
    detected_gesture = None
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

            # Predict gesture
            norm_landmarks = normalize(landmarks)
            feature_vector = norm_landmarks.flatten().reshape(1, -1)
            detected_gesture = classifier.predict(feature_vector)[0]

    # game logic based on current state
    current_time = time.time()

    if current_round_state == "COUNTDOWN":
        elapsed = current_time - countdown_start_time
        remaining = int(np.ceil(COUNTDOWN_DURATION - elapsed))

        if remaining > 0:
            # Big Countdown Banner
            cv2.putText(frame, f"GET READY: {remaining}", (w // 2 - 180, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)
        else:
            # Time's up! Lock choices
            if detected_gesture in ['rock', 'paper', 'scissors']:
                user_move = detected_gesture
                ai_move = random.choice(['rock', 'paper', 'scissors'])
                winner = get_winner(user_move, ai_move)

                if winner == "PLAYER":
                    player_score += 1
                    result_text = "YOU WIN!"
                elif winner == "AI":
                    ai_score += 1
                    result_text = "AI WINS!"
                else:
                    result_text = "IT'S A DRAW!"

                current_round_state = "RESULT"
            else:
                result_text = "NO VALID GESTURE DETECTED!"
                current_round_state = "RESULT"

    # Top Scoreboard Banner
    cv2.rectangle(frame, (0, 0), (w, 80), (30, 30, 30), cv2.FILLED)
    cv2.putText(frame, f"PLAYER: {player_score}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(frame, f"AI: {ai_score}", (w - 180, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    if detected_gesture:
        cv2.putText(frame, f"Detected: {detected_gesture.upper()}", (30, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display Round Results
    if current_round_state == "RESULT":
        cv2.putText(frame, f"YOU: {user_move.upper() if user_move else 'N/A'}", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"AI: {ai_move.upper() if ai_move else 'N/A'}", (w - 250, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Winner Color
        color = (0, 255, 0) if "WIN" in result_text and "AI" not in result_text else (0, 0, 255)
        if "DRAW" in result_text:
            color = (0, 255, 255)

        cv2.putText(frame, result_text, (w // 2 - 150, h // 2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.putText(frame, "Press 's' to play next round", (w // 2 - 200, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    elif current_round_state == "WAITING":
        cv2.putText(frame, "Press 's' to Start Round!", (w // 2 - 220, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    cv2.imshow("AirControl - Rock Paper Scissors Game", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        current_round_state = "COUNTDOWN"
        countdown_start_time = time.time()
    elif key == ord('r'):
        player_score = 0
        ai_score = 0
        current_round_state = "WAITING"

cap.release()
cv2.destroyAllWindows()