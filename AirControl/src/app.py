import cv2
import os
import sys
import numpy as np
import joblib
import random
import time
from collections import deque, Counter

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
DRAWINGS_DIR = os.path.join(BASE_DIR, 'drawings')

# 3. Load ONNX and Scikit-Learn Models
palm_detector = MPPalmDet(modelPath=PALM_MODEL_PATH)
hand_estimator = MPHandPose(modelPath=HAND_MODEL_PATH)
classifier = joblib.load(CLASSIFIER_PATH)

CONF_THRESHOLD = 0.8
POINT_LABEL = 'point'          # label used in your model for the pointing gesture
INDEX_TIP = 8                  # landmark index of the index fingertip

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
    winning_rules = {'rock': 'scissors', 'scissors': 'paper', 'paper': 'rock'}
    if winning_rules.get(player) == ai:
        return "PLAYER"
    return "AI"


# ---------------------------------------------------------------
# Drawing settings
# ---------------------------------------------------------------
TOOLBAR_H = 90
BTN_SIZE = 60
BTN_GAP = 15
BTN_Y = 15
DWELL_TIME = 0.8           # seconds pointing at a button to activate it
SMOOTHING = 0.5            # 0..1, lower = smoother but laggier cursor
ERASER_THICKNESS = 40

# BGR colors
PEN_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 255)]
TOOLBAR_ITEMS = (
    [('color', c) for c in PEN_COLORS]
    + [('eraser', None), ('undo', None), ('clear', None)]
)


def button_rect(i):
    x1 = BTN_GAP + i * (BTN_SIZE + BTN_GAP)
    return x1, BTN_Y, x1 + BTN_SIZE, BTN_Y + BTN_SIZE


def button_at(x, y):
    for i in range(len(TOOLBAR_ITEMS)):
        x1, y1, x2, y2 = button_rect(i)
        if x1 <= x <= x2 and y1 <= y <= y2:
            return i
    return None


def draw_strokes(canvas, strokes):
    for s in strokes:
        pts = s['points']
        if len(pts) == 1:
            cv2.circle(canvas, pts[0], max(1, s['thickness'] // 2), s['color'], -1, cv2.LINE_AA)
        else:
            cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False,
                          s['color'], s['thickness'], cv2.LINE_AA)


# ---------------------------------------------------------------
# State
# ---------------------------------------------------------------
mode = "GAME"  # GAME or DRAW

# Game state
player_score = 0
ai_score = 0
current_round_state = "WAITING"  # WAITING, COUNTDOWN, RESULT
countdown_start_time = 0
COUNTDOWN_DURATION = 3
user_move = None
ai_move = None
result_text = ""

# Drawing state
strokes = []
current_stroke = None
brush_color = PEN_COLORS[0]
brush_thickness = 6
eraser_on = False
smooth_pos = None
dwell_target = None
dwell_start = 0.0
dwell_done = False

# Gesture stabilisation (majority vote over the last frames)
gesture_history = deque(maxlen=5)

cap = cv2.VideoCapture(0)

print("\n==========================================")
print("         AIR CONTROL APP READY            ")
print("==========================================")
print("Global : 'm' = switch GAME/DRAW | 'q' = quit")
print("Game   : 's' = start round | 'r' = reset score")
print("Draw   : point to draw | 'c' clear | 'z' undo | 'e' eraser | '+'/'-' size | 'p' save\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    now = time.time()

    # -----------------------------------------------------------
    # Hand tracking + gesture classification
    # -----------------------------------------------------------
    landmarks = None
    detected_gesture = None
    palms = palm_detector.infer(frame)

    if palms is not None and len(palms) > 0:
        best_palm = palms[np.argmax(palms[:, -1])]
        hand_res = hand_estimator.infer(frame, best_palm)
        lm, bbox, confidence = parse_hand_result(hand_res)

        if lm is not None and confidence >= CONF_THRESHOLD:
            landmarks = lm
            feature_vector = normalize(landmarks).flatten().reshape(1, -1)
            detected_gesture = classifier.predict(feature_vector)[0]

    gesture_history.append(detected_gesture)
    votes = Counter(g for g in gesture_history if g is not None)
    stable_gesture = None
    if votes:
        g, count = votes.most_common(1)[0]
        if count >= 3:
            stable_gesture = g

    # Smoothed index fingertip position
    if landmarks is not None:
        tip = landmarks[INDEX_TIP][:2].astype(np.float32)
        smooth_pos = tip if smooth_pos is None else SMOOTHING * tip + (1 - SMOOTHING) * smooth_pos
    else:
        smooth_pos = None

    # -----------------------------------------------------------
    # DRAW mode logic
    # -----------------------------------------------------------
    if mode == "DRAW":
        pointing = (stable_gesture == POINT_LABEL) and smooth_pos is not None
        in_toolbar = smooth_pos is not None and smooth_pos[1] < TOOLBAR_H

        if pointing and not in_toolbar:
            px = int(np.clip(smooth_pos[0], 0, w - 1))
            py = int(np.clip(smooth_pos[1], 0, h - 1))
            if current_stroke is None:
                current_stroke = {
                    'color': (0, 0, 0) if eraser_on else brush_color,
                    'thickness': ERASER_THICKNESS if eraser_on else brush_thickness,
                    'points': [(px, py)],
                }
                strokes.append(current_stroke)
            else:
                current_stroke['points'].append((px, py))
        else:
            current_stroke = None  # pen up

        # Toolbar selection by pointing at a button and holding
        hovered = None
        if pointing and in_toolbar:
            hovered = button_at(smooth_pos[0], smooth_pos[1])

        if hovered is None:
            dwell_target = None
            dwell_done = False
        else:
            if hovered != dwell_target:
                dwell_target = hovered
                dwell_start = now
                dwell_done = False
            elif not dwell_done and now - dwell_start >= DWELL_TIME:
                kind, value = TOOLBAR_ITEMS[hovered]
                if kind == 'color':
                    brush_color = value
                    eraser_on = False
                elif kind == 'eraser':
                    eraser_on = not eraser_on
                elif kind == 'undo' and strokes:
                    strokes.pop()
                elif kind == 'clear':
                    strokes.clear()
                dwell_done = True

        # Composite the drawing over the camera image
        canvas = np.zeros_like(frame)
        draw_strokes(canvas, strokes)
        mask = canvas.any(axis=2)
        frame[mask] = canvas[mask]

    # -----------------------------------------------------------
    # Draw hand skeleton
    # -----------------------------------------------------------
    if landmarks is not None:
        for a, b in HAND_CONNECTIONS:
            pa = (int(landmarks[a][0]), int(landmarks[a][1]))
            pb = (int(landmarks[b][0]), int(landmarks[b][1]))
            cv2.line(frame, pa, pb, (0, 0, 255), 2)
        for pt in landmarks:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)

    # -----------------------------------------------------------
    # GAME mode logic + UI
    # -----------------------------------------------------------
    if mode == "GAME":
        if current_round_state == "COUNTDOWN":
            elapsed = now - countdown_start_time
            remaining = int(np.ceil(COUNTDOWN_DURATION - elapsed))

            if remaining > 0:
                cv2.putText(frame, f"GET READY: {remaining}", (w // 2 - 180, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)
            else:
                # Use the stabilised gesture so one bad frame can't decide the round
                if stable_gesture in ['rock', 'paper', 'scissors']:
                    user_move = stable_gesture
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
                else:
                    user_move = None
                    ai_move = None
                    result_text = "NO VALID GESTURE DETECTED!"
                current_round_state = "RESULT"

        # Scoreboard
        cv2.rectangle(frame, (0, 0), (w, 80), (30, 30, 30), cv2.FILLED)
        cv2.putText(frame, f"PLAYER: {player_score}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"AI: {ai_score}", (w - 180, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        if current_round_state == "RESULT":
            cv2.putText(frame, f"YOU: {user_move.upper() if user_move else 'N/A'}", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, f"AI: {ai_move.upper() if ai_move else 'N/A'}", (w - 250, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            color = (0, 255, 0) if "WIN" in result_text and "AI" not in result_text else (0, 0, 255)
            if "DRAW" in result_text:
                color = (0, 255, 255)

            cv2.putText(frame, result_text, (w // 2 - 150, h // 2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            cv2.putText(frame, "Press 's' to play next round", (w // 2 - 200, h - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        elif current_round_state == "WAITING":
            cv2.putText(frame, "Press 's' to Start Round!", (w // 2 - 220, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # -----------------------------------------------------------
    # DRAW mode UI (toolbar + cursor)
    # -----------------------------------------------------------
    else:
        cv2.rectangle(frame, (0, 0), (w, TOOLBAR_H), (30, 30, 30), cv2.FILLED)

        for i, (kind, value) in enumerate(TOOLBAR_ITEMS):
            x1, y1, x2, y2 = button_rect(i)
            if kind == 'color':
                cv2.rectangle(frame, (x1, y1), (x2, y2), value, cv2.FILLED)
                selected = (not eraser_on) and value == brush_color
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), cv2.FILLED)
                label = {'eraser': 'ERASE', 'undo': 'UNDO', 'clear': 'CLEAR'}[kind]
                cv2.putText(frame, label, (x1 + 4, y1 + 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                selected = (kind == 'eraser' and eraser_on)

            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (255, 255, 255) if selected else (120, 120, 120),
                          3 if selected else 1)

            # Dwell progress bar on the hovered button
            if dwell_target == i and not dwell_done:
                prog = min(1.0, (now - dwell_start) / DWELL_TIME)
                cv2.rectangle(frame, (x1, y2 - 6), (x1 + int(BTN_SIZE * prog), y2),
                              (0, 255, 255), cv2.FILLED)

        # Cursor at the fingertip
        if smooth_pos is not None:
            cx, cy = int(smooth_pos[0]), int(smooth_pos[1])
            if stable_gesture == POINT_LABEL:
                if eraser_on:
                    cv2.circle(frame, (cx, cy), ERASER_THICKNESS // 2, (255, 255, 255), 2)
                else:
                    cv2.circle(frame, (cx, cy), max(4, brush_thickness // 2 + 2), brush_color, 2)
            else:
                cv2.circle(frame, (cx, cy), 8, (180, 180, 180), 1)

        status = "DRAWING" if current_stroke is not None else "Pen up (point to draw)"
        cv2.putText(frame, status, (w - 300, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Common footer
    if detected_gesture:
        cv2.putText(frame, f"Detected: {str(detected_gesture).upper()}", (30, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Mode: {mode}  ('m' to switch)", (w - 330, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    cv2.imshow("AirControl", frame)

    # -----------------------------------------------------------
    # Keyboard
    # -----------------------------------------------------------
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        mode = "DRAW" if mode == "GAME" else "GAME"
        current_stroke = None
        dwell_target = None
        if mode == "GAME":
            current_round_state = "WAITING"
    elif mode == "GAME":
        if key == ord('s'):
            current_round_state = "COUNTDOWN"
            countdown_start_time = time.time()
        elif key == ord('r'):
            player_score = 0
            ai_score = 0
            current_round_state = "WAITING"
    else:  # DRAW
        if key == ord('c'):
            strokes.clear()
            current_stroke = None
        elif key == ord('z'):
            if strokes:
                strokes.pop()
            current_stroke = None
        elif key == ord('e'):
            eraser_on = not eraser_on
        elif key in (ord('+'), ord('=')):
            brush_thickness = min(40, brush_thickness + 2)
        elif key in (ord('-'), ord('_')):
            brush_thickness = max(2, brush_thickness - 2)
        elif key == ord('p'):
            os.makedirs(DRAWINGS_DIR, exist_ok=True)
            out = np.full((h, w, 3), 255, dtype=np.uint8)
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            draw_strokes(canvas, strokes)
            m = canvas.any(axis=2)
            out[m] = canvas[m]
            path = os.path.join(DRAWINGS_DIR, f"drawing_{int(time.time())}.png")
            cv2.imwrite(path, out)
            print(f"Saved drawing to {path}")

cap.release()
cv2.destroyAllWindows()