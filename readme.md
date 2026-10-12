# AirControl

Control your computer with hand gestures, using only a webcam. AirControl tracks your hand in real time, recognizes gestures with a small machine learning model, and has two modes:

- **Game mode:** play Rock Paper Scissors against the computer.
- **Draw mode:** draw on the screen in the air with your index finger, and erase with an open palm.

> **Ready to run:** the project already comes with a trained gesture model, `models/rps_point_model.pkl`. You can start the app right away. Collecting data and training are only needed if you want to retrain the model on your own hands.

## How it works

The app is a three-stage pipeline that runs on every camera frame:

1. **Palm detection:** an ONNX model (MediaPipe palm detector) finds where the hand is.
2. **Hand pose estimation:** a second ONNX model returns 21 landmarks (x, y, z) for the hand, plus a bounding box and a confidence score.
3. **Gesture classification:** the landmarks are normalized (wrist as origin, scaled by hand size) and passed to a scikit-learn classifier that predicts the gesture.

Normalizing the landmarks makes the prediction independent of where the hand is in the frame and how far it is from the camera.

## Data flow

![AirControl data flow](docs/dataflow.svg)

The two MediaPipe ONNX models (orange) are used in the same way in data collection and in the live app. Their output is normalized and then either saved to the CSV (collection) or sent to the classifier (live app).

## Project structure

```
AirControl/
├── data/
│   ├── dataset_collector.py       # record labeled gesture samples
│   └── raw_landmarks.csv          # collected training data (normalized landmarks)
├── models/
│   ├── palm_detection_mediapipe_2023feb.onnx
│   ├── handpose_estimation_mediapipe_2023feb.onnx
│   └── rps_point_model.pkl        # trained gesture classifier (included)
├── src/
│   ├── app.py                     # the main app (game + draw modes)
│   ├── train.py                   # trains the gesture classifier
│   ├── mp_palmdet.py              # palm detector wrapper (OpenCV Zoo)
│   └── mp_handpose.py             # hand pose wrapper (OpenCV Zoo)
├── docs/
│   └── dataflow.svg               # data flow diagram used in this README
├── drawings/                      # saved drawings (created automatically)
└── readme.md
```

## Requirements

- Python 3.9 or newer
- A webcam
- Python packages: `opencv-python`, `numpy`, `scikit-learn`, `pandas`, `joblib`
- The two ONNX models and their wrapper files (`mp_palmdet.py`, `mp_handpose.py`) from the [OpenCV Zoo](https://github.com/opencv/opencv_zoo) (palm detection and hand pose estimation, MediaPipe versions)

## Installation

```bash
# macOS example, using Homebrew's Python
brew install python
python3 -m pip install --break-system-packages opencv-python numpy scikit-learn pandas joblib
```

If you prefer, install the packages in a virtual environment instead (`python3 -m venv venv && source venv/bin/activate`, then `pip install ...`).

Then place the ONNX models in `models/` and the two wrapper files in `src/`, as shown in the project structure. Each script finds the project folder from its own location, so the paths work no matter where you run it from.

On macOS, the first run will ask for camera permission for your terminal.

## Usage

Run these commands from the `AirControl` folder.

### Run the app (quick start)

```bash
python3 src/app.py
```

This uses the included `models/rps_point_model.pkl`, so there is nothing to train first. Press `m` at any time to switch between game and draw mode.

### Retrain on your own data (optional)

Only needed if you want the model to fit your own hands or add gestures. First collect data, then train.

#### Step 1: collect training data

```bash
python3 data/dataset_collector.py
```

Show a gesture to the camera and press its key to save a sample. A sample is only saved when a hand is detected with confidence of at least 0.8. Press `q` to quit.

The keys are defined in `LABEL_MAP` at the top of `dataset_collector.py`. The labels you record must match the ones the app expects: `rock`, `paper`, `scissors` and `point`.

Tips for good data:
- Record 300-500 samples per gesture.
- Vary the hand position, distance from the camera, angle and lighting.
- Use both hands.
- Add an idle class (relaxed hand or half-formed gestures) so the model doesn't force every hand shape into one of the gestures.

The CSV has the columns `label, x_0, y_0, z_0, ..., x_20, y_20, z_20`. Values are already normalized, and the wrist columns (`x_0`, `y_0`, `z_0`) are always 0.

#### Step 2: train the model

```bash
python3 src/train.py
```

This reads `data/raw_landmarks.csv`, trains the gesture classifier and saves it to `models/rps_point_model.pkl`.

**What the app expects from the model:** a scikit-learn classifier saved with joblib, which takes the full 63-value normalized landmark vector (wrist included) and predicts one of `rock`, `paper`, `scissors` or `point`.

## Game mode

Rock Paper Scissors against the computer.

1. Press `s` to start a round.
2. A 3-second countdown runs. Make your gesture before it ends.
3. Your move is read from the last few frames (not a single frame), the computer picks randomly, and the result and score are shown.

| Key | Action |
|-----|--------|
| `s` | start a round |
| `r` | reset the score |
| `m` | switch to draw mode |
| `q` | quit |

## Draw mode

| Gesture | Action |
|---------|--------|
| `point` | draw with the index fingertip |
| `paper` (open palm) | erase a large area under the palm. The eraser size follows the size of your hand |
| any other gesture | pen up (nothing is drawn) |

**Toolbar:** point at a button in the top bar and hold for about 0.8 seconds to select it (a progress bar fills while you hold). It has five colors, a small eraser, undo and clear. The pen doesn't draw while you are over the toolbar.

| Key | Action |
|-----|--------|
| `c` | clear everything |
| `z` | undo the last stroke |
| `e` | toggle the small eraser |
| `+` / `-` | change the brush thickness |
| `p` | save the drawing as a PNG in `drawings/` (white background) |
| `m` | switch to game mode |
| `q` | quit |

## Configuration

Settings are constants near the top of `app.py`:

| Setting | Default | What it does |
|---------|---------|--------------|
| `CONF_THRESHOLD` | `0.8` | minimum hand-detection confidence |
| `SMOOTHING` | `0.5` | cursor smoothing (lower = smoother, but with more lag) |
| `DWELL_TIME` | `0.8` | seconds to hold on a toolbar button to select it |
| `PALM_ERASER_SCALE` | `0.9` | palm eraser size relative to hand size |
| `PALM_ERASER_MIN` / `MAX` | `30` / `140` | palm eraser size limits (pixels) |
| `POINT_LABEL` | `'point'` | label used for the drawing gesture |
| `ERASE_GESTURE` | `'paper'` | label used for the palm eraser |

## Troubleshooting

- **Hand points are scattered:** make sure you read the landmarks from `[4:67]` of the hand pose output, which has 132 values. The first four values are the bounding box.
- **Predictions are wrong in the app but fine in training:** the app must use the same `normalize()` function and the same horizontal flip as the data collector.
- **Only some gestures are recognized:** collect more varied samples for the weak gestures and check the confusion matrix from training.
- **Cursor is jittery or lags:** adjust `SMOOTHING`.
- **`pip: command not found` on macOS:** use `python3 -m pip` instead.
- **Camera doesn't open:** check camera permissions for your terminal, or change the index in `cv2.VideoCapture(0)`.

## Credits

Palm detection and hand pose models come from MediaPipe, via the [OpenCV Zoo](https://github.com/opencv/opencv_zoo).