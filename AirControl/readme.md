# Air Control

Control your computer with hand gestures, using nothing but a webcam.

<!-- Add a demo GIF here: ![demo](demo.gif) -->

## About the project

Air Control is a real-time hand gesture recognition system. It tracks your hand through the webcam, reads 21 landmarks (the wrist plus four points on each finger), and uses a classifier I trained on my own recorded data to figure out which gesture you are making.

Those gestures then control an air canvas: point with your index finger to draw, open your palm to erase, show a peace sign to change color, and make a fist to clear the screen.

## What makes it interesting

Instead of relying only on a pretrained model, I built the full pipeline myself:

- **Data collection:** I recorded my own labeled samples for each gesture, from different angles and distances.
- **Feature engineering:** the landmarks are normalized relative to the wrist and hand size, so gestures are recognized no matter where the hand is in the frame or how close it is to the camera.
- **Training and evaluation:** a scikit-learn classifier trained and tested on held-out data.
- **Real-time app:** predictions are smoothed over several frames so the output doesn't flicker.

## Results

<!-- Fill in after training -->

Test accuracy: _XX%_

## Built with

Python, OpenCV, NumPy, scikit-learn, and the palm detection and hand pose models from the OpenCV Zoo (converted from MediaPipe).


openCV/
├── data/
│   ├── raw_landmarks.csv         # Generated CSV output
│   └── dataset_collector.py      # Script to capture landmarks
│
├── src/
│   ├── __init__.py
│   ├── feature_engineering.py    # Preprocessing / normalization
│   ├── model.py                  # Training script
│   ├── gesture_classifier.py     # Inference logic
│   └── canvas.py                 # Drawing application state
│
├── models/
│   ├── gesture_model.pkl         # Trained model artifact
│   └── palm_detection_handpose.onnx   # The OpenCV Zoo ONNX model file
│
├── classes/                      # Existing class files/helpers
├── .gitignore
├── README.md
└── requirements.txt