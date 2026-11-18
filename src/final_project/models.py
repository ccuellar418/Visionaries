import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

# max number of hands to detect
MAX_NUM_HANDS = 2
# min detection/tracking confidence
MIN_CONFIDENCE = 0.5
# mediapipe task file location
MODEL_PATH = str(Path(__file__).resolve().parents[2] / "models" / "gesture_recognizer.task")

# main MediaPipe hand gesture recognition model
gesture_recognizer = vision.GestureRecognizer.create_from_options(vision.GestureRecognizerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=MAX_NUM_HANDS,
    min_hand_detection_confidence=MIN_CONFIDENCE,
    min_hand_presence_confidence=MIN_CONFIDENCE,
    min_tracking_confidence=MIN_CONFIDENCE,
))