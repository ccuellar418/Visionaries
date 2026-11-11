import cv2
import mediapipe as mp
import sys
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# max number of hands to detect
MAX_NUM_HANDS = 2
# min detection/tracking confidence
MIN_CONFIDENCE = 0.5
# mediapipe task file location
MODEL_PATH = "../../models/gesture_recognizer.task"

# main hand gesture model
model = vision.GestureRecognizer.create_from_options(vision.GestureRecognizerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=MAX_NUM_HANDS,
    min_hand_detection_confidence=MIN_CONFIDENCE,
    min_hand_presence_confidence=MIN_CONFIDENCE,
    min_tracking_confidence=MIN_CONFIDENCE,
))

# Google sample code. Source: ???
def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    # coordinates are normalized (i.e., in the range of 0-1), so we must convert them back to pixel coordinates
    points = [(int(l.x * w), int(l.y * h)) for l in landmarks]
    # draw line between each landmark
    for conn in vision.HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, points[conn.start], points[conn.end], (255, 255, 255), 2)
    # draw circle for each point
    for p in points:
        cv2.circle(frame, p, 8, (0, 0, 255), -1)
    return points

# Google sample code. Source: ???
def handle_results(frame, result):
    for i, landmarks in enumerate(result.hand_landmarks):
        points = draw_hand(frame, landmarks)
        # label each hand with its handedness (i.e., left/right) and highest-rated gesture prediction
        handedness = result.handedness[i][0].category_name
        label = handedness
        if result.gestures and result.gestures[i]:
            gesture = result.gestures[i][0]
            label = f"{handedness}: {gesture.category_name} ({gesture.score:.2f})"

        x = min(p[0] for p in points)
        y = min(p[1] for p in points) - 10
        cv2.putText(frame, label, (x, max(y, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"Error opening camera, exiting...")
        sys.exit(1)
        
    start = time.monotonic()
    with model:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # create mediapipe image from mirrored view
            # note: must convert from BGR to RGB first, since opencv returns BGR images
            frame = cv2.flip(frame, 1)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # video mode needs a steadily increasing timestamp in ms.
            # see: https://developers.google.com/edge/api/mediapipe/js/tasks-vision.gesturerecognizer#gesturerecognizerrecognizeforvideo_gesturerecognizerrecognizeforvideo
            timestamp_ms = int((time.monotonic() - start) * 1000)

            """
            predictions look like this:
            gestures = [[Category(category_name="Thumb_Up", score=0.95)]]
            handedness = [[Category(category_name="Right", score=0.99)]]
            hand_landmarks = [[NormalizedLandmark(x=0.5, y=0.6, z=0.0), + 20 more]] 
            hand_world_landmarks = [[Landmark(x=0.02, y=0.04, z=-0.01), + 20 more]]
            """
            # see: https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer#task_output
            prediction = model.recognize_for_video(mp_image, timestamp_ms)

            # draw hand & labels on each frame w/ the predicted results
            handle_results(frame, prediction)
            cv2.imshow("Gesture Recognition", frame)

            # press q to quit
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()