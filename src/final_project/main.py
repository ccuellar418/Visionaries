import cv2
import mediapipe as mp
import sys
import time

from editing import apply_filter, rectangle_mask
from gesture_detection import FrameGestureDetector, draw_rectangle
from handlers import handle_mediapipe_predictions
from models import gesture_recognizer

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"Error opening camera, exiting...")
        sys.exit(1)
        
    frame_gesture = FrameGestureDetector()
    start = time.monotonic()
    with gesture_recognizer:
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
            prediction = gesture_recognizer.recognize_for_video(mp_image, timestamp_ms)

            # detect the two-handed frame gesture and the pinch
            gesture_result = frame_gesture.update(prediction, frame.shape)

            # apply the filter inside the rectangle once it's locked
            if gesture_result.rect is not None:
                mask = rectangle_mask(frame, *gesture_result.rect)
                frame = apply_filter(frame, mask, cv2.bitwise_not)
            
            # draw hand & labels on each frame w/ the predicted results
            handle_mediapipe_predictions(frame, prediction)
            draw_rectangle(frame, gesture_result)

            # show the current gesture state in the top left corner
            if gesture_result.pinch:
                state_label = "freeze"
            else:
                state_label = gesture_result.state            
            cv2.putText(frame, state_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.imshow("Gesture Recognition", frame)

            # press q to quit
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
