import argparse
import cv2
import imageio as iio
import mediapipe as mp
import time

from models import gesture_recognizer
from pathlib import Path

DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "gesture_data.csv"

# builds a csv with "landmark_x", "landmark_y", "landmark_z" for each landmark in LANDMARKS 
def _build_csv_header():
    LANDMARKS = "WRIST,THUMB_CMC,THUMB_MCP,THUMB_IP,THUMB_TIP,INDEX_FINGER_MCP,INDEX_FINGER_PIP,INDEX_FINGER_DIP,INDEX_FINGER_TIP,MIDDLE_FINGER_MCP,MIDDLE_FINGER_PIP,MIDDLE_FINGER_DIP,MIDDLE_FINGER_TIP,RING_FINGER_MCP,RING_FINGER_PIP,RING_FINGER_DIP,RING_FINGER_TIP,PINKY_MCP,PINKY_PIP,PINKY_DIP,PINKY_TIP"
    return "".join(list(map(lambda l: f"{l}_x,{l}_y,{l}_z,", LANDMARKS.split(",")))) + "Gesture"

# predicts
def _build_csv(prediction, gesture_name):
    observations = []
    for i, (gesture, landmarks) in enumerate(zip(prediction.gestures, prediction.hand_landmarks)):
        # if no gestures/landmarks detected, continue
        if not gesture or not landmarks:
            continue

        # the gesture of current frame to write to the csv file (by default, the gesture in the current video)
        observed_gesture = gesture_name

        # if mediapipe gestures were found in the current frame, use them instead.
        # otherwise, the current row will contain the gesture in the video
        if (predicted_gesture := gesture[0].category_name) != "None":
            observed_gesture = predicted_gesture

        observation = ""
        # add each landmark's (x, y, z) and then the gesture's name
        for l in landmarks:
            observation += f"{l.x},{l.y},{l.z},"

        observation += observed_gesture + "\n"
        observations.append(observation)

    return observations

def main():
    arg_parser = argparse.ArgumentParser(description="Custom landmark dataset creation script")
    arg_parser.add_argument("-g", "--gesture_name", help="The name of the gesture in the video", required=True)
    arg_parser.add_argument("-v", "--video_path", help="The absolute video path", required=True)

    args  = arg_parser.parse_args()
    start = time.monotonic()
    observations = []

    with gesture_recognizer:
        for i, frame in enumerate(iio.imiter(args.video_path)):
            print(f"Processing frame {i}", end="\r")
            frame        = cv2.flip(frame, 1)
            mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp_ms = int((time.monotonic() - start) * 1000)
            prediction   = gesture_recognizer.recognize_for_video(mp_image, timestamp_ms)
            # add each row (observation) to the final list
            observations.extend(_build_csv(prediction, args.gesture_name))

    # create dataset file if it doesn't exist and write the header
    if not DATASET_PATH.exists():
        with open(DATASET_PATH, mode="a") as csv:
            csv.write(_build_csv_header() + "\n")

    # write rows once
    with open(DATASET_PATH, mode="a") as csv:
        csv.writelines(observations)

if __name__ == "__main__":
    main()