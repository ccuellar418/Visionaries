# Visionaries

Computer Vision Project that we believe is really cool!

## Getting started

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and open a terminal in `Visionaries`.
2. Run `uv sync` to install dependencies.
3. Go to [MediaPipe Gesture Recognizer page](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer) and download the Gesture Recognizer model from its Models section. Create a folder named `models` inside `Visionaries` (alongside `src`), then move the downloaded `gesture_recognizer.task` file into it. The final location should be `Visionaries/models/gesture_recognizer.task`

4. Start the final project:

   ```sh
   cd src/final_project
   uv run python main.py
   ```