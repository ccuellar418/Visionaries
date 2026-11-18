import cv2
from mediapipe.tasks.python import vision

# Google sample code. Source: ???
def draw_hand(frame, landmarks):
    """
    Draws hand structures on the current frame from MediaPipe hand landmark data.
    """
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
def handle_mediapipe_predictions(frame, predictions):
    """
    Per-frame MediaPipe prediction handler.
    """
    for i, landmarks in enumerate(predictions.hand_landmarks):
        points = draw_hand(frame, landmarks)
        # label each hand with its handedness (i.e., left/right) and highest-rated gesture prediction
        handedness = predictions.handedness[i][0].category_name
        label = handedness
        if predictions.gestures and predictions.gestures[i]:
            gesture = predictions.gestures[i][0]
            label = f"{handedness}: {gesture.category_name} ({gesture.score:.2f})"

        x = min(p[0] for p in points)
        y = min(p[1] for p in points) - 10
        cv2.putText(frame, label, (x, max(y, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)