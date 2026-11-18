import math
from dataclasses import dataclass
import cv2

# MediaPipe hand landmark indices
# see: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker
THUMB_TIP = 4
INDEX_TIP = 8
THUMB_NEAR_WEB = 2
INDEX_NEAR_WEB = 5


@dataclass
class FrameGestureResult:
    state: str = "idle"
    # rectangle as (x1, y1, x2, y2) in pixels placed on your two thumb-index webs, set once locked
    rect: tuple | None = None
    # True while a pinch is being activated
    pinch: bool = False

#This is used for gesture detection
class FrameGestureDetector:
    def __init__(
        self,
        touch_px=50,       # max tip-to-tip distance, in pixels, to count as touching
        pinch_px=25,       # same idea, for thumb-to-index on one hand
        lock_frames=10,     # frames the frame gesture must be held before confirming
        pinch_frames=5,    # frames a pinch must be held before confirming
        smoothing=0.5,     #mess w this one# 0 = no smoothing, closer to 1 = smoother but laggier
    ):
        self.touch_px = touch_px
        self.pinch_px = pinch_px
        self.lock_frames = lock_frames
        self.pinch_frames = pinch_frames
        self.smoothing = smoothing 
        self.idle_state()

    def idle_state(self):
        """Drop the rectangle and go back to waiting for the frame gesture."""
        self.locked = False
        self.hold_count = 0
        self.pinch_count = 0
        self.rect = None

    def confirm_rectangle(self, prediction, frame_shape):
        """
        Frame gesture detection only. Returns True once both thumb-to-index connections
        have been held for lock_frames consecutive frames.
        """
        hands = _hands_in_pixels(prediction, frame_shape)
        if len(hands) < 2: #need 2 hands to make arectangle
            self.hold_count = 0
            return False

        a, b = hands[0], hands[1]

        # are these two within touching distance? boolean style
        touch1 = math.dist(a[THUMB_TIP], b[INDEX_TIP]) < self.touch_px
        touch2 = math.dist(b[THUMB_TIP], a[INDEX_TIP]) < self.touch_px

        if touch1 and touch2: 
            self.hold_count += 1
        else:
            self.hold_count = 0
        return self.hold_count >= self.lock_frames

    def confirm_pinch(self, prediction, frame_shape):
        """
        Pinch detection only. Returns True once any visible hand has held its own
        thumb tip on its own index tip for pinch_frames consecutive frames.
        Works with just one hand in view.
        """
        pinching = False
        for hand in _hands_in_pixels(prediction, frame_shape):
            if math.dist(hand[THUMB_TIP], hand[INDEX_TIP]) < self.pinch_px:
                pinching = True
                break  # one pinching hand is good to switch the boolean
        if pinching:
            self.pinch_count += 1
        else:
            self.pinch_count = 0
        return self.pinch_count >= self.pinch_frames

    def update(self, prediction, frame_shape):
        """Call once per frame with the MediaPipe result. Returns a FrameGestureResult."""
        pinch = False
        if not self.locked:
            self.locked = self.confirm_rectangle(prediction, frame_shape)
        else: # if locked, start checking to see if they start pinching
            pinch = self.confirm_pinch(prediction, frame_shape)

        # once locked, follow the anchors whenever both hands are visible;
        # if a hand drops out, keep the last rectangle
        hands = _hands_in_pixels(prediction, frame_shape)
        if self.locked and len(hands) >= 2:
            self._update_rect(hands[0],hands[1])

        if self.locked:
            state = "locked"
        elif self.hold_count > 0:
            state = "locking"
        else: 
            state = "idle"

        if self.locked and self.rect:
            x1, y1, x2, y2 = self.rect
            rect = (round(x1), round(y1), round(x2), round(y2))
        else:
            rect = None
        
        return FrameGestureResult(state, rect, pinch)

    def _update_rect(self, a, b):
        web_a = _midpoint(a[THUMB_NEAR_WEB], a[INDEX_NEAR_WEB])
        web_b = _midpoint(b[THUMB_NEAR_WEB], b[INDEX_NEAR_WEB])

        # web_{a|b}[0] = horizontal coordinate
        # web_{a|b}[1] = vertical coordinate
        left = min(web_a[0], web_b[0])
        right = max(web_a[0], web_b[0])
        top = min(web_a[1], web_b[1])
        bottom = max(web_a[1], web_b[1])

        if self.rect is None or self.smoothing <= 0:
            self.rect = (left, top, right, bottom)
            return
        
        s = self.smoothing
        old_left, old_top, old_right, old_bottom = self.rect
        self.rect = (
            s * old_left + (1 - s) * left,
            s * old_top + (1 - s) * top,
            s * old_right + (1 - s) * right,
            s * old_bottom + (1 - s) * bottom,
        )


def _hands_in_pixels(prediction, frame_shape):
    """Return every detected hand as a list of (x, y) pixel points."""
    h, w = frame_shape[:2]
    hands = []
    for hand in prediction.hand_landmarks:
        # give the hands their landmark points
        points = [(landmark.x * w, landmark.y * h) for landmark in hand]
        hands.append(points)
    return hands  

def _midpoint(p, q):
    return ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)          

def draw_rectangle(frame, result):
    """Draw the rectangle outline once it's locked."""
    if result.rect is None:
        return frame
    x1, y1, x2, y2 = result.rect
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return frame