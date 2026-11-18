import math
from dataclasses import dataclass

# MediaPipe hand landmark indices
# see: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker
THUMB_TIP = 4
INDEX_TIP = 8


@dataclass
class FrameGestureResult:
    state: str = "idle"
    # rectangle as (x1, y1, x2, y2) in pixels placed on your two thumbs, set once locked
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
            a, b = hands[0], hands[1]
            anchors = [a[THUMB_TIP], b[THUMB_TIP]]
            self._update_rect(anchors)

        if self.locked:
            state = "locked"
        elif self.hold_count > 0:
            state = "locking"
        else: # to remove the rectangle just set the self.hold_count back to 0 
            #(when we do the gesture that removes the filter)
            state = "idle"

        if self.locked and self.rect:
            x1, y1, x2, y2 = self.rect
            rect = (round(x1), round(y1), round(x2), round(y2))
        else:
            rect = None
        
        return FrameGestureResult(state, rect, pinch)

    def _update_rect(self, anchors):
        (ax, ay), (bx, by) = anchors
        new_x1, new_y1 = min(ax, bx), min(ay, by)
        new_x2, new_y2 = max(ax, bx), max(ay, by)

        if self.rect is None or self.smoothing <= 0:
            self.rect = (new_x1, new_y1, new_x2, new_y2)
            return 

        s = self.smoothing
        old_x1, old_y1, old_x2, old_y2 = self.rect
        self.rect = (
            s * old_x1 + (1 - s) * new_x1,
            s * old_y1 + (1 - s) * new_y1,
            s * old_x2 + (1 - s) * new_x2,
            s * old_y2 + (1 - s) * new_y2,
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
