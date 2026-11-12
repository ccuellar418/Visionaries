import numpy as np

#function to apply rectangular mask using coordinate of top corner of rectangle and bottom corner
def rectangle_mask(frame, x, y, x2, y2):
    h, w = frame.shape[:2]

    #find top and bottom and left vs right
    x, x2 = sorted((x, x2))
    y, y2 = sorted((y, y2))

    #corners max at image boundaries
    x1 = max(0, min(w, round(x)))
    y1 = max(0, min(h, round(y)))
    x2 = max(0, min(w, round(x2)))
    y2 = max(0, min(h, round(y2)))

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask

#apply filter to frame within mask using any filter function
def apply_filter(frame, mask, filter_func):
    filtered = filter_func(frame.copy())
    output = frame.copy()
    selected = mask != 0
    
    output[selected] = filtered[selected]
    return output