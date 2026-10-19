"""Small OpenCV helpers that draw on a frame.

All functions draw *in place* on a NumPy image (BGR order, like OpenCV).
"""
import cv2
import numpy as np

# Colours in BGR. A tracked object keeps the same colour because the colour
# is chosen from its track ID.
PALETTE = [
    (56, 56, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
    (10, 249, 72), (134, 219, 61), (187, 212, 0), (255, 194, 0),
    (255, 115, 100), (236, 24, 0), (255, 56, 132), (255, 56, 203),
    (199, 55, 255), (200, 149, 255), (23, 204, 146), (168, 153, 44),
]


def color_for(key: int):
    """Pick a stable colour for a track ID or class ID."""
    return PALETTE[int(key) % len(PALETTE)]


def _scale(frame):
    h, w = frame.shape[:2]
    return max(0.5, min(h, w) / 900)


def draw_box(frame, box, label, color):
    """Draw a bounding box with a filled label tag above it."""
    x1, y1, x2, y2 = [int(v) for v in box]
    scale = _scale(frame)
    thickness = max(1, int(round(scale * 2)))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    font_scale = scale * 0.6
    text_thick = max(1, thickness - 1)
    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thick
    )
    pad = 3
    tag_h = th + baseline + 2 * pad
    top = y1 - tag_h
    if top < 0:  # box touches the top edge: put the tag inside the box
        top = y1
    cv2.rectangle(frame, (x1, top), (x1 + tw + 2 * pad, top + tag_h), color, -1)

    # Choose black or white text depending on how bright the colour is.
    b, g, r = color
    text_color = (0, 0, 0) if (0.114 * b + 0.587 * g + 0.299 * r) > 150 else (255, 255, 255)
    cv2.putText(
        frame, label, (x1 + pad, top + th + pad),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, text_thick, cv2.LINE_AA,
    )


def draw_trail(frame, points, color):
    """Draw a line through an object's recent centre positions."""
    if len(points) < 2:
        return
    pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts], False, color, max(1, int(round(_scale(frame) * 2))), cv2.LINE_AA)


def draw_hud(frame, lines):
    """Write short status lines (FPS, counts) in the top-left corner."""
    scale = _scale(frame) * 0.6
    thick = max(1, int(round(scale * 2)))
    y = int(24 * _scale(frame))
    for line in lines:
        # Dark outline first, then white text on top: readable on any background.
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
        y += int(26 * _scale(frame))
