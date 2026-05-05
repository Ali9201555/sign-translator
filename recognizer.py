"""Rule-based ASL fingerspelling and common-sign recognizer.

Works out of the box with no training - everything is derived from the
geometry of MediaPipe's 21 hand landmarks.

Reliably handles:
  Letters: A, B, D, F, I, L, U, V, W, Y
  Signs:   "yes" (thumbs up), "hello" (open palm), "I love you" (index+pinky+thumb)
"""
import numpy as np

# MediaPipe hand-landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


def _angle(v1, v2):
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def _finger_curl(pts, mcp, pip, tip):
    # Angle between MCP->PIP and PIP->TIP segments. 0 = straight, ~pi = fully curled.
    return _angle(pts[pip] - pts[mcp], pts[tip] - pts[pip])


def _finger_extended(pts, mcp, pip, tip, thresh=0.6):
    return _finger_curl(pts, mcp, pip, tip) < thresh


def _thumb_extended(pts):
    # Thumb is "extended" when the tip sits well away from the index-MCP knuckle.
    palm = np.linalg.norm(pts[INDEX_MCP] - pts[WRIST]) + 1e-9
    return np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_MCP]) > 0.55 * palm


def classify(pts):
    """Take a (21, 3) hand-landmark array, return (label, confidence)."""
    thumb = _thumb_extended(pts)
    index = _finger_extended(pts, INDEX_MCP, INDEX_PIP, INDEX_TIP)
    middle = _finger_extended(pts, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP)
    ring = _finger_extended(pts, RING_MCP, RING_PIP, RING_TIP)
    pinky = _finger_extended(pts, PINKY_MCP, PINKY_PIP, PINKY_TIP)

    pattern = (index, middle, ring, pinky)
    palm = np.linalg.norm(pts[INDEX_MCP] - pts[WRIST]) + 1e-9

    # Fist (no fingers up)
    if pattern == (False, False, False, False):
        if thumb:
            return ("yes", 0.85)        # thumbs up
        return ("A", 0.7)               # ASL A: closed fist, thumb on side

    # Index up
    if pattern == (True, False, False, False):
        if thumb:
            return ("L", 0.9)
        return ("D", 0.8)

    # Index + middle up
    if pattern == (True, True, False, False):
        spread = np.linalg.norm(pts[INDEX_TIP] - pts[MIDDLE_TIP])
        ref = np.linalg.norm(pts[INDEX_MCP] - pts[MIDDLE_MCP])
        if spread > 1.8 * ref:
            return ("V", 0.85)
        return ("U", 0.8)

    # Index + middle + ring up
    if pattern == (True, True, True, False):
        return ("W", 0.85)

    # All four up
    if pattern == (True, True, True, True):
        if thumb:
            return ("hello", 0.85)      # open palm / wave
        return ("B", 0.85)

    # Pinky up
    if pattern == (False, False, False, True):
        if thumb:
            return ("Y", 0.9)
        return ("I", 0.85)

    # Index + pinky up (with thumb = "I love you")
    if pattern == (True, False, False, True):
        if thumb:
            return ("I love you", 0.9)
        return ("?", 0.4)

    # Middle + ring + pinky up, thumb pinching index (F)
    if pattern == (False, True, True, True):
        pinch = np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP])
        if pinch < 0.4 * palm:
            return ("F", 0.85)
        return ("?", 0.4)

    return ("?", 0.0)
