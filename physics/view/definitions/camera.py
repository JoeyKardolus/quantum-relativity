"""Camera: where the eye is and which way it looks, circling the cell's middle.

The eye sits `distance` from the middle, turned `yaw` about the upright axis and
`pitch` upward, and looks at the middle. `FOCAL` is how far the picture plane lies
before the eye, in half picture heights (a 40 degree view). The wheel moves the eye
between `NEAREST` and `FARTHEST` and no further.
"""
import numpy as np

FOCAL = 2.75
DISTANCE = 2.6   # the eye's first distance, in cell edges
LINE_DISTANCE = 4.6  # and for three cells in a row: the whole line fits a window 1000 by 800
YAW, PITCH = 0.2, 0.15
NEAREST, FARTHEST = 0.05, 200.0


def camera(yaw: float, pitch: float, distance: float) -> tuple:
    """The eye's place and its right, up and forward directions."""
    outward = np.array([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)])
    right = np.cross(-outward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    return distance * outward, right, np.cross(right, -outward), -outward


def zoomed(distance: float, notches: float) -> float:
    """The eye's distance after the wheel turned `notches`, kept in its range."""
    return float(np.clip(distance * 0.9 ** np.clip(notches, -50.0, 50.0), NEAREST, FARTHEST))
