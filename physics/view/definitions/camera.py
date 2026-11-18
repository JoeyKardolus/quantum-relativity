"""Camera: where the eye is and which way it looks, circling the cell's middle.

The eye sits `distance` from the middle, turned `yaw` about the upright axis and
`pitch` upward, and looks at the middle. `FOCAL` is how far the picture plane lies
before the eye, in half picture heights (a 40 degree view).
"""
import numpy as np

FOCAL = 2.75
DISTANCE = 2.6   # the eye's first distance, in cell edges
YAW, PITCH = 0.2, 0.15


def camera(yaw: float, pitch: float, distance: float) -> tuple:
    """The eye's place and its right, up and forward directions."""
    outward = np.array([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)])
    right = np.cross(-outward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    return distance * outward, right, np.cross(right, -outward), -outward
