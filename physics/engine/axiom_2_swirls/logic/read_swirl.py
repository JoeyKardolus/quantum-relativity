"""Read swirl: what a cell's fluid is doing, read off its boxes and never off a label.

A layer across the axis is square rings round it. Once round a ring the phase comes
back to itself after a whole number of turns: that number is the swirl.
"""
import numpy as np


def ring_places(across: int, ring: int) -> tuple[list[int], list[int]]:
    """The boxes of one square ring in order round the axis: their two places across it."""
    low, high = ring, across - 1 - ring
    up = list(range(low, high))
    down = list(range(high, low, -1))
    first = up + [high] * len(up) + down + [low] * len(up)
    second = [low] * len(up) + up + [high] * len(up) + down
    return first, second


def across_first(boxes: np.ndarray, laid: int) -> np.ndarray:
    """A block's numbers as a cube, the two directions across the swirl's axis first."""
    across, axis = round(boxes.size ** (1 / 3)), laid // 2
    cube = boxes.reshape((across,) * 3).astype(float)
    return np.moveaxis(cube, ((axis + 1) % 3, (axis + 2) % 3), (0, 1))


def phase_fall(angle: np.ndarray, places: tuple) -> np.ndarray:
    """How far the phase falls from each box of a ring to the next, between -pi and pi."""
    fall = angle[places] - np.roll(angle[places], -1, axis=0)
    return fall - 2 * np.pi * np.floor((fall + np.pi) / (2 * np.pi))


def whole_turns(phase: np.ndarray, laid: int) -> int | None:
    """The whole turns of the phase once round the axis; None when rings disagree."""
    angle, found = across_first(phase, laid), set()
    for ring in range(len(angle) // 2):
        fall = phase_fall(angle, ring_places(len(angle), ring))
        found.update(np.rint(-fall.sum(axis=0) / (2 * np.pi)))
    return int(found.pop()) if len(found) == 1 else None


def round_flow(quanta: np.ndarray, phase: np.ndarray, laid: int) -> float:
    """The fluid going round the axis, summed over the rings: above 0 the way the phase falls."""
    has, angle, flow = across_first(quanta, laid), across_first(phase, laid), 0.0
    for ring in range(len(has) // 2):
        places = ring_places(len(has), ring)
        between = np.sqrt(has[places] * np.roll(has[places], -1, axis=0))
        flow += float((between * np.sin(phase_fall(angle, places))).sum())
    return flow


def core_share(quanta: np.ndarray, laid: int) -> float:
    """The quanta of a box next to the axis, over the mean of a box."""
    has = across_first(quanta, laid)
    return float(has[ring_places(len(has), len(has) // 2 - 1)].mean() / has.mean())


def off_rule(quanta: np.ndarray, rule: np.ndarray) -> float:
    """How far the boxes are from the calm rule's; 1 is the counting noise of whole quanta."""
    return float(((quanta - rule) ** 2 / np.maximum(rule, 1)).mean())
