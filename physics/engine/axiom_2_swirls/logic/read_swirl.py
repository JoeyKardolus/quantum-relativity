"""Read swirl: what a cell's fluid is doing, read off its boxes and never off a label.

A layer across one of the three directions is square rings round the line through the
middle along it. Once round a ring the phase comes back to itself after a whole number of
turns: the whirl's turns about that direction. They are counted on the inner rings in the
outer layers: there the diagonal's line lies well outside the ring and fluid enough lies on it.
"""
import numpy as np


def ring_places(across: int, ring: int) -> tuple[list[int], list[int]]:
    """The boxes of one square ring in order round the line: their two places across it."""
    low, high = ring, across - 1 - ring
    up = list(range(low, high))
    down = list(range(high, low, -1))
    first = up + [high] * len(up) + down + [low] * len(up)
    second = [low] * len(up) + up + [high] * len(up) + down
    return first, second


def across_first(boxes: np.ndarray, axis: int) -> np.ndarray:
    """A block's numbers as a cube, the two directions across direction `axis` first."""
    across = round(boxes.size ** (1 / 3))
    cube = boxes.reshape((across,) * 3).astype(float)
    return np.moveaxis(cube, ((axis + 1) % 3, (axis + 2) % 3), (0, 1))


def phase_fall(angle: np.ndarray, places: tuple) -> np.ndarray:
    """How far the phase falls from each box of a ring to the next, between -pi and pi."""
    fall = angle[places] - np.roll(angle[places], -1, axis=0)
    return fall - 2 * np.pi * np.floor((fall + np.pi) / (2 * np.pi))


def rings_of(across: int) -> tuple[list, list]:
    """The rings to count on and their layers: the inner half of the rings, the outer quarter each side."""
    rings = [ring_places(across, ring) for ring in range((across // 2 + 1) // 2, across // 2)]
    return rings, [*range(across // 4), *range(across - across // 4, across)]


def whole_turns(phase: np.ndarray, axis: int) -> int | None:
    """The whole turns of the phase once round the line along `axis`; None when rings disagree."""
    angle, found = across_first(phase, axis), set()
    rings, layers = rings_of(len(angle))
    for places in rings:
        found.update(np.rint(-phase_fall(angle, places)[:, layers].sum(axis=0) / (2 * np.pi)))
    return int(found.pop()) if len(found) == 1 else None


def round_flow(quanta: np.ndarray, phase: np.ndarray, axis: int) -> float:
    """The fluid going round the line along `axis` on those rings: above 0 the way the phase falls."""
    has, angle, flow = across_first(np.abs(quanta), axis), across_first(phase, axis), 0.0
    rings, layers = rings_of(len(has))
    for places in rings:
        between = np.sqrt(has[places] * np.roll(has[places], -1, axis=0))
        flow += float((between * np.sin(phase_fall(angle, places)))[:, layers].sum())
    return flow


def core_share(quanta: np.ndarray, axis: int) -> float:
    """The quanta of a box next to the line along `axis`, over the mean of a box."""
    has = across_first(np.abs(quanta), axis)
    return float(has[ring_places(len(has), len(has) // 2 - 1)].mean() / has.mean())


def off_rule(quanta: np.ndarray, rule: np.ndarray) -> float:
    """How far the boxes are from the calm rule's; 1 is the counting noise of whole quanta."""
    return float(((quanta - rule) ** 2 / np.maximum(np.abs(rule), 1)).mean())


def flow_along(quanta: np.ndarray, phase: np.ndarray, axis: int) -> float:
    """The fluid going up a block's direction `axis`, summed over the faces between its boxes."""
    across = round(quanta.size ** (1 / 3))
    has = np.moveaxis(np.abs(quanta).reshape((across,) * 3).astype(float), axis, 0)
    angle = np.moveaxis(phase.reshape((across,) * 3).astype(float), axis, 0)
    return float((np.sqrt(has[:-1] * has[1:]) * np.sin(angle[:-1] - angle[1:])).sum())
