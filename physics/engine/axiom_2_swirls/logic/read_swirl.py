"""Read swirl: what a cell's fluid is doing, read off its boxes and never off a label.

Each of a cell's three rings is rows of boxes that go once round its core and close. Once
round a row the phase comes back to itself after a whole number of turns: the whirl's
turns in that ring. The phase falls the way the fluid goes, so a volume's whirl reads -1
in each ring and a joint's, its mirror, +1.
"""
import numpy as np


def rows_of(boxes: np.ndarray, ring: int) -> np.ndarray:
    """One ring's numbers as (around, rows): each column a row of boxes once round the core."""
    across = round(boxes.size ** (1 / 3))
    block = boxes.reshape(3, across // 3, across, across)[ring].astype(float)
    return np.moveaxis(block, 1, 0).reshape(across, -1)


def phase_fall(angle: np.ndarray) -> np.ndarray:
    """How far the phase falls from each box of a row to the next, between -pi and pi."""
    fall = angle - np.roll(angle, -1, axis=0)
    return fall - 2 * np.pi * np.floor((fall + np.pi) / (2 * np.pi))


def whole_turns(phase: np.ndarray, ring: int) -> int | None:
    """The whole turns of the phase once round the core of `ring`; None when its rows disagree."""
    found = set(np.rint(-phase_fall(rows_of(phase, ring)).sum(axis=0) / (2 * np.pi)))
    return int(found.pop()) if len(found) == 1 else None


def round_flow(quanta: np.ndarray, phase: np.ndarray, ring: int) -> float:
    """The fluid going round the core of `ring`, a link: above 0 the way a volume's whirl goes."""
    has = rows_of(np.abs(quanta), ring)
    between = np.sqrt(has * np.roll(has, -1, axis=0))
    return float((between * np.sin(phase_fall(rows_of(phase, ring)))).sum() / len(has))


def flow_along(quanta: np.ndarray, phase: np.ndarray, ring: int) -> float:
    """The fluid of `ring` going from its back cap to its front cap, by both ways round the core."""
    has, fall = rows_of(np.abs(quanta), ring), phase_fall(rows_of(phase, ring))
    flow = (np.sqrt(has * np.roll(has, -1, axis=0)) * np.sin(fall)).sum(axis=1)
    back = np.roll(flow, len(flow) // 3)  # the links from the back cap through the jet to the front
    return float(back[:len(flow) // 2].sum() - back[len(flow) // 2:].sum())


def still_share(quanta: np.ndarray, phase: np.ndarray) -> float:
    """The share of a cell's links round its cores where the fluid stands or goes back."""
    moves = [np.sin(phase_fall(rows_of(phase, ring))) * np.sign(quanta.sum()) for ring in range(3)]
    return float(np.mean(np.concatenate(moves) <= 0))


def off_rule(quanta: np.ndarray, rule: np.ndarray) -> float:
    """How far the boxes are from the calm rule's; 1 is the counting noise of whole quanta."""
    return float(((quanta - rule) ** 2 / np.maximum(np.abs(rule), 1)).mean())
