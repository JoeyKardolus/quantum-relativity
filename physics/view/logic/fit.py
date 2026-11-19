"""Fit: the volumes' middles that put every joint's two volumes the engine's distance apart.

A joint says how far apart its two volumes stand, not which way: the pattern's direction is
only where a new volume starts (`begun`: from a neighbour that is placed, the joint's
distance along its direction). `fitted` then moves all the middles at once so that each
joint's length is the engine's, counted as a share of that length: a short joint at the
centre weighs as much as a long one at the rim. One call aims every joint along the way it
points now and solves for the middles (`ROUNDS` rounds of conjugate gradients, from where
they stood), so a frame costs a few rounds and the cluster follows as it changes.

The engine's distances do agree, but only in a grid that crumples: toward the rim a layer
is longer round than its distance from the middle allows in flat room. So each joint is
also held a little (`HELD`) along its pattern direction: the grid stays a grid, and the
lengths come out some hundredths off, which the title says (`squeeze_read`).
"""
import numpy as np

ROUNDS = 20  # rounds of the solve a frame, a drawing number
HELD = 0.1   # how much a joint's pattern direction counts beside its length, a drawing number


def begun(spot: np.ndarray, first: np.ndarray, second: np.ndarray, aim: np.ndarray) -> np.ndarray:
    """Middles for the volumes that have none (not a number), outward from those that have."""
    while len(first):
        known = ~np.isnan(spot[:, 0])
        on, back = known[first] & ~known[second], known[second] & ~known[first]
        if not (on.any() or back.any()):
            break
        spot[second[on]] = spot[first[on]] + aim[on]
        spot[first[back]] = spot[second[back]] - aim[back]
    return spot


def _handed(first: np.ndarray, second: np.ndarray, pulls: np.ndarray, count: int) -> np.ndarray:
    """Each joint's pull given to its volume after and taken from the one before."""
    return np.stack([np.bincount(second, pull, count) - np.bincount(first, pull, count)
                     for pull in pulls.T], axis=1)


def fitted(spot: np.ndarray, first: np.ndarray, second: np.ndarray, apart: np.ndarray,
           aim: np.ndarray) -> np.ndarray:
    """The middles (volumes, 3) moved toward every joint's distance `apart`; `aim` is the
    way a joint points while its two volumes still lie in one place."""
    if not len(first):
        return spot
    gap = spot[second] - spot[first]
    long = np.linalg.norm(gap, axis=1, keepdims=True)
    points = apart[:, None] * gap / np.maximum(long, 1e-300)  # its length, the way it points now
    aim = np.where(long > 0, (points + HELD * aim) / (1 + HELD), aim)
    weight = (1 / apart ** 2)[:, None]
    heavy = np.maximum(np.bincount(np.concatenate([first, second]),
                                   np.tile(weight[:, 0], 2), len(spot)), 1e-300)[:, None]
    left = _handed(first, second, weight * (aim - gap), len(spot))
    eased = left / heavy
    way, power = eased.copy(), (left * eased).sum()
    for _ in range(ROUNDS):
        turned = _handed(first, second, weight * (way[second] - way[first]), len(spot))
        if power <= 0 or (way * turned).sum() <= 0:
            break
        far = power / (way * turned).sum()
        spot, left = spot + far * way, left - far * turned
        eased = left / heavy
        power, was = (left * eased).sum(), power
        way = eased + power / was * way
    return spot

