"""Through ends (test helper): the plus boxes at the ends of lines that run through their own shared joint.

Contract: `through_ends(quanta, found)` takes one universe's signed quanta and its
cells() result and returns a bool array of the quanta's shape: True for every plus box
at either end of a straight line along an axis from a plus box through one or more
minus boxes to a plus box of another energy volume, whose minus boxes all belong to the
one joint lying between exactly those two volumes (cells()["between"]): the boxes the
leak (leak.py) may change.
"""
import numpy as np


def through_ends(quanta: np.ndarray, found: dict) -> np.ndarray:
    """Plus boxes ending a line through the joint their two volumes share."""
    owner = found["owner"]
    between = found["between"]
    ends = np.zeros(quanta.shape, dtype=bool)
    for axis in range(3):
        signs = np.moveaxis(np.sign(quanta), axis, -1)
        owners = np.moveaxis(owner, axis, -1)
        marked = np.moveaxis(ends, axis, -1)
        for index in np.ndindex(signs.shape[:2]):
            line, line_owner = signs[index], owners[index]
            last_plus = -1
            for along, sign in enumerate(line):
                if sign == 0:
                    last_plus = -1
                elif sign > 0:
                    if last_plus >= 0 and along - last_plus > 1:
                        pair = sorted((int(line_owner[last_plus]), int(line_owner[along])))
                        middle = set(line_owner[last_plus + 1:along].tolist())
                        if len(middle) == 1 and sorted(between[middle.pop()].tolist()) == pair:
                            marked[index + (last_plus,)] = True
                            marked[index + (along,)] = True
                    last_plus = along
    return ends
