"""Crossings (test helper): count the leak's straight crossings and those that go through their own joint.

Contract: `crossings(quanta, found)` takes one universe's signed quanta and its cells()
result and returns (through, total): total counts every straight line along an axis
from a plus box through one or more minus boxes to a plus box of a different energy
volume (the leak's crossing, leak.py); through counts those whose minus boxes all
belong to one joint lying between exactly those two volumes (cells()["between"]).
"""
import numpy as np


def crossings(quanta: np.ndarray, found: dict) -> tuple:
    """(crossings through their own shared joint, all crossings between two volumes)."""
    owner = found["owner"]
    between = found["between"]
    through = total = 0
    for axis in range(3):
        signs = np.moveaxis(np.sign(quanta), axis, -1).reshape(-1, quanta.shape[axis])
        owners = np.moveaxis(owner, axis, -1).reshape(-1, quanta.shape[axis])
        for line, line_owner in zip(signs, owners):
            last_plus = -1
            for along, sign in enumerate(line):
                if sign == 0:
                    last_plus = -1
                elif sign > 0:
                    if last_plus >= 0 and along - last_plus > 1:
                        ends = sorted((int(line_owner[last_plus]), int(line_owner[along])))
                        if ends[0] != ends[1]:
                            total += 1
                            middle = set(line_owner[last_plus + 1:along].tolist())
                            if len(middle) == 1 and sorted(between[middle.pop()].tolist()) == ends:
                                through += 1
                    last_plus = along
    return through, total
