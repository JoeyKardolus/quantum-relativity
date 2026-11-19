"""Census: where a grown universe's energy is, read on the host for a far run's log. No step reads it.

`gatherings` names the places where energy has gathered away from the first volume: a
volume that holds more than each of the volumes round it (the 26), at least twice their
mean, and more than a volume holds when it is torn off (seven least whirls), so a fresh
volume at the rim is not counted. Twice is a reading number, not physics. `whole_whirls`
counts the volumes whose fluid winds its one whole turn in each of its three rings.
"""
import numpy as np

from physics.engine.axiom_3_structure.definitions.kinds import QUANTA_A_BOX
from physics.engine.axiom_3_structure.definitions.store import Store
from physics.engine.axiom_5_packing.logic.between import between, held_of
from physics.engine.axiom_6_limits.definitions.least_whirl import A_SIXTIETH, PARTS

TWICE = 2.0


def gatherings(block: np.ndarray) -> list:
    """(energy, steps from the middle in volumes) of each gathering in a block of volumes, fullest first."""
    wide, padded = block.shape[0], np.pad(block, 1)
    round_it = [padded[1 + one:1 + one + wide, 1 + two:1 + two + wide, 1 + three:1 + three + wide]
                for one in (-1, 0, 1) for two in (-1, 0, 1) for three in (-1, 0, 1) if (one, two, three) != (0, 0, 0)]
    mean = sum(round_it) / np.maximum(sum(one > 0 for one in round_it), 1)
    peak = (block > np.max(round_it, axis=0)) & (block >= TWICE * mean) & (block > PARTS / A_SIXTIETH)
    peak[(wide // 2,) * 3] = False  # the first volume is not away from itself
    return sorted(((float(block[tuple(at)]), tuple(int(one) for one in at - wide // 2))
                   for at in np.argwhere(peak)), reverse=True)


def apart_of_two(store: Store, held: np.ndarray, found: list) -> float:
    """How far apart the two fullest gatherings stand, in calm lengths through the cells between; 0 with fewer."""
    if len(found) < 2:
        return 0.0
    at = 2 * np.array([found[0][1], found[1][1]]) + store.side // 2
    return float(np.linalg.norm(between(held, at)[0, 1]))


def whole_whirls(store: Store) -> int:
    """How many own volumes wind one whole turn round each of their three cores."""
    own, across = store.own_cell.numpy()[:store.count], store.across
    slots = np.flatnonzero(store.cell_quanta.numpy()[own] > 0)
    phase = store.box_phase.numpy()[slots].astype(float).reshape(len(slots), 3, across // 3, across, across)
    fall = phase - np.roll(phase, -1, axis=3)  # from each box to the next round its core
    fall -= 2 * np.pi * np.floor((fall + np.pi) / (2 * np.pi))
    return int((np.rint(-fall.sum(axis=3) / (2 * np.pi)) == -1).all(axis=(1, 2, 3)).sum())


def census(store: Store) -> dict:
    """The universe in numbers: energies in calm volumes, distances in volumes from the first."""
    held, calm = held_of(store), QUANTA_A_BOX * store.across ** 3
    block, every = held[::2, ::2, ::2].clip(0), store.cell_quanta.numpy()
    middle, plus = block[(block.shape[0] // 2,) * 3], every[every > 0].sum() / calm
    others = np.sort(np.delete(block.ravel(), block.size // 2))[::-1]  # every volume but the first, fullest first
    found = gatherings(block)
    return {"turns": store.step * store.turn_per_step / (2 * np.pi), "volumes": int((block > 0).sum()),
            "layers": int(np.abs(np.argwhere(block > 0) - block.shape[0] // 2).max()),
            "first": float(middle), "share": float(middle / plus), "plus": float(plus),
            "minus": float(-every[every < 0].sum() / calm), "over_calm": int((others > 1).sum()),
            "biggest": [round(float(one), 3) for one in others[:4]], "gatherings": len(found),
            "fullest": [(round(energy, 3), at) for energy, at in found[:3]],
            "apart": apart_of_two(store, held, found), "collapses": store.collapses,
            "deepest": float(store.cell_pull.numpy().max())}


def line_of(seen: dict) -> str:
    """A census as one short line for the log."""
    return (f"turn {seen['turns']:.0f}  volumes {seen['volumes']}  layers {seen['layers']}  first {seen['first']:.1f} "
            f"({100 * seen['share']:.2f}% of plus {seen['plus']:.1f})  minus {seen['minus']:.0f}  over a calm volume "
            f"{seen['over_calm']}  biggest {seen['biggest']}  gatherings {seen['gatherings']} {seen['fullest']} "
            f"apart {seen['apart']:.1f}  collapses {seen['collapses']}  deepest {seen['deepest']:.0f}")
