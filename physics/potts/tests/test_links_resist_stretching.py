"""Links bend, never stretch: with the link rule on, links stretch less than with it off.

A link is the distance between two touching cells' centres; its rest is half of each
cell's width, n**(1/3) / 2. Side by side, 30 steps of the same uneven start with every
rule on, once with link_stiffness 1 (the default) and once with 0 (the rule off, as a
comparison run only): the mean stretch past rest over all links is smaller with the
rule on.
"""
import numpy as np

from physics.potts import Settings, contact_pairs, new_universe, stats, step
from physics.potts.tests._gpu import needs_gpu


def _mean_stretch(universe) -> float:
    stats(universe)
    boxes = universe.cell_boxes.numpy().astype(np.float64)
    centre = universe.cell_place.numpy().reshape(-1, 3) / np.maximum(boxes, 1)[:, None]
    low, high, _ = contact_pairs(universe.owner.numpy())
    length = np.linalg.norm(centre[low] - centre[high], axis=1)
    rest = 0.5 * (np.cbrt(boxes[low]) + np.cbrt(boxes[high]))
    return float(np.maximum(length / rest - 1.0, 0.0).mean())


@needs_gpu
def test_links_stretch_less_with_the_link_rule_on():
    stretch = {}
    for stiffness in (1.0, 0.0):
        universe = new_universe(6, 4, seed=1, unevenness=0.3)
        settings = Settings.all_on(link_stiffness=stiffness)
        for _ in range(30):
            step(universe, settings)
        stretch[stiffness] = _mean_stretch(universe)
    print(f"mean link stretch past rest: rule on {stretch[1.0]:.4f}, off {stretch[0.0]:.4f}")
    assert stretch[1.0] < stretch[0.0]
