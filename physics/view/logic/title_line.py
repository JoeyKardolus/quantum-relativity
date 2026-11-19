"""Title line: a scene's numbers in words, for the window's title and the pictures' log."""
import numpy as np

from physics.engine.axiom_3_structure.definitions.kinds import QUANTA_A_BOX
from physics.engine.axiom_3_structure.definitions.places import steps_of
from physics.engine.axiom_3_structure.definitions.store import Store, peek, slot_of
from physics.engine.axiom_3_structure.logic.boxes_of import swirl_of


def title_line(store: Store) -> str:
    """Whole turns about each direction, total quanta, step, core, distance from the rule, the ledger, own or calm."""
    seen = swirl_of(store, 0)
    turns = "mixed" if None in seen["turns"] else " ".join(map(str, seen["turns"]))
    kept = "own" if slot_of(store, 0) >= 0 else "calm"
    return (f"whole turns {turns}  quanta {peek(store.cell_quanta, 0):,}  step {store.step}  "
            f"core {seen['core']:.3f} of the mean  off the rule {seen['off_rule']:.2f}  "
            f"ledger {store.ledger:,}  {kept}")


def _sizes(held: np.ndarray, calm: int) -> str:
    """The cells' sizes in calm volumes: one by one when few, else the least and the most."""
    sizes = calm / held
    if sizes.size > 3:
        return f"sizes {sizes.min():.2g} to {sizes.max():.2g}"
    return "size " + " ".join(f"{one:.3g}" for one in sizes)


def cells_title(store: Store) -> str:
    """Layers, volumes and joints with their sizes; plus and minus to the quantum; what joints made; the step."""
    held = store.cell_quanta.numpy().astype(np.int64)
    own = store.own_cell.numpy()[:store.count]
    calm, has = QUANTA_A_BOX * store.across ** 3, held[own]
    volumes, joints = has[has > 0], -has[has < 0]
    out = np.abs(steps_of(own[has > 0], store.side) - store.side // 2).max() // 2
    layers = f"{out} layers  " if store.leaks else ""
    blue = f"  {joints.size} joints {_sizes(joints, calm)}" if joints.size else "  no joint"
    return (f"{layers}{volumes.size} volumes {_sizes(volumes, calm)}{blue}  "
            f"plus {held[held > 0].sum():,}  minus {joints.sum():,}  "
            f"made {store.cell_made.numpy().sum():,}  step {store.step}")
