"""Title line: a scene's numbers in words, for the window's title and the pictures' log."""
import numpy as np

from physics.engine.axiom_3_structure.definitions.kinds import QUANTA_A_BOX
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
        return f"sizes {sizes.min():.1f} to {sizes.max():.1f}"
    return "size " + " ".join(f"{one:.2f}" for one in sizes)


def cells_title(store: Store) -> str:
    """How many volumes, joints and tears; plus and minus to the quantum; sizes; what joints made; the step."""
    held = store.cell_quanta.numpy()[:store.cells].astype(np.int64)
    calm = QUANTA_A_BOX * store.across ** 3
    volumes, joints = held[held > 0], -held[held < 0]
    blue = f"  {joints.size} joints {_sizes(joints, calm)}" if joints.size else "  no joint"
    return (f"{volumes.size} volumes {_sizes(volumes, calm)}{blue}  {store.tears} tears  "
            f"plus {volumes.sum():,}  minus {joints.sum():,}  "
            f"made {store.cell_made.numpy().sum():,}  step {store.step}")
