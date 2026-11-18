"""Title line: a scene's numbers in words, for the window's title and the pictures' log."""
import numpy as np

from physics.engine.axiom_3_structure.definitions.kinds import QUANTA_A_BOX
from physics.engine.axiom_3_structure.definitions.store import Store, peek, slot_of
from physics.engine.axiom_3_structure.logic.boxes_of import swirl_of
from physics.engine.axiom_5_packing.logic.squeeze import squeeze


def title_line(store: Store) -> str:
    """Whole turns, total quanta, step, core, distance from the rule, the ledger, own or calm."""
    seen = swirl_of(store, 0)
    turns = "mixed" if seen["turns"] is None else seen["turns"]
    kept = "own" if slot_of(store, 0) >= 0 else "calm"
    return (f"whole turns {turns}  quanta {peek(store.cell_quanta, 0):,}  step {store.step}  "
            f"core {seen['core']:.3f} of the mean  off the rule {seen['off_rule']:.2f}  "
            f"ledger {store.ledger:,}  {kept}")


def cells_title(store: Store) -> str:
    """The volumes' and joints' quanta, the joints' thinnest place, what they made, heat, step."""
    held = store.cell_quanta.numpy()
    calm = QUANTA_A_BOX * store.across ** 3
    squeeze(store)
    apart = store.own_squeeze.numpy()[:store.count, 2].max()  # the scenes' joints all agree
    volumes, joints = held[held > 0], -held[held < 0]
    thin = apart - calm * (1 / volumes).mean()  # less two radii: every joint has one of each
    return (f"volumes {' '.join(f'{one / calm:.2f}' for one in volumes)}  "
            f"joints {' '.join(f'{one / calm:.2f}' for one in joints)}  "
            f"{apart:.2f} apart, joint {thin:.2f} thin  made {store.cell_made.numpy().sum():,}  "
            f"heat {store.cell_heat.numpy().sum():,}  all {volumes.sum() + joints.sum():,}  "
            f"step {store.step}")
