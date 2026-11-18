"""Title line: the one cell's numbers in words, for the window's title and the pictures' log."""
from physics.engine.axiom_3_structure.definitions.store import Store, peek, slot_of
from physics.engine.axiom_3_structure.logic.boxes_of import swirl_of


def title_line(store: Store) -> str:
    """Step, total quanta, whole turns, core, distance from the rule, the ledger, own or calm."""
    seen = swirl_of(store, 0)
    turns = "mixed" if seen["turns"] is None else seen["turns"]
    kept = "own" if slot_of(store, 0) >= 0 else "calm"
    return (f"step {store.step}  quanta {peek(store.cell_quanta, 0):,}  whole turns {turns}  "
            f"core {seen['core']:.3f} of the mean  off the rule {seen['off_rule']:.2f}  "
            f"ledger {store.ledger:,}  {kept}")
