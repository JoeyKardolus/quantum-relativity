"""Title line: a scene's numbers in words, for the window's title and the pictures' log."""
from physics.engine.axiom_3_structure.definitions.size_law import size_of
from physics.engine.axiom_3_structure.definitions.store import Store, peek, slot_of
from physics.engine.axiom_3_structure.logic.boxes_of import swirl_of


def title_line(store: Store) -> str:
    """Whole turns, total quanta, step, core, distance from the rule, the ledger, own or calm."""
    seen = swirl_of(store, 0)
    turns = "mixed" if seen["turns"] is None else seen["turns"]
    kept = "own" if slot_of(store, 0) >= 0 else "calm"
    return (f"whole turns {turns}  quanta {peek(store.cell_quanta, 0):,}  step {store.step}  "
            f"core {seen['core']:.3f} of the mean  off the rule {seen['off_rule']:.2f}  "
            f"ledger {store.ledger:,}  {kept}")


def line_title(store: Store) -> str:
    """Each cell's quanta and size, what the joint has made, the heat, all the energy, the step."""
    left, joint, right = (int(held) for held in store.cell_quanta.numpy()[:3])
    heat, made = store.cell_heat.numpy(), int(store.cell_made.numpy()[1])
    cells = "  ".join(f"{name} {held:,} size {size_of(held, store.across):.2f}"
                      for name, held in (("A", left), ("joint", joint), ("B", right)))
    return (f"{cells}  joint made {made:,}  heat A {heat[0]:,} B {heat[2]:,}  "
            f"plus and minus {left + right - joint:,}  step {store.step}")
