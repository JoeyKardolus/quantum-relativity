# Axiom 6: limits, logic

What happens in a step under this axiom. Each file's docstring is its contract.

- `start_fresh.py`: each start joint remembers what it was made with.
- `make_minus.py`: a squeezed joint makes whole minus quanta in its own boxes.
- `find_limits.py`: marks joints past their limit and volumes that tear.
- `run_events.py`: carries the marked events out on the host and returns the cells whose relations changed; the step hands them to `relink` (axiom 3).
- `collapse.py` with `remove_joint.py`, `close_web.py`, `take_off.py`; `tear.py`; `hold_cell.py`, `cell_value.py`, `set_value.py`.

## Must not get wrong

- **Only a joint past its limit goes.** There is no `swallow`: the old rule removed every joint a merged volume enclosed, squeezed or not, and took its energy off the volume a second time. That was 86% of all collapses and emptied a universe.
- **No flagged joint is skipped.** A joint past its limit collapses in that step even when an earlier event touched it.
- **The way table is stale inside a step.** `collapse` follows each volume of the hardest way through `fell_into` to the volume it is now; when the two are one, the next hardest way. Never fall back on the lowest ids.
- **When a joint goes, only joints gain a relation**: the ones on its opposite sides touch (`close_web`). No volume is handed a joint to fill a side.
- **A short take is booked whole**: `short_takes` and `short_quanta`. `book_law`'s `taken` must stay 0.
- **A tear cuts across the volume's own swirl axis** and the new joint wraps the whole torn side of both pieces (nine codes each).
- **Per-cell values an event sets go through `set_value`** and are read with `cell_value`, so later events of the step see them.
- `make_minus` gives nothing to a box that holds nothing.

## Tests

The collapse and tear tests use a real 4- or 8-a-side universe; `tests/conftest.py` stages a small lattice for `start_fresh`.
