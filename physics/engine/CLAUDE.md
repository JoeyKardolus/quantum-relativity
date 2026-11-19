# physics/engine

Starts: `first_volume()` one volume, room for layers; `three_cells()` a volume, a joint, a volume (a bench: nothing gathers outside it); `one_cell()`; `calm_cells()`; `hole()` a grown patch, its middle packed. `step(store)`: tear, collapse, flow, turn, cross, leak, spread, squeeze, make, press.

One directory per axiom. Axioms 1 to 6 hold code; 7 waits.

- The kernels run only over own cells. Tears and collapses are found on the host.
- CPU and card run nearly alike, not box for box; tests use the CPU.
- Gaps, as strict xfails: a fuller region does not gain, two do not draw together, one set going does not travel; not every volume is a whole whirl.
