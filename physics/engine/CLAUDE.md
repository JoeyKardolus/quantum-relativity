# physics/engine

Starts: `first_volume()` one volume, room for layers; `three_cells()` a volume, a joint, a volume (a bench: nothing gathers outside it); `one_cell()`; `calm_cells()`; `hole()` a grown patch, its middle packed; `calm_line()` the line bench. `step(store)`: tear, collapse, flow, turn, cross, leak, spread, fall, squeeze, make, press.

A directory per axiom; 7 waits.

- The kernels run only over own cells. Tears and collapses are found on the host.
- CPU and card run nearly alike, not box for box; tests use the CPU.
- Gaps, as strict xfails: grown, the first volume empties, no fuller region holds, two do not draw together; no hole by gathering; half the volumes are no whole whirls.
