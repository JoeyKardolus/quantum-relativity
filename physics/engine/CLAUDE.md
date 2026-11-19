# physics/engine

Starts: `first_volume()` one volume, room for layers; `three_cells()` a volume, a joint, a volume (a bench: nothing gathers outside it); `one_cell()`; `calm_cells()`; `hole()` a grown patch, its middle packed; `calm_line()`, `calm_block()` the benches. `step(store)`: tear, collapse, flow, turn, cross, leak, pull, squeeze, make, press.

A directory per axiom; 7 waits.

- The kernels run only over own cells. Tears, collapses and the pull are done on the host.
- CPU and card run nearly alike, not box for box; tests use the CPU.
- Gaps, as strict xfails: grown, the first volume pours out from 17 turns; no coasting, no orbit; no hole by gathering; most volumes are no whole whirls.
