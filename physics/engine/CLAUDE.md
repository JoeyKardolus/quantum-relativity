# physics/engine

Starts: `first_volume()` one volume, room for layers; `three_cells()` a volume, a joint, a volume; `one_cell()`; `calm_cells()`; `ball()` the grown ball, kept; `hole()` it with its middle packed; `calm_line()`, `calm_block()` benches for tests. `step(store)`: tear, collapse, flow, turn, cross, leak, pull, squeeze, make, press.

A directory per axiom; 7 waits.

- The kernels run only over own cells. Tears, collapses and the pull are done on the host.
- CPU and card run nearly alike, not box for box; tests use the CPU.
- Gaps, as strict xfails: in the grown ball fuller regions do not draw together, hold or coast; no orbit; no hole by gathering; 84 of 111 volumes are whole whirls.
