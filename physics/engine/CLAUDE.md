# physics/engine

`calm_grid()` (`axiom_3_structure/logic/lay_grid.py`) lays a grid that wraps round, every cell own. `three_cells()`: a volume, a fresh joint, a volume (a ring whose second joint is empty). `one_cell()` is layer 1's; `calm_cells()` lays rules. `step(store)` moves the fluid, lets it cross, reads the squeeze, lets squeezed joints make quanta and presses their fluid.

One directory per axiom. Axioms 1 to 6 hold code; 7 waits.

- A step runs only over own cells. Nothing is copied to the host inside a step.
- CPU and card run alike. Tests use the CPU.
- The same scene runs the same course at 6, 12 and 24 boxes across, in turns of a calm swirl (tested).
- Measured on the card, 6 boxes: 64 volumes 1,735 steps a second, 512 volumes 1,141.
