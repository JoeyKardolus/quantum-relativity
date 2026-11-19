# physics/engine

Starts (`axiom_3_structure/logic/starts.py`): `first_volume()` one overstretched volume, no joint; `three_cells()` a volume, a joint, a volume; `one_cell()`; `calm_cells()` lays rules. `step(store)`: tear, flow, cross, squeeze, make, press.

One directory per axiom. Axioms 1 to 6 hold code; 7 waits.

- The kernels run only over own cells. The tear alone is found on the host, from the cells' totals, once a step.
- CPU and card run alike. Tests use the CPU.
- A line of cells runs the same course at 6, 12 and 24 boxes across (tested). A growing cluster does not: the fluid's noise picks where the first volumes tear.
