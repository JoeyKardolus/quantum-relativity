# Axiom 4: crossing, definitions

What things are under this axiom. Each file's docstring is its contract.

- `crossing_arrays.py`: what each way carried this step, what each joint let through and heated, and `heat_quanta`, the heat each cell holds.
- `crossing_settings.py`: `friction`, the one number the axiom leaves open.
- `mouth.py`: the boxes of a volume's block that look out toward one neighbour, and which box faces which across a joint.
- `crossing_clock.py`: a crossing runs on the slowest of its three clocks.
- `friction_law.py`: how many crossing quanta lose their direction, how many were heat already, how much new heat that is.
- `arrival_law.py`: the phase of a box that takes quanta: what kept its direction goes on as the step left it, what lost it arrives at rest.

## Must not get wrong

- **There is no law of leaving here.** What leaves a cell is `pair_rotation` (axiom 1) between two boxes that face each other across a joint: the fluid's own step, nothing else. A separate law for a cell's side made motion out of nothing; do not write one.
- **A box never changes cell and belongs to one cell.** A facing pair only says which fluid of one cell trades with which fluid of the other. Boxes are not space.
- **Friction takes direction, never quanta.** Every quantum that crosses arrives. `friction_law` only says how many arrive at rest.
- **Friction never makes motion.** In `arrival_law` the more quanta lost their direction, the less the taking box turns; with all of them lost it turns least. A test pins the order.
- **No angle is drawn.** Where two fluids cancel, the box keeps its standing phase.
- **Heat is a count per cell.** A box has one phase for all its quanta, so heat moves with the rest of the fluid; `heat_leaving` gives a cell's share, never more than it has.
- **No rate setting, no seal, no off switch.** How much crosses is set by the fluid and the clocks.
- Definitions import only definitions, of axioms 3, 2, 1 and 7 and this one.
