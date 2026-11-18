# Axiom 5: packing

The owner's words (`docs/bible.md#physicsquantum-relativity`, "The axioms", 5):

> Packing: more packed is smaller, slower and pulls harder; pull spreads at a finite speed, weaker with distance; squeezed minus pushes back steeply.

They win over this file and the code.

## Clause by clause

- "more packed is smaller": `size_law` (axiom 3), `packing_law.py`.
- "slower": `clock_rate.py`, `logic/tick_clocks.py`.
- "pulls harder": `pull_source.py`. `logic/pull_fluid.py` is where plus energy feels the pull (`inside_pull.py`).
- "spreads at a finite speed, weaker with distance": `pull_wave_law.py`, `heard_pull.py`, `logic/spread_pull.py`, `logic/start_pull.py`.
- "squeezed minus pushes back steeply": `push_law.py` (quanta to the fourth power), `press_law.py`, `logic/press_joints.py`.

## Assumed

- Packing is quanta over length cubed; one clock law for both kinds. The pull is on no cell's clock.
- The pull's source is the cell's quanta over the room it stands in (its mean distance to its neighbours, cubed), and a neighbour counts by its distance (2026-11-18). So what leaves a cell is its energy, two cells pull each other equally hard, and a fuller volume sits deeper. Before: the cell's own packing, quanta to the fourth power.
- The pull at a cell's side is its neighbours' mean there; `pull_fluid` turns a cell's fluid by its own pull and the slope across it.
- A way's press is the pull's slope from the joint down into its two volumes (2026-11-18). A joint's press is its hardest way; the start lays every joint at the energy where its push equals it (`balance_level`).

## Measured (16 a side, seed 1, 2026-11-18)

- Default start: every joint's squeeze between 0.9996 and 1; a joint 22 long, a volume 10. No collapse in 40 steps (largest squeeze 1.05).
- One volume made a tenth heavier, the pull's wave alone: half the change is 1 to 4 neighbours away at steps 225, 294, 445, 524 and a hundredth at 102, 158, 261, 333; light at 65, 130, 194, 259.
- No pull turns a box by half a turn a step in any run.

## Not done

- **The wave is slower than light, not at it**: every neighbour counts as one reach away, so along a row the bulk runs at 0.59 of `light_speed`. A trace below a hundredth still goes one neighbour a step.
- **Changes ring**: nothing damps the pull's wave.
- **A slow clock gathers energy by itself**, pull off: fluid moves slower through a packed cell and piles up; and the pull's slope acts on the cell's local time, so a packed lump falls slower. Two run tests are expected failures.
- **The push is a number, not a push** (axiom 2).

## Open

- **Balance cannot give thin joints.** A thinner joint is a fuller one: at the defaults a joint is 2.2 times a volume's length and the joints hold 76% of the energy.
- **The pull's whole turn of a cell is on counter time**, not the cell's clock, so the sinking level all cells share does nothing.
