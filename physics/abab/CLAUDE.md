# physics/abab/

The owner's foundation (bible `physics/quantum-relativity`, "The foundation"): an irregular ABAB
structure. A = energy volume (pulls), B = joint (pushes). Module map in `__init__.py`.

## Must not get wrong

- **Physics reads only kind, energy, size, rate and contacts.** `new_state` returns positions beside
  the State for drawing; nothing in a step may read them.
- **ABAB always**: every contact row is (energy volume, joint). Collapse and split keep it; both
  tests check it.
- **Energy moves only A -> B -> A** (`passages.py`, `leak.py`); a joint is the passage, its own
  energy changes only when it collapses. Openness falls to 0 at `seal_point` (`joint_openness.py`).
- **Joint swing undamped**: `swing_joints.py` (moved from the old cells engine) on the joint's size against
  `joint_rest_size(energy)`. The pull's conductance reads only energy-volume sizes (`2 / l_a` per
  contact): a joint's own size in it pumps a lone swing (measured, 1% in 2000 steps).
- **Split**: the new joint is priced at rest (`joint_energy.py`) at 3% of the parent's size; a
  parent too dilute to pay does not split and shows in `stats()["too_big"]`.
- **Honesty.** Today (seed 1, 2000 volumes, unevenness 0.5, defaults, 300 steps): 2903 energy
  volumes and joints, 0 collapses, 1903 splits, 79 too big, size 4.02 x start, 11 lumps, 0 sealed;
  13.6 ms per step.

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/abab
```
