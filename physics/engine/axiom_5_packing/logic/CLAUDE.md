# Axiom 5, logic

`squeeze.py`: one launch over the own slots writes each own joint's pull, push, its volumes' distance and its axis into `own_squeeze`, before anything acts on it. `press.py`: the squeeze turns the phase of the joint's facing boxes on, so its fluid is pushed inward.

- `squeeze` runs before `make` and `press`: both read `own_squeeze`, so a run repeats.
- The press and the making are one squeeze: both use `given` times `pressed`.
- Only the joint's phases change here. No press once the push answers the pull.
- Measured (6 across, 300 steps): the squeezed joint is 549 off its rule; one not squeezed stays under 1.
