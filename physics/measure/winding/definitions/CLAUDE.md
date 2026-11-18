# Measure: winding, definitions

What things are. Each file's docstring is its contract.

- `in_plane.py`: the two axes a smallest loop (a unit square of four boxes) lies in, and
  the order it is walked: right-handed about its normal.
- `phase_step.py`: the phase change from one box to the next, taken the short way round.
- `expected_turns.py`: the turns a cell of the engine reads while it is one whole swirl:
  minus its sense about axes 0 and 2, plus its sense about axis 1.

## Must not get wrong

- **Every sign in this measure comes from `in_plane`.** Change it and `expected_turns`
  changes with it.
- **The short way is what makes a winding whole.** A step of exactly half a turn has no
  short way; the test blocks are wound off-middle to stay clear of it.
- **A box is a complex number**: its amount (the square root of its quanta, sign dropped)
  times exp(i phase). Zero means no phase.
- **`expected_turns` copies a rule of the engine** (`swirl_phase.py`, axiom 2). If the
  engine winds its start another way, `physics/measure/tests` goes red; fix the table, not the test.
- Definitions import only definitions, and nothing from the engine.
