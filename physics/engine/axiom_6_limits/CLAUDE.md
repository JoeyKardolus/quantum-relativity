# Axiom 6: limits

The owner's words (`docs/bible.md#physicsquantum-relativity`, "The axioms", 6):

> Limits: a joint's energy rises only while it is squeezed, so a fresh joint is the fattest it will ever be and thins as it is squeezed; squeezed too far it collapses, its energy is taken off the rest, and its two volumes merge; overstretched plus tears, and only that cell splits.

They win over this file and the code.

## Clause by clause

- "rises only while it is squeezed": `make_law.py`, `logic/make_minus.py`.
- "squeezed too far it collapses": `collapse_law.py`; only a joint past its limit ever goes.
- "taken off the rest", "its two volumes merge": `remove_joint.py`, `take_off.py` (a shortfall is booked, `short_quanta`), `collapse.py`.
- "overstretched plus tears": `tear_law.py`, `tear_map.py`, `logic/tear.py`.

## Assumed

- A squeezed joint goes toward the level where its push equals the press, at `squeeze_rate` on its own clock.
- "Too far" is `collapse_thin` times the joint's own fresh quanta.
- "The rest" is the merged volume.
- A joint between more than two volumes collapses along its hardest way; a volume merged away earlier in the step is the one it fell into. The joints on its opposite sides then touch (`close_web`).
- A volume tears across its own swirl axis; the middle two layers flip.

## Real runs (16 a side, seed 1, 2026-11-18)

- Default: no collapse and no tear in 40 steps; joints squeezed by the fluid's first moves make 5.7 million quanta (largest squeeze 1.05).
- Collapsing (`collapse_thin` 1.01): 495 collapses by step 8, 1,874 by step 24, each of a joint past its limit; books exact.

## Not done

- **A collapse storm takes every plus quantum**: balanced joints hold three quarters of the energy and a collapse takes as much again off its volumes; the collapsing run ends empty (one expected failure).
- **A torn volume's side joints gain the new volume**, so the ways through them double.
- **A joint inside one merged volume is dead**: no way, so no press.
- **A tear cuts along `swirl_axis`, a label**: most torn volumes are no longer wound round it.

## Open: the tear chain

A torn piece holds a third, so by the size law it is three times longer and tears again. Uneven run (fix round 3): 5 tears in step 1, 316 by step 5, 3,024 by step 8, about doubling a step; 512 volumes become some 3,500, the emptiest a few quanta. Every tear turns a third of the volume's plus into minus. It ends only where a piece would hold nothing; such a piece stays far too long. No tear-made joint ever collapsed, so nothing puts pieces back. The owner's open question.
