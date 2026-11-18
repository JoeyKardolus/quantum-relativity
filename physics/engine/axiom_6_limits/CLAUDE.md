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

## Real runs (16 a side, seed 1)

- Default: 1,928 collapses in 9 steps, each of a joint past its limit; about three in four had no two volumes left to merge and only went. 512 volumes become 4. No short take; books exact.
- Held settings, 8 a side, 12,000 steps: a lump gathers until its joints are squeezed: first making at step 1,280; the fattest joint is 1.34, 1.46, 1.92 times fresh at steps 4,000, 6,000, 10,000; the first reaches its limit and collapses at step 11,590. By step 12,000: 52 collapses, 13 volumes, one holding 38.2 of 41.7 million plus quanta; made 4.68 million, taken 3.42 million.

## Not done

- **At the default strengths no joint thins step by step**: the start's largest squeeze is 1.4 million and the joints are past the limit at once.
- **A torn volume's side joints gain the new volume**, so the ways through them double.
- **A joint inside one merged volume is dead**: no way, so no press.
- **A tear cuts along `swirl_axis`, a label**: most torn volumes are no longer wound round it.

## Open: the tear chain

A torn piece holds a third, so by the size law it is three times longer and tears again. Uneven run (fix round 3): 5 tears in step 1, 316 by step 5, 3,024 by step 8, about doubling a step; 512 volumes become some 3,500, the emptiest a few quanta. Every tear turns a third of the volume's plus into minus. It ends only where a piece would hold nothing; such a piece stays far too long. No tear-made joint ever collapsed, so nothing puts pieces back. The owner's open question.
