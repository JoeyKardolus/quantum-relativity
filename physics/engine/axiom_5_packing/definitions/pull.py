"""Pull: one number a cell, handed on from cell to cell.

The law, two lines. A volume's pull is its own plus energy, counted in calm volumes (more
packed, stronger), and a sixth of the pull of the joint at each of its six sides; a side
with no joint gives none. A joint's pull is the mean of its two volumes': it carries the
pull and adds none. Both read the pull as it was a step ago, so it goes one neighbour a
step and no faster; it fades with distance because every volume hands it on in six parts.
The law has no setting.

What it does to energy: it adds to the flow of the fluid it acts on, toward the deeper side
(`logic/fall.py`), and squeezes the joints (`pull_push.py`).
"""
SIDES = 6.0  # a volume's sides: no number of its own

