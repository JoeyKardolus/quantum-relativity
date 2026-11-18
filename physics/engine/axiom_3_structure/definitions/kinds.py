"""Kinds: the two kinds of cell, and how many compute boxes a cell needs.

Plus quanta make a cell an energy volume, minus quanta a joint: the sign of its total
is the kind, so nothing else stores it. `ACROSS` is the least number of Planck boxes
along a cell's edge that holds a proper swirl. The boxes are not space (axiom 3): they
sit inside the cell only to compute its fluid.
"""
ACROSS = 6
# A scene number, chosen once and not moved: enough quanta for a swirl to outlast the noise
# of whole quanta. Measured at 6 across: 3,000 a box holds, 20 a box is gone in 100 steps.
QUANTA_A_BOX = 3_000
