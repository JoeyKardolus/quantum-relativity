"""Rebuild tables: lay the contact and link tables out anew, from given pairs or from the boxes.

Contract: `rebuild_tables(universe, pairs=None)` replaces universe.neighbour, faces, slots
(contact_table.py) and links, link_slots (link_table.py) with tables for all
universe.cell_count cells, rows widened when a cell touches more cells than a row holds.
`pairs` is the (low, high, faces) triple of every touching pair (contact_pairs.py's
form); without it the pairs are counted from the box owners of the active region.
Joint collapse and volume split pass the pairs they worked out from the old table
(table_pairs.py, merged_pairs.py); box moves keep the tables themselves. Returns None.
"""
from __future__ import annotations

import warp as wp

from physics.potts.contact_pairs import contact_pairs
from physics.potts.contact_table import contact_table
from physics.potts.link_table import link_table
from physics.potts.region_host import region_host
from physics.potts.universe import Universe


def rebuild_tables(universe: Universe, pairs: tuple | None = None) -> None:
    """Lay out the contact and link tables from the pairs (counted from the boxes when not given)."""
    if pairs is None:
        pairs = contact_pairs(region_host(universe, universe.owner))
    device = universe.owner.device
    neighbour, faces, slots = contact_table(None, universe.cell_count, universe.slots, pairs)
    links, link_slots = link_table(None, universe.cell_count, universe.link_slots, pairs)
    universe.neighbour = wp.array(neighbour, dtype=wp.int32, device=device)
    universe.faces = wp.array(faces, dtype=wp.int32, device=device)
    universe.slots = slots
    universe.links = wp.array(links, dtype=wp.int32, device=device)
    universe.link_slots = link_slots
