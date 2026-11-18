"""Axiom 3, structure, definitions: what things are under this axiom.

The state it owns, its settings and its laws as formulas. Definitions import only
definitions and never logic. Import a name from its own file; this file holds nothing.
  kinds.py                ENERGY_VOLUME, JOINT, NO_CELL, BLOCK, BOXES
  direction_codes.py      CENTRE, opposite, step_of and the host twins, code_of_host
  structure_arrays.py     ARRAYS, HOST: kind, length, the relation tables
  structure_settings.py   StructureSettings: size_constant
  size_law.py             size_law, size_law_host: length = constant / quanta
  neighbour_of.py         neighbour_count, neighbour_id, neighbour_code
  grid.py                 Grid: the one state object
"""
