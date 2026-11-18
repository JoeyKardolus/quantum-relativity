"""Axiom 1, energy, definitions: what things are under this axiom.

The state it owns, its settings and its laws as formulas. Definitions import only
definitions and never logic. Import a name from its own file; this file holds nothing.
  energy_arrays.py        ARRAYS, HOST: quanta, cell_quanta, the step's ledgers, the books' counts
  energy_settings.py      EnergySettings: step_time, the start's energies
  pair_rotation.py        pair_rotation: the exact step of two waves trading fluid
  whole_move.py           whole_move: a flow as whole quanta, cut at 0
  cell_sum.py             cell_sum: the exact sum of one cell's own boxes
  passes.py               the pass numbers of the draws
  draw_uniform.py         draw_uniform, draw_uniform_host, draw_uniform_rows: the engine's one source of random numbers
  whole_shares.py         whole_shares: whole quanta dealt in proportion, exactly
  book_law.py             book_law: what the totals must be
"""
