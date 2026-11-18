"""Structure parts: the four gains in structure since the start, each 0 for a loss.

Contract: `structure_parts(first, last, held_share)` takes the first and the last sample
(`measure_cells` dicts) and the run's held share and returns
  Sr  gain(rise of region_contrast, REGION_SCALE)   unevenness over regions grew
  Sl  gain(rise of lump_share, LUMP_SCALE)          more of the energy sits in lumps
  Sh  gain(held_share - the start's lump_share, HELD_SCALE)   lumps that hold: energy
      in lumps two thirds through is still in lumps at the end
  Sp  gain(rise of lump_spread, SPREAD_SCALE) * Sl  the lump volumes' sizes spread out,
      counted only as far as energy gathered into lumps (a stand-in for a web: no
      measure of threads is built)
There is no part for sealing: the engine has no seal.
"""
from __future__ import annotations

from physics.fit.score.definitions.gain import gain
from physics.fit.score.definitions.score_marks import HELD_SCALE, LUMP_SCALE, REGION_SCALE, SPREAD_SCALE


def structure_parts(first: dict, last: dict, held_share: float) -> dict:
    """Sr, Sl, Sh and Sp from the first and last sample and the held share."""
    gathered = gain(last["lump_share"] - first["lump_share"], LUMP_SCALE)
    return {"Sr": gain(last["region_contrast"] - first["region_contrast"], REGION_SCALE),
            "Sl": gathered,
            "Sh": gain(held_share - first["lump_share"], HELD_SCALE),
            "Sp": gain(last["lump_spread"] - first["lump_spread"], SPREAD_SCALE) * gathered}
