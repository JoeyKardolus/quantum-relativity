"""Read sent: what each volume's fluid sent into each other volume in this step's crossing.

Contract: `read_sent(state)` returns host int64 columns, one row per (giving volume,
taking volume) that something crossed between this step, in rising order of `key`
(`giver * 2 ** 32 + taker`): `key`, `giver`, `taker`, `sent` (whole quanta, every way
between the two added up) and `lost` (those of them whose direction the joints'
friction took). It reads `way_out_a`, `way_out_b`, `way_lost_a`, `way_lost_b` and the
way table, so it is right only between `cross` and the step's events. It writes nothing.
"""
from __future__ import annotations

import numpy as np

_SOURCE_BITS = 32


def read_sent(state) -> dict:
    """This step's crossings, one row per giving and taking volume."""
    living = state.host("way_joint") >= 0
    way_a, way_b = state.host("way_a")[living].astype(np.int64), state.host("way_b")[living].astype(np.int64)
    sent = np.concatenate([state.host("way_out_a")[living], state.host("way_out_b")[living]]).astype(np.int64)
    lost = np.concatenate([state.host("way_lost_a")[living], state.host("way_lost_b")[living]]).astype(np.int64)
    key = (np.concatenate([way_a, way_b]) << _SOURCE_BITS) | np.concatenate([way_b, way_a])
    keys, row_of = np.unique(key[sent > 0], return_inverse=True)
    rows = {"key": keys, "giver": keys >> _SOURCE_BITS, "taker": keys & ((1 << _SOURCE_BITS) - 1),
            "sent": np.zeros(len(keys), dtype=np.int64), "lost": np.zeros(len(keys), dtype=np.int64)}
    np.add.at(rows["sent"], row_of, sent[sent > 0])
    np.add.at(rows["lost"], row_of, lost[sent > 0])
    return rows
