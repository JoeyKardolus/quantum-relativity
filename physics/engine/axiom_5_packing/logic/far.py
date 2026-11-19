"""Far: the grow scene's universe run far forward, a line to a log every so many turns and its state kept.

No rule is changed to go far: the same `step`, on the card, the pull read by groups past
the `NEAR` groups round a volume (`definitions/groups.py`; what that leaves out is measured
in `tests/test_groups.py`). Every `every` turns of a calm swirl a census line goes to
`far.log` in the folder; every `keep_every` the state is kept as `turn_<n>.npz`, with the
whole whirls counted; `resumed` reads one back, for the window (`--from`) or to run on.
The run stops after `minutes` or at `turns` and keeps `last.npz`; when the store's room is
full the universe moves into one `MORE` layers roomier (`roomier.py`) and goes on. `ROOM`,
`MORE` and `NEAR` are run settings chosen before the run, not physics.
"""
import time
from pathlib import Path

import numpy as np

from physics.engine import first_volume, step
from physics.engine.axiom_3_structure.definitions.store import Store
from physics.engine.axiom_3_structure.logic.kept import keep, kept
from physics.engine.axiom_3_structure.logic.roomier import roomier
from physics.engine.axiom_5_packing.logic.census import census, line_of, whole_whirls

ROOM, MORE, NEAR = 14, 6, 2  # layers the store has room for, and more when full; groups read volume by volume


def resumed(path: Path, device: str = "cpu", more: int = 0) -> Store:
    """The universe kept at `path`, on `device`, in a store with the room it was kept with and `more` layers."""
    with np.load(path) as held:
        across, side, near = int(held["across"]), int(held["side"]), int(held["near"])
    store = kept(path, first_volume(across, device, side // 4, near=near))
    return roomier(store, side // 4 + more) if more else store


def far(folder: Path, minutes: float, device: str = "cpu", turns: float = np.inf, every: int = 50,
        keep_every: int = 500, start: Store = None) -> Store:
    """Run the universe (from `start`, or from one first volume) and log it; gives the store as it ended."""
    store = start or first_volume(device=device, layers=ROOM, near=NEAR)
    a_turn, until = 2 * np.pi / store.turn_per_step, time.monotonic() + 60 * minutes
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "far.log").open("a") as log:
        def say(words: str) -> None:
            log.write(words + "\n")
            log.flush()
        while time.monotonic() < until and store.step < turns * a_turn:
            if store.step % int(every * a_turn) == 0:
                say(line_of(census(store)))
            if store.step % int(keep_every * a_turn) == 0 and store.step:
                keep(store, folder / f"turn_{store.step / a_turn:05.0f}.npz")
                say(f"kept at turn {store.step / a_turn:.0f}: {whole_whirls(store)} whole whirls")
            try:
                step(store)
            except RuntimeError as full:  # the room is full, found before the step changed anything
                store = roomier(store, store.side // 4 + MORE)
                say(f"{full}; moved into room for {store.side // 4}")
        keep(store, folder / "last.npz")
        say(line_of(census(store)) + f"\nended at step {store.step}: {whole_whirls(store)} whole whirls, {store.count} cells")
    return store
