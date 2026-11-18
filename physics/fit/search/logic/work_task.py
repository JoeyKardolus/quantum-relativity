"""Work task: in a worker process, run one task and send its record back.

Contract: `work_task(task, size, seconds, device, pipe)` runs `run_universe` for the
task's overrides and seed at `size` (cells, steps, every) and sends the record through
`pipe`, then closes it. A run that raises is a result: its record has score 0, the error
and the end of its traceback, never nothing. A score that is not a finite number is
recorded as 0 with an error saying so.
"""
from __future__ import annotations

import math
import traceback

from physics.fit.search.logic.run_universe import run_universe

_NO_SCORE = {"score": 0.0, "E": 0.0, "S": 0.0}


def _record(task: dict, size: tuple, seconds, device: str) -> dict:
    """The task's record: the run's own, or score 0 with the error when it raised or left the numbers' range."""
    try:
        record = run_universe(task["overrides"], task["seed"], *size, seconds=seconds, device=device)
    except Exception as error:  # a failed run is a result: recorded with its error, never hidden
        return {"overrides": task["overrides"], "seed": task["seed"], **_NO_SCORE, "error": repr(error),
                "trace": traceback.format_exc()[-800:]}
    if not math.isfinite(record["score"]):
        record.update(**_NO_SCORE, error="the score is not a number: the run left the numbers' range")
    return record


def work_task(task: dict, size: tuple, seconds, device: str, pipe) -> None:
    """Run one task and send its record through the pipe."""
    pipe.send(_record(task, size, seconds, device))
    pipe.close()
