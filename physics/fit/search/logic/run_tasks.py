"""Run tasks: the runs not yet done, a process per universe, several side by side on one GPU.

Contract: `run_tasks(results, tasks, size, workers, seconds, until, device)` runs every
task whose id is not yet in `results`, `workers` processes at a time; each universe
steps exactly as it would alone, and a run that hangs or dies takes nothing with it.
Every run is written to `results` as it ends (`write_record`). A process that gives no
answer within twice `seconds` is killed, and one that dies without an answer is noted:
both are recorded with score 0 and the reason, never dropped. After the time `until`
(seconds since the epoch; None: never) no new run is started: those under way finish.
Returns False when `until` cut the tasks short, True when all were run.
"""
from __future__ import annotations

import multiprocessing
import time

from physics.fit.search.logic.load_records import load_records
from physics.fit.search.logic.work_task import work_task
from physics.fit.search.logic.write_record import write_record

_DIED = "the process died without an answer"


def _lost(task: dict, why: str) -> dict:
    """The record of a run whose process gave no answer."""
    return {"overrides": task["overrides"], "seed": task["seed"], "score": 0.0, "E": 0.0, "S": 0.0, "error": why}


def _start(context, task: dict, size: tuple, seconds, device: str) -> tuple:
    """Start one task in its own process; the task, the process, its pipe and when it started."""
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=work_task, args=(task, size, seconds, device, sender), daemon=True)
    process.start()
    sender.close()
    return task, process, receiver, time.time()


def _answer(run: tuple, seconds) -> dict | None:
    """A run's record when it is over (its own, or why it was lost); None while it is still going."""
    task, process, receiver, started = run
    if receiver.poll(0.05):
        try:
            record = receiver.recv()
        except EOFError:
            record = _lost(task, _DIED)
        process.join()
        return record
    if not process.is_alive():
        return _lost(task, _DIED)
    if seconds is not None and time.time() - started > 2 * seconds:
        process.kill()
        process.join()
        return _lost(task, f"no answer within {2 * seconds} s; process stopped")
    return None


def run_tasks(results, tasks: list, size: tuple, workers: int, seconds, until, device: str) -> bool:
    """Run the tasks not yet in the results file; False when `until` cut them short."""
    done = {record["id"] for record in load_records(results)}
    waiting = [task for task in tasks if task["id"] not in done][::-1]
    context = multiprocessing.get_context("spawn")
    running, cut = [], False
    while waiting or running:
        if until is not None and time.time() > until and waiting:
            waiting, cut = [], True
        while waiting and len(running) < workers:
            running.append(_start(context, waiting.pop(), size, seconds, device))
        still = []
        for run in running:
            record = _answer(run, seconds)
            if record is None:
                still.append(run)
            else:
                write_record(results, run[0], record)
        running = still
    return not cut
