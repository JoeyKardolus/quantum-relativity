"""Write record: one run's record, appended to the results file as it ends.

Contract: `write_record(results, task, record)` adds the task's `phase`, `id` and `unit`
to the record, appends it to `results` as one JSON line and prints one progress line
(id, score, wall seconds, error). Nothing already in the file is touched.
"""
from __future__ import annotations

import json


def write_record(results, task: dict, record: dict) -> None:
    """Append one run's record with its task's phase, id and unit point."""
    record.update(phase=task["phase"], id=task["id"], unit=task["unit"])
    with results.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    print(task["id"], round(record["score"], 4), round(record.get("seconds", 0.0)), record.get("error", ""), flush=True)
