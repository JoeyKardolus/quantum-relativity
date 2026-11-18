# The search, logic

What happens in a search. Each file's docstring is its contract.

- `run_universe.py`: one universe stepped, sampled on the observer's clock and scored.
- `work_task.py`: in a worker process, one task run and its record sent back.
- `run_tasks.py`: the tasks not yet done, a process each, several side by side.
- `load_records.py`, `write_record.py`: the results file, one JSON line a run.
- `find_holders.py`: which points still stand when run four times longer.
- `run_search.py`: the phases in order.
- `summarize_search.py`: the summary text.

## Must not get wrong

- **Every override is a Settings field.** The start's energies are settings now; there
  are no separate start arguments.
- **A universe that is over is ended early** and scored on its samples so far; with
  fewer than four it scores 0 and says why.
- **The only `except` is in `work_task`**, and it records the error. Do not add one
  that hides a failure.
- **Workers are spawned**, so `work_task` must stay importable by name.
- **The hold tasks are worked out again from the records** at every stage; ids already
  in the file are skipped, so asking twice costs nothing.
- **The engine is deterministic**: the same settings, seed and size give the same
  series. A test checks it.
